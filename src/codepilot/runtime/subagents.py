from __future__ import annotations

"""Plan-mode read-only exploration subagents."""

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from codepilot.core.contracts import (
    AgentLoopInput,
    AgentLoopLimits,
    AgentLoopPorts,
    RunCorrelation,
)
from codepilot.core.runner import run_agent_loop
from codepilot.llm.ports import ModelDescriptor, ModelPort
from codepilot.protocols import TextContent, UserMessage
from codepilot.sessions.subagents import SubagentStore
from codepilot.tools.contracts import (
    ToolCallRequest,
    ToolDefinition,
    ToolMetadata,
    ToolPort,
    ToolResult,
)
from codepilot.tools.restricted import DEFAULT_READ_ONLY_TOOL_NAMES, RestrictedToolPort

if TYPE_CHECKING:
    from .sessions import RuntimeSession


LIST_EXPLORATION_AGENTS_TOOL = "list_exploration_agents"
DISPATCH_EXPLORATION_TOOL = "dispatch_exploration"
MAX_TASKS_PER_BATCH = 4
DEFAULT_MAX_PARALLEL = 3
SUBAGENT_TIMEOUT_SECONDS = 90
REPORT_TEXT_LIMIT = 12000

_EXPECTED_OUTPUTS = {"architecture", "flow", "risk", "tests", "open"}
_REUSE_MODES = {"auto", "no_reuse", "force_refresh"}
_FAILURE_STATUSES = {"failed", "timeout", "invalid_output", "cancelled"}


@dataclass(frozen=True)
class ExplorationTask:
    task_id: str
    subagent_id: str
    purpose: str
    instruction: str
    scope_key: str
    focus_paths: list[str] = field(default_factory=list)
    expected_output: str = "open"
    critical: bool = False


@dataclass
class SubagentRunner:
    workspace: Path
    session_id: str
    model: ModelDescriptor
    model_port: ModelPort
    tool_port: ToolPort
    timeout_seconds: int = SUBAGENT_TIMEOUT_SECONDS

    async def run(
        self,
        task: ExplorationTask,
        *,
        peer_assignments: list[dict[str, Any]],
        previous_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        events: list[dict[str, Any]] = []
        loop_input = AgentLoopInput(
            run_id=f"subrun_{uuid4().hex[:12]}",
            correlation=RunCorrelation(session_id=self.session_id),
            messages=[UserMessage(content=_subagent_user_prompt(task, peer_assignments, previous_report))],
            user_prompt=task.instruction,
            context={"system_prompt": _subagent_system_prompt(self.workspace)},
            model=self.model,
            mode="read",
            limits=AgentLoopLimits(
                max_model_turns=4,
                max_tool_iterations=6,
                max_tool_calls_per_turn=4,
                max_tool_calls=16,
                repeated_tool_call_limit=3,
            ),
        )
        ports = AgentLoopPorts(
            model=self.model_port,
            tools=RestrictedToolPort(self.tool_port),
            context=None,
            events=events.append,
        )
        outcome = await asyncio.wait_for(
            run_agent_loop(loop_input, ports),
            timeout=self.timeout_seconds,
        )
        if outcome.status != "completed":
            return _failure_report(
                task,
                status="failed",
                error={
                    "code": f"subagent.{outcome.stop_reason}",
                    "message": outcome.final_text or str(outcome.error or outcome.stop_reason),
                },
                event_count=len(events),
            )
        parsed = _parse_json_object(outcome.final_text)
        if parsed is None:
            return _invalid_output_report(task, outcome.final_text, event_count=len(events))
        return _normalize_report(task, parsed, event_count=len(events))


@dataclass
class ExplorationCoordinator:
    workspace: Path
    session_id: str
    model: ModelDescriptor
    model_port: ModelPort | None
    tool_port: ToolPort | None
    store: SubagentStore
    runner_factory: Callable[[], SubagentRunner] | None = None

    async def dispatch(self, arguments: dict[str, Any]) -> dict[str, Any]:
        tasks = _parse_tasks(arguments)
        reuse = _reuse_mode(arguments.get("reuse"))
        max_parallel = _max_parallel(arguments.get("max_parallel"))
        warnings: list[str] = []
        reports: list[dict[str, Any]] = []
        created_ids: list[str] = []
        reused_ids: list[str] = []
        existing_ids = {
            str(item.get("subagent_id"))
            for item in self.store.list_agents()
            if item.get("subagent_id")
        }

        unique_tasks: list[ExplorationTask] = []
        seen_ids: set[str] = set()
        seen_scopes: set[str] = set()
        for task in tasks:
            duplicate_reason = ""
            if task.subagent_id in seen_ids:
                duplicate_reason = "duplicate_subagent_id"
            elif task.scope_key in seen_scopes:
                duplicate_reason = "duplicate_scope_key"
            if duplicate_reason:
                warnings.append(f"{task.task_id}: {duplicate_reason}")
                reports.append(_skipped_duplicate_report(task, duplicate_reason))
                continue
            seen_ids.add(task.subagent_id)
            seen_scopes.add(task.scope_key)
            unique_tasks.append(task)

        runnable: list[tuple[ExplorationTask, dict[str, Any] | None]] = []
        for task in unique_tasks:
            if task.subagent_id not in existing_ids:
                created_ids.append(task.subagent_id)
            cached = self._cached_report_for(task)
            if reuse == "auto" and cached is not None and not self.store.report_is_stale(cached):
                report = dict(cached)
                report["reused"] = True
                report["stale"] = False
                reports.append(report)
                reused_ids.append(task.subagent_id)
                continue
            previous = cached if reuse in {"auto", "force_refresh"} else None
            runnable.append((task, previous))

        if runnable:
            if self.model_port is None:
                reports.extend(
                    _failure_report(task, status="failed", error={"code": "subagent.missing_model_port"})
                    for task, _ in runnable
                )
            elif self.tool_port is None:
                reports.extend(
                    _failure_report(task, status="failed", error={"code": "subagent.missing_tool_port"})
                    for task, _ in runnable
                )
            else:
                reports.extend(
                    await self._run_parallel(
                        runnable,
                        max_parallel=max_parallel,
                        peer_assignments=[_task_summary(task) for task in unique_tasks],
                    )
                )

        for report in reports:
            if report.get("reused") or report.get("status") == "skipped_duplicate":
                continue
            task = next(
                (item for item in tasks if item.subagent_id == report.get("subagent_id")),
                None,
            )
            if task is None:
                continue
            stored = self.store.append_report(
                subagent_id=task.subagent_id,
                purpose=task.purpose,
                scope_key=task.scope_key,
                focus_paths=task.focus_paths,
                report=report,
                evidence_paths=_evidence_paths(report),
            )
            report["report_id"] = stored["report_id"]

        return {
            "batch_status": _batch_status(reports, tasks),
            "reports": reports,
            "warnings": warnings,
            "created_subagent_ids": sorted(set(created_ids)),
            "reused_subagent_ids": sorted(set(reused_ids)),
        }

    async def _run_parallel(
        self,
        tasks: list[tuple[ExplorationTask, dict[str, Any] | None]],
        *,
        max_parallel: int,
        peer_assignments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(max_parallel)

        async def run_one(task: ExplorationTask, previous: dict[str, Any] | None) -> dict[str, Any]:
            async with semaphore:
                try:
                    return await self._runner().run(
                        task,
                        peer_assignments=peer_assignments,
                        previous_report=previous,
                    )
                except asyncio.TimeoutError:
                    return _failure_report(task, status="timeout", error={"code": "subagent.timeout"})
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    return _failure_report(
                        task,
                        status="failed",
                        error={
                            "code": "subagent.exception",
                            "message": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    )

        return list(await asyncio.gather(*(run_one(task, previous) for task, previous in tasks)))

    def _runner(self) -> SubagentRunner:
        if self.runner_factory is not None:
            return self.runner_factory()
        if self.model_port is None or self.tool_port is None:
            raise RuntimeError("Subagent runner requires model and tool ports")
        return SubagentRunner(
            workspace=self.workspace,
            session_id=self.session_id,
            model=self.model,
            model_port=self.model_port,
            tool_port=self.tool_port,
        )

    def _cached_report_for(self, task: ExplorationTask) -> dict[str, Any] | None:
        direct = self.store.latest_report(task.subagent_id)
        if direct is not None:
            return direct
        existing = self.store.find_by_scope(task.scope_key)
        if existing is None:
            return None
        subagent_id = str(existing.get("subagent_id") or "")
        return self.store.latest_report(subagent_id)


def create_exploration_tools(
    *,
    workspace: Path,
    session_provider: Callable[[], "RuntimeSession"],
) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name=LIST_EXPLORATION_AGENTS_TOOL,
            label="List exploration agents",
            description=(
                "List session-scoped read-only exploration subagents and their latest reports. "
                "Use this in plan mode before re-dispatching similar repository exploration."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "focus_paths": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
            metadata=_exploration_metadata(LIST_EXPLORATION_AGENTS_TOOL),
            execute=lambda request, signal=None, on_update=None: _execute_list_agents(
                request,
                workspace=workspace,
                session_provider=session_provider,
            ),
        ),
        ToolDefinition(
            name=DISPATCH_EXPLORATION_TOOL,
            label="Dispatch exploration",
            description=(
                "Run up to four read-only exploration subagents in parallel and return "
                "structured evidence for plan generation. Child agents cannot edit files, "
                "update plans, run shell, or dispatch more subagents."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_TASKS_PER_BATCH,
                        "items": {
                            "type": "object",
                            "properties": {
                                "subagent_id": {"type": "string"},
                                "purpose": {"type": "string"},
                                "instruction": {"type": "string"},
                                "focus_paths": {"type": "array", "items": {"type": "string"}},
                                "expected_output": {
                                    "type": "string",
                                    "enum": sorted(_EXPECTED_OUTPUTS),
                                },
                                "critical": {"type": "boolean"},
                            },
                            "required": ["purpose", "instruction"],
                            "additionalProperties": False,
                        },
                    },
                    "max_parallel": {"type": "integer", "minimum": 1, "maximum": MAX_TASKS_PER_BATCH},
                    "reuse": {"type": "string", "enum": sorted(_REUSE_MODES)},
                },
                "required": ["tasks"],
                "additionalProperties": False,
            },
            metadata=_exploration_metadata(DISPATCH_EXPLORATION_TOOL),
            execute=lambda request, signal=None, on_update=None: _execute_dispatch(
                request,
                workspace=workspace,
                session_provider=session_provider,
            ),
        ),
    ]


async def _execute_list_agents(
    request: ToolCallRequest,
    *,
    workspace: Path,
    session_provider: Callable[[], "RuntimeSession"],
) -> ToolResult:
    _ = request
    session = session_provider()
    store = SubagentStore(workspace, session.session_id)
    result = {
        "agents": store.list_agents(
            query=_optional_text(request.arguments.get("query")),
            focus_paths=_string_list(request.arguments.get("focus_paths")),
        )
    }
    return _json_tool_result(result, metadata={"exploration_agents": result})


async def _execute_dispatch(
    request: ToolCallRequest,
    *,
    workspace: Path,
    session_provider: Callable[[], "RuntimeSession"],
) -> ToolResult:
    session = session_provider()
    coordinator = ExplorationCoordinator(
        workspace=workspace,
        session_id=session.session_id,
        model=session.controller.model,
        model_port=session.model_port,
        tool_port=session.tool_port,
        store=SubagentStore(workspace, session.session_id),
    )
    try:
        result = await coordinator.dispatch(dict(request.arguments))
    except ValueError as exc:
        return ToolResult(
            content=[TextContent(text=str(exc))],
            status="error",
            is_error=True,
            error_code="invalid_exploration_request",
        )
    return _json_tool_result(result, metadata={"exploration_batch": result})


def _parse_tasks(arguments: dict[str, Any]) -> list[ExplorationTask]:
    raw_tasks = arguments.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("dispatch_exploration requires a non-empty tasks array")
    if len(raw_tasks) > MAX_TASKS_PER_BATCH:
        raise ValueError(f"dispatch_exploration supports at most {MAX_TASKS_PER_BATCH} tasks")
    tasks: list[ExplorationTask] = []
    for index, raw in enumerate(raw_tasks, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"tasks[{index}] must be an object")
        purpose = _required_text(raw.get("purpose"), f"tasks[{index}].purpose")
        instruction = _required_text(raw.get("instruction"), f"tasks[{index}].instruction")
        focus_paths = _string_list(raw.get("focus_paths"))
        expected_output = _expected_output(raw.get("expected_output"))
        scope_key = _scope_key(purpose, focus_paths)
        subagent_id = _subagent_id(raw.get("subagent_id"), purpose=purpose, scope_key=scope_key)
        tasks.append(
            ExplorationTask(
                task_id=f"task_{index}",
                subagent_id=subagent_id,
                purpose=purpose,
                instruction=instruction,
                scope_key=scope_key,
                focus_paths=focus_paths,
                expected_output=expected_output,
                critical=bool(raw.get("critical", False)),
            )
        )
    return tasks


def _subagent_system_prompt(workspace: Path) -> str:
    allowed = ", ".join(sorted(DEFAULT_READ_ONLY_TOOL_NAMES))
    cwd = str(workspace.resolve()).replace("\\", "/")
    return f"""你是 Codepilot 的只读探索 Subagent。
你的唯一职责是阅读仓库、定位证据、输出结构化事实，帮助主 Agent 在 plan 模式制定计划。

硬性边界：
1. 只能使用这些只读工具：{allowed}。
2. 不要修改文件、不要运行 shell、不要调用 update_plan、不要派发其他 subagent。
3. 不要给最终实施方案下结论；只给证据、风险、开放问题和计划提示。
4. 当前工作目录：{cwd}

最终回复必须是一个 JSON object，不要使用 Markdown 代码块。字段必须包含：
status, summary, findings, relevant_files, evidence, risks, suggested_plan_notes, open_questions, confidence。
confidence 使用 0 到 1 之间的小数。"""


def _subagent_user_prompt(
    task: ExplorationTask,
    peer_assignments: list[dict[str, Any]],
    previous_report: dict[str, Any] | None,
) -> str:
    payload: dict[str, Any] = {
        "task": _task_summary(task),
        "instruction": task.instruction,
        "peer_assignments": peer_assignments,
        "output_contract": {
            "status": "completed | failed",
            "summary": "short factual summary",
            "findings": ["fact with file or symbol evidence"],
            "relevant_files": ["relative/path.py"],
            "evidence": [{"path": "relative/path.py", "note": "why it matters"}],
            "risks": ["risk or uncertainty"],
            "suggested_plan_notes": ["facts the main agent should consider"],
            "open_questions": ["unknowns that still matter"],
            "confidence": 0.0,
        },
    }
    if previous_report is not None:
        payload["previous_report_seed"] = _compact_report(previous_report)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _normalize_report(
    task: ExplorationTask,
    payload: dict[str, Any],
    *,
    event_count: int = 0,
) -> dict[str, Any]:
    if isinstance(payload.get("report"), dict):
        payload = dict(payload["report"])
    status = _optional_text(payload.get("status")) or "completed"
    if status not in {"completed", "failed"}:
        status = "completed"
    report = {
        "status": status,
        "task_id": task.task_id,
        "subagent_id": task.subagent_id,
        "purpose": task.purpose,
        "scope_key": task.scope_key,
        "focus_paths": list(task.focus_paths),
        "expected_output": task.expected_output,
        "summary": _optional_text(payload.get("summary")) or "",
        "findings": _string_list(payload.get("findings")),
        "relevant_files": _string_list(payload.get("relevant_files")),
        "evidence": _evidence_items(payload.get("evidence")),
        "risks": _string_list(payload.get("risks")),
        "suggested_plan_notes": _string_list(payload.get("suggested_plan_notes")),
        "open_questions": _string_list(payload.get("open_questions")),
        "confidence": _confidence(payload.get("confidence")),
        "event_count": event_count,
    }
    if status == "failed" and payload.get("error"):
        report["error"] = payload["error"]
    return report


def _invalid_output_report(task: ExplorationTask, raw_text: str, *, event_count: int = 0) -> dict[str, Any]:
    return _failure_report(
        task,
        status="invalid_output",
        error={
            "code": "subagent.invalid_output",
            "message": "Subagent final answer was not a JSON object",
            "raw_text": raw_text[:REPORT_TEXT_LIMIT],
        },
        event_count=event_count,
    )


def _failure_report(
    task: ExplorationTask,
    *,
    status: str,
    error: dict[str, Any],
    event_count: int = 0,
) -> dict[str, Any]:
    return {
        "status": status,
        "task_id": task.task_id,
        "subagent_id": task.subagent_id,
        "purpose": task.purpose,
        "scope_key": task.scope_key,
        "focus_paths": list(task.focus_paths),
        "expected_output": task.expected_output,
        "summary": "",
        "findings": [],
        "relevant_files": [],
        "evidence": [],
        "risks": [],
        "suggested_plan_notes": [],
        "open_questions": [],
        "confidence": 0.0,
        "event_count": event_count,
        "error": error,
    }


def _skipped_duplicate_report(task: ExplorationTask, reason: str) -> dict[str, Any]:
    return _failure_report(
        task,
        status="skipped_duplicate",
        error={"code": reason},
    )


def _batch_status(reports: list[dict[str, Any]], tasks: list[ExplorationTask]) -> str:
    if not reports:
        return "failed"
    failure_ids = {
        str(report.get("subagent_id"))
        for report in reports
        if report.get("status") in _FAILURE_STATUSES
    }
    critical_failed = any(task.critical and task.subagent_id in failure_ids for task in tasks)
    completed = any(report.get("status") == "completed" for report in reports)
    if critical_failed or (failure_ids and not completed):
        return "failed"
    if failure_ids:
        return "partial_failed"
    return "completed"


def _task_summary(task: ExplorationTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "subagent_id": task.subagent_id,
        "purpose": task.purpose,
        "scope_key": task.scope_key,
        "focus_paths": task.focus_paths,
        "expected_output": task.expected_output,
        "critical": task.critical,
    }


def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "status",
        "summary",
        "findings",
        "relevant_files",
        "evidence",
        "risks",
        "suggested_plan_notes",
        "open_questions",
        "confidence",
    ]
    return {key: report.get(key) for key in keys if key in report}


def _evidence_paths(report: dict[str, Any]) -> list[str]:
    paths = list(report.get("relevant_files") or [])
    for item in report.get("evidence") or []:
        if isinstance(item, dict):
            paths.append(str(item.get("path") or item.get("file") or ""))
        elif isinstance(item, str):
            paths.append(item)
    return _unique_strings(paths)


def _evidence_items(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            path = _optional_text(item.get("path") or item.get("file") or item.get("source")) or ""
            note = _optional_text(item.get("note") or item.get("summary") or item.get("reason")) or ""
            items.append({"path": path, "note": note})
        else:
            text = _optional_text(item)
            if text:
                items.append({"path": text, "note": ""})
    return items


def _parse_json_object(text: str) -> dict[str, Any] | None:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _json_tool_result(payload: dict[str, Any], *, metadata: dict[str, Any]) -> ToolResult:
    return ToolResult(
        content=[
            TextContent(
                text=json.dumps(payload, ensure_ascii=False, indent=2)[:REPORT_TEXT_LIMIT]
            )
        ],
        status="success",
        is_error=False,
        metadata=metadata,
    )


def _exploration_metadata(name: str) -> ToolMetadata:
    return ToolMetadata(
        name=name,
        category="exploration",
        read_only=True,
        concurrency_safe=True,
        exclusive=False,
        requires_approval=False,
        risk_level="low",
        scopes=("plan",),
    )


def _scope_key(purpose: str, focus_paths: list[str]) -> str:
    payload = json.dumps(
        {
            "purpose": purpose.strip().lower(),
            "focus_paths": sorted(path.strip().replace("\\", "/") for path in focus_paths),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _subagent_id(value: object, *, purpose: str, scope_key: str) -> str:
    text = _optional_text(value)
    generated = text is None
    if generated:
        text = purpose
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip().lower()).strip("-_")
    if not slug:
        slug = "explorer"
    if generated:
        return f"{slug[:40]}-{scope_key[:8]}"
    return slug[:64]


def _reuse_mode(value: object) -> str:
    text = _optional_text(value) or "auto"
    if text not in _REUSE_MODES:
        raise ValueError(f"Unknown exploration reuse mode: {text}")
    return text


def _max_parallel(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return DEFAULT_MAX_PARALLEL
    return min(MAX_TASKS_PER_BATCH, max(1, value))


def _expected_output(value: object) -> str:
    text = _optional_text(value) or "open"
    if text not in _EXPECTED_OUTPUTS:
        raise ValueError(f"Unknown expected_output: {text}")
    return text


def _confidence(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def _required_text(value: object, field_name: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    return _unique_strings(str(item).strip() for item in value if str(item).strip())


def _unique_strings(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip().replace("\\", "/")
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


__all__ = [
    "DISPATCH_EXPLORATION_TOOL",
    "LIST_EXPLORATION_AGENTS_TOOL",
    "ExplorationCoordinator",
    "ExplorationTask",
    "SubagentRunner",
    "create_exploration_tools",
]
