from __future__ import annotations

# 新手导读：trace.py 组织一次 run 的 trace/report/audit bundle。
# 关注点：它把分散事件串成可追踪的执行故事。

"""Typed trace projection built from canonical run events."""

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from .events import RUN_EVENT_TYPES, event_to_record
from .redact import redact_artifact


@dataclass(frozen=True)
class ModelCallTrace:
    provider: str = ""
    model: str = ""
    api: str = ""
    stop_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    timestamp_ms: int | None = None


@dataclass(frozen=True)
class ContextTrace:
    context_id: str = ""
    mode: str = "normal"
    budget_tokens: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    selected_items: list[dict[str, Any]] = field(default_factory=list)
    stale_items: list[str] = field(default_factory=list)
    dropped_counts: dict[str, int] = field(default_factory=dict)
    tokens_by_layer: dict[str, int] = field(default_factory=dict)
    memory_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ToolCallTrace:
    tool_call_id: str
    tool_name: str
    status: str
    is_error: bool = False
    error_reason: str | None = None
    approved: bool = True
    permission: str | None = None
    duration_ms: int | None = None
    affected_paths: list[str] = field(default_factory=list)
    workspace_changed: bool | None = None
    verification_status: str = "none"
    output_truncated: bool = False


@dataclass(frozen=True)
class TaskTrace:
    task_id: str = ""
    mode: str = ""
    phase: str = ""
    step_id: str = ""
    step_title: str = ""
    step_status: str = ""
    decision: str = ""
    reason: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    completion_satisfied: bool | None = None
    completion_reason: str | None = None


@dataclass(frozen=True)
class MemoryTrace:
    memory_ids: list[str] = field(default_factory=list)
    action: str = ""
    reasons: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class RunTrace:
    run_id: str
    session_id: str | None
    status: str = ""
    stop_reason: str = ""
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    model_calls: list[ModelCallTrace] = field(default_factory=list)
    contexts: list[ContextTrace] = field(default_factory=list)
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    tasks: list[TaskTrace] = field(default_factory=list)
    memories: list[MemoryTrace] = field(default_factory=list)
    affected_paths: list[str] = field(default_factory=list)
    workspace_changed: bool = False
    errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuditBundle:
    """New read-only run evidence bundle.

    The name is kept for runtime API continuity, but it no longer exposes the
    old audit-report schema.
    """

    run_id: str
    session_id: str | None
    events: list[dict[str, Any]]
    result: dict[str, Any]
    trace: RunTrace
    workspace: Path


def build_run_trace(
    events: list[dict[str, Any]],
    result: dict[str, Any] | Any | None = None,
) -> RunTrace:
    result = _result_dict(result)
    events = _canonical_events(events)
    run_id = _first_text(events, "run_id") or str(result.get("run_id") or "")
    session_id = _first_text(events, "session_id") or _optional_str(
        result.get("session_id")
    )
    status = str(result.get("status") or "")
    stop_reason = str(result.get("stop_reason") or "")
    started_at: int | None = None
    finished_at: int | None = None
    model_calls: list[ModelCallTrace] = []
    contexts: list[ContextTrace] = []
    tool_calls: list[ToolCallTrace] = []
    tasks: list[TaskTrace] = []
    memories: list[MemoryTrace] = []
    errors: list[dict[str, Any]] = []

    for event in events:
        event_type = event.get("type")
        timestamp = _optional_int(event.get("timestamp_ms"))
        if event_type == "run_started":
            started_at = timestamp
        elif event_type == "run_finished":
            finished_at = timestamp
            status = str(event.get("status") or status)
            stop_reason = str(event.get("stop_reason") or stop_reason)
        elif event_type == "model_call_finished":
            model_calls.append(_model_call(event))
        elif event_type == "context_built":
            contexts.append(_context(event))
        elif event_type == "tool_call_finished":
            tool_calls.append(_tool_call(event))
        elif event_type in {
            "task_plan_created",
            "task_step_updated",
            "task_decision_made",
            "completion_checked",
        }:
            tasks.append(_task(event))
        elif event_type in {"memory_retrieved", "memory_written"}:
            memories.append(_memory(event))
        elif event_type == "error":
            errors.append(dict(event))

    if not model_calls:
        model_calls.extend(_model_calls_from_result(result))
    if not tool_calls:
        tool_calls.extend(_tool_calls_from_result(result))

    affected = sorted(
        {
            *[
                str(path)
                for item in tool_calls
                for path in item.affected_paths
                if path
            ],
            *[
                str(path)
                for path in result.get("affected_paths", [])
                if isinstance(path, str)
            ],
        }
    )
    workspace_changed = bool(result.get("workspace_changed")) or any(
        item.workspace_changed is True for item in tool_calls
    )
    return RunTrace(
        run_id=run_id,
        session_id=session_id,
        status=status,
        stop_reason=stop_reason,
        started_at_ms=started_at,
        finished_at_ms=finished_at,
        model_calls=model_calls,
        contexts=contexts,
        tool_calls=tool_calls,
        tasks=tasks,
        memories=memories,
        affected_paths=affected,
        workspace_changed=workspace_changed,
        errors=errors,
    )


def _canonical_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") in RUN_EVENT_TYPES:
            records.append(dict(event))
            continue
        record = event_to_record(event)
        if record:
            records.append(record)
    return records


def _result_dict(result: dict[str, Any] | Any | None) -> dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    if is_dataclass(result):
        return asdict(result)
    if hasattr(result, "to_dict"):
        value = result.to_dict()
        return dict(value) if isinstance(value, dict) else {}
    return {}


def _model_calls_from_result(result: dict[str, Any]) -> list[ModelCallTrace]:
    messages = _list_of_dicts(result.get("messages"))
    final = result.get("final_message")
    candidates = [final] if isinstance(final, dict) else []
    candidates.extend(messages)
    traces: list[ModelCallTrace] = []
    for message in candidates:
        if message.get("role") != "assistant":
            continue
        usage = _dict(message.get("usage"))
        if not usage:
            continue
        cost = _dict(usage.get("cost"))
        traces.append(
            ModelCallTrace(
                provider=str(message.get("provider") or ""),
                model=str(message.get("model") or ""),
                api=str(message.get("api") or ""),
                stop_reason=str(message.get("stop_reason") or ""),
                input_tokens=_optional_int(usage.get("input")) or 0,
                output_tokens=_optional_int(usage.get("output")) or 0,
                total_tokens=_optional_int(usage.get("total_tokens")) or 0,
                total_cost=float(cost.get("total") or 0.0),
                timestamp_ms=_optional_int(message.get("timestamp")),
            )
        )
        break
    return traces


def _tool_calls_from_result(result: dict[str, Any]) -> list[ToolCallTrace]:
    traces: list[ToolCallTrace] = []
    for message in _list_of_dicts(result.get("messages")):
        if message.get("role") != "toolResult":
            continue
        verification = _dict(message.get("verification"))
        traces.append(
            ToolCallTrace(
                tool_call_id=str(message.get("tool_call_id") or ""),
                tool_name=str(message.get("tool_name") or ""),
                status=str(message.get("status") or "success"),
                is_error=bool(message.get("is_error", False)),
                error_reason=message.get("error_code"),
                approved=bool(message.get("approved", True)),
                affected_paths=[
                    str(path)
                    for path in message.get("affected_paths", [])
                    if isinstance(path, str)
                ],
                workspace_changed=_optional_bool(message.get("workspace_changed")),
                verification_status=str(verification.get("status") or "none"),
                output_truncated=False,
            )
        )
    return traces


def write_run_trace(path: str | Path, trace: RunTrace) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(redact_artifact(trace.to_dict()), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_run_trace(path: str | Path) -> RunTrace:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Trace must be a JSON object: {path}")
    return RunTrace(
        run_id=str(payload.get("run_id", "")),
        session_id=_optional_str(payload.get("session_id")),
        status=str(payload.get("status", "")),
        stop_reason=str(payload.get("stop_reason", "")),
        started_at_ms=_optional_int(payload.get("started_at_ms")),
        finished_at_ms=_optional_int(payload.get("finished_at_ms")),
        model_calls=[
            ModelCallTrace(**item)
            for item in _list_of_dicts(payload.get("model_calls"))
        ],
        contexts=[
            ContextTrace(**item)
            for item in _list_of_dicts(payload.get("contexts"))
        ],
        tool_calls=[
            ToolCallTrace(**item)
            for item in _list_of_dicts(payload.get("tool_calls"))
        ],
        tasks=[
            TaskTrace(**item)
            for item in _list_of_dicts(payload.get("tasks"))
        ],
        memories=[
            MemoryTrace(**item)
            for item in _list_of_dicts(payload.get("memories"))
        ],
        affected_paths=[
            str(path)
            for path in payload.get("affected_paths", [])
            if isinstance(path, str)
        ],
        workspace_changed=bool(payload.get("workspace_changed", False)),
        errors=_list_of_dicts(payload.get("errors")),
    )


def load_audit_bundle(
    run_dir: str | Path,
    *,
    workspace: str | Path | None = None,
) -> AuditBundle:
    root = Path(run_dir)
    events = _read_jsonl(root / "events.jsonl")
    result = _read_json(root / "run.json")
    trace_path = root / "trace.json"
    trace = (
        load_run_trace(trace_path)
        if trace_path.is_file()
        else build_run_trace(events, result=result)
    )
    return AuditBundle(
        run_id=trace.run_id or root.name,
        session_id=trace.session_id,
        events=events,
        result=result,
        trace=trace,
        workspace=Path(
            workspace
            or result.get("workspace_path")
            or root.parents[2]
        ),
    )


def _model_call(event: dict[str, Any]) -> ModelCallTrace:
    return ModelCallTrace(
        provider=str(event.get("provider", "")),
        model=str(event.get("model", "")),
        api=str(event.get("api", "")),
        stop_reason=str(event.get("stop_reason", "")),
        input_tokens=_int(event.get("input_tokens")),
        output_tokens=_int(event.get("output_tokens")),
        total_tokens=_int(event.get("total_tokens")),
        total_cost=_float(event.get("total_cost")),
        timestamp_ms=_optional_int(event.get("timestamp_ms")),
    )


def _context(event: dict[str, Any]) -> ContextTrace:
    return ContextTrace(
        context_id=str(event.get("context_id", "")),
        mode=str(event.get("mode", "normal")),
        budget_tokens=_int(event.get("budget_tokens")),
        tokens_before=_int(event.get("tokens_before")),
        tokens_after=_int(event.get("tokens_after")),
        selected_items=_list_of_dicts(event.get("selected_items")),
        stale_items=[str(item) for item in event.get("stale_items", []) if isinstance(item, str)],
        dropped_counts={
            str(key): _int(value)
            for key, value in _dict(event.get("dropped_counts")).items()
        },
        tokens_by_layer={
            str(key): _int(value)
            for key, value in _dict(event.get("tokens_by_layer")).items()
        },
        memory_ids=[
            str(item) for item in event.get("memory_ids", []) if isinstance(item, str)
        ],
    )


def _tool_call(event: dict[str, Any]) -> ToolCallTrace:
    return ToolCallTrace(
        tool_call_id=str(event.get("tool_call_id", "")),
        tool_name=str(event.get("tool_name", "")),
        status=str(event.get("status", "")),
        is_error=bool(event.get("is_error", False)),
        error_reason=_optional_str(event.get("error_reason")),
        approved=bool(event.get("approved", True)),
        permission=_optional_str(event.get("permission")),
        duration_ms=_optional_int(event.get("duration_ms")),
        affected_paths=[
            str(path)
            for path in event.get("affected_paths", [])
            if isinstance(path, str)
        ],
        workspace_changed=_optional_bool(event.get("workspace_changed")),
        verification_status=str(event.get("verification_status") or "none"),
        output_truncated=bool(event.get("output_truncated", False)),
    )


def _task(event: dict[str, Any]) -> TaskTrace:
    return TaskTrace(
        task_id=str(event.get("task_id", "")),
        mode=str(event.get("mode", "")),
        phase=str(event.get("phase", "")),
        step_id=str(event.get("step_id", "")),
        step_title=str(event.get("step_title", "")),
        step_status=str(event.get("step_status", "")),
        decision=str(event.get("decision", "")),
        reason=str(event.get("reason", "")),
        evidence_refs=[
            str(item)
            for item in event.get("evidence_refs", [])
            if isinstance(item, str)
        ],
        completion_satisfied=_optional_bool(event.get("completion_satisfied")),
        completion_reason=_optional_str(event.get("completion_reason")),
    )


def _memory(event: dict[str, Any]) -> MemoryTrace:
    return MemoryTrace(
        memory_ids=[
            str(item) for item in event.get("memory_ids", []) if isinstance(item, str)
        ],
        action=str(event.get("action", "")),
        reasons={
            str(key): [str(item) for item in value if isinstance(item, str)]
            for key, value in _dict(event.get("reasons")).items()
            if isinstance(value, list)
        },
    )


def _first_text(events: list[dict[str, Any]], key: str) -> str | None:
    return next(
        (
            str(event[key])
            for event in events
            if isinstance(event.get(key), str) and str(event[key])
        ),
        None,
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            records.append(value)
    return records


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _float(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


__all__ = [
    "AuditBundle",
    "ContextTrace",
    "MemoryTrace",
    "ModelCallTrace",
    "RunTrace",
    "TaskTrace",
    "ToolCallTrace",
    "build_run_trace",
    "load_audit_bundle",
    "load_run_trace",
    "write_run_trace",
]
