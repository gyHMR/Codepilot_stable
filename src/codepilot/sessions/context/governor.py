from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from codepilot.core.contracts import AgentContext, ContextPreparationRequest, PreparedAgentContext
from codepilot.llm.estimation import (
    ContextUsageCalibrator,
    estimate_context,
    estimate_context_tokens,
    estimate_text_tokens,
    estimate_tools_tokens,
)
from codepilot.protocols import (
    ContextItem,
    ContextReport,
    ContextSectionReport,
    ContextView,
    DroppedContextItem,
    Message,
    TextContent,
    ToolResultMessage,
)
from codepilot.sessions.memory import MemoryQuery, MemoryRecall, render_memory
from codepilot.sessions.storage import SessionLayout, SessionStore
from codepilot.sessions.task_state import TaskStateStore

from .ledger import ContextLedger, ToolArtifactLedger
from .policy import ContextPressurePolicy
from .repository_tracker import RepositoryTracker, render_repository_snapshot
from .state import SessionContextState


_LAYER_ORDER = ("system", "task_state", "working_set", "memory", "conversation")
_LAYER_BUDGET_RATIOS = {
    "system": 0.15,
    "task_state": 0.15,
    "working_set": 0.42,
    "memory": 0.10,
    "conversation": 0.18,
}
_KEEP_COUNTS = {
    "normal": {"task_state": 20, "working_set": 18, "memory": 5, "conversation": 8},
    "tight": {"task_state": 12, "working_set": 10, "memory": 4, "conversation": 4},
    "critical": {"task_state": 8, "working_set": 6, "memory": 2, "conversation": 2},
}


class ContextGovernor:
    """Prepare the model-visible context for one agent turn.

    The implementation is intentionally linear: collect facts, turn them into
    candidate lines, trim candidates, then return a prompt plus an audit report.
    """

    def __init__(
        self,
        *,
        workspace_dir: str | Path,
        session_id: str,
        state: SessionContextState | None = None,
        memory_retriever: Any | None = None,
        pressure_policy: ContextPressurePolicy | None = None,
        task_state_store: TaskStateStore | None = None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.session_id = session_id
        self.layout = SessionLayout.for_workspace(self.workspace_dir, self.session_id)
        self.state = state or SessionContextState(workspace_dir=self.workspace_dir)
        self.memory_retriever = memory_retriever
        self.pressure_policy = pressure_policy or ContextPressurePolicy()
        self.task_state_store = task_state_store or TaskStateStore(
            SessionStore(self.workspace_dir, self.session_id)
        )
        self.repository = RepositoryTracker(self.workspace_dir)
        self.tool_ledger = ToolArtifactLedger(
            workspace_dir=self.workspace_dir,
            session_id=self.session_id,
        )
        self.context_ledger = ContextLedger(
            workspace_dir=self.workspace_dir,
            session_id=self.session_id,
        )
        self.calibrator = ContextUsageCalibrator(self.workspace_dir)

    async def prepare(
        self,
        context: AgentContext,
        request: ContextPreparationRequest,
    ) -> PreparedAgentContext:
        provider, model = _request_provider_model(request)
        correction_factors = self.calibrator.factors_for(provider, model)

        snapshot, delta = self.repository.refresh(self.state.last_repository_snapshot)
        self.state.last_repository_snapshot = snapshot
        if delta.changed:
            self.state.invalidate_paths([*delta.modified_paths, *delta.deleted_paths])
            self.state.invalidate_verification()

        artifact_refs = []
        for message in context.messages:
            if isinstance(message, ToolResultMessage):
                self.state.observe_tool_result(
                    message,
                    repository_fingerprint=snapshot.fingerprint,
                )
                artifact_refs.append(
                    self.tool_ledger.record_tool_result(run_id=_request_run_id(request), message=message).artifact
                )
        artifact_refs.extend(self.tool_ledger.artifact_refs())
        artifact_refs = _dedupe_artifacts(artifact_refs)

        stale_items = self.state.validate_sources(snapshot.fingerprint)
        task_state = _task_state_from_context(context, self.task_state_store)
        raw_estimate = estimate_context(
            context.messages,
            context.system_prompt,
            context.tools,
            correction_factors=correction_factors,
        )
        history_tokens = estimate_context_tokens(
            context.messages,
            "",
            correction_factors=correction_factors,
        )
        tool_tokens = estimate_tools_tokens(context.tools, correction_factors=correction_factors)
        pressure = self.pressure_policy.evaluate(
            request,
            estimated_tokens=raw_estimate.total,
            tool_output_tokens=_tool_output_tokens(context.messages),
            history_tokens=history_tokens,
        )

        memory_recall = self._recall_memory(context, task_state)
        candidates = {
            "system": self._system_items(context.system_prompt, task_state),
            "task_state": _task_items(task_state),
            "working_set": self._working_items(snapshot, delta, stale_items, artifact_refs),
            "memory": _memory_items(memory_recall),
            "conversation": _conversation_items(context.messages),
        }
        selected, dropped, sections = _select_context(candidates, pressure.level, pressure.effective_budget)
        view = ContextView(
            system=[item.content for item in selected["system"]],
            task_state=[item.content for item in selected["task_state"]],
            working_set=[item.content for item in selected["working_set"]],
            memory=[item.content for item in selected["memory"]],
            conversation=[item.content for item in selected["conversation"]],
        )
        selected_messages = _select_messages(context.messages, pressure.level, self.tool_ledger)
        system_prompt = _compose_system_prompt(context.system_prompt, view)
        after_estimate = estimate_context(
            selected_messages,
            system_prompt,
            context.tools,
            correction_factors=correction_factors,
        )
        tokens_by_layer = {
            layer: sum(estimate_text_tokens(line).total for line in getattr(view, layer))
            for layer in _LAYER_ORDER
        }
        tokens_by_layer["tools"] = tool_tokens.total

        dynamic_text = "\n".join(
            [
                *view.task_state,
                *view.working_set,
                *view.memory,
                *view.conversation,
            ]
        )
        report = ContextReport(
            context_id=f"ctx_{_hash_text(dynamic_text)}",
            repository_fingerprint=snapshot.fingerprint,
            total_budget_tokens=pressure.effective_budget,
            estimated_tokens_before=raw_estimate.total,
            estimated_tokens_after=after_estimate.total,
            sections=sections,
            selected_items=_selected_summaries(selected),
            stale_items=stale_items,
            dropped_items=dropped,
            repository_delta=delta,
            retrieved_memory_ids=[item.record.id for item in memory_recall.retrieved],
            memory_retrieval_reasons={
                item.record.id: list(item.reasons)
                for item in memory_recall.retrieved
            },
            dropped_memory_ids=list(memory_recall.dropped),
            dropped_memory_reasons=dict(memory_recall.dropped),
            memory_tokens=tokens_by_layer["memory"],
            context_mode=_context_mode(context),
            pressure=pressure,
            context_view=view,
            artifact_refs=artifact_refs,
            tokens_by_layer=tokens_by_layer,
            prefix_hash=_hash_text("\n".join(view.system)),
            dynamic_hash=_hash_text(dynamic_text),
            estimation={
                "raw_estimate": raw_estimate.total,
                "estimated_after": after_estimate.total,
                "by_type": dict(raw_estimate.by_type),
            },
        )
        self.context_ledger.append_projection(report, run_id=_request_run_id(request))
        return PreparedAgentContext(
            system_prompt=system_prompt,
            messages=selected_messages,
            tools=list(context.tools),
            report=report,
        )

    def finalize_run(self, result: object) -> None:
        """Kept as a named lifecycle hook for future session-local cleanup."""

    def _recall_memory(
        self,
        context: AgentContext,
        task_state: Mapping[str, object] | None,
    ) -> MemoryRecall:
        if self.memory_retriever is None:
            return MemoryRecall()
        query = MemoryQuery(
            latest_user_message=_latest_user_text(context.messages),
            raw_user_request=_task_string(task_state, "raw_user_request") or "",
            goal=_task_goal(task_state),
            current_mode=_task_string(task_state, "current_mode") or _context_mode(context),
            verification_status=_task_string(task_state, "verification_status"),
            blocked_reason=_task_string(task_state, "blocked_reason"),
            active_paths=sorted(self.state.active_files),
            changed_paths=[],
            session_id=self.session_id,
            run_id=_signal_text(context.task_signal, "run_id"),
            limit=5,
        )
        return self.memory_retriever.recall(query)

    def _system_items(
        self,
        system_prompt: str,
        task_state: Mapping[str, object] | None,
    ) -> list[ContextItem]:
        lines = [line.strip() for line in system_prompt.splitlines() if line.strip()]
        mode = _task_string(task_state, "current_mode")
        if mode:
            lines.append(f"Current task mode: {mode}")
        return [_item(f"system:{index}", "system", line, "system_prompt", 100 - index) for index, line in enumerate(lines[:10])]

    def _working_items(
        self,
        snapshot: Any,
        delta: Any,
        stale_items: list[str],
        artifact_refs: list[Any],
    ) -> list[ContextItem]:
        lines = [render_repository_snapshot(snapshot, delta)]
        if stale_items:
            lines.append("Stale context: " + ", ".join(stale_items[:10]))
        for active in sorted(
            self.state.active_files.values(),
            key=lambda item: (item.role, item.access_count, item.last_accessed_at),
            reverse=True,
        )[:10]:
            lines.append(
                f"Active file: {active.path} role={active.role} freshness={active.freshness} reason={active.reason}"
            )
        for evidence in self.state.evidence[-8:]:
            lines.append(f"Evidence from {evidence.source}: {evidence.content}")
        for ref in artifact_refs[:8]:
            lines.append(f"Artifact: {ref.path} summary={ref.summary}")
        return [
            _item(f"working:{index}", "working_set", line, "session_context", 80 - index)
            for index, line in enumerate(lines)
            if line
        ]


def calibrate_context_usage(
    *,
    workspace_dir: str | Path,
    provider: str | None,
    model: str | None,
    report: Mapping[str, object] | None,
    actual_input_tokens: int,
) -> None:
    if not report or actual_input_tokens <= 0:
        return
    estimation = report.get("estimation")
    if not isinstance(estimation, Mapping):
        return
    raw_estimate = estimation.get("raw_estimate")
    by_type = estimation.get("by_type")
    if not isinstance(raw_estimate, int) or not isinstance(by_type, dict):
        return
    ContextUsageCalibrator(workspace_dir).update(
        provider=provider,
        model=model,
        raw_estimate=raw_estimate,
        actual_input_tokens=actual_input_tokens,
        breakdown={
            str(key): int(value)
            for key, value in by_type.items()
            if isinstance(value, int) and not isinstance(value, bool)
        },
    )


def _select_context(
    candidates: dict[str, list[ContextItem]],
    pressure_level: str,
    total_budget: int,
) -> tuple[dict[str, list[ContextItem]], list[DroppedContextItem], list[ContextSectionReport]]:
    selected: dict[str, list[ContextItem]] = {}
    dropped: list[DroppedContextItem] = []
    sections: list[ContextSectionReport] = []
    for layer in _LAYER_ORDER:
        items = sorted(candidates[layer], key=lambda item: (item.priority, -item.estimated_tokens), reverse=True)
        layer_budget = max(32, int(total_budget * _LAYER_BUDGET_RATIOS[layer]))
        keep_count = _KEEP_COUNTS.get(pressure_level, _KEEP_COUNTS["tight"]).get(layer, 10)
        kept: list[ContextItem] = []
        token_total = 0
        for item in items:
            over_budget = token_total + item.estimated_tokens > layer_budget
            over_count = len(kept) >= keep_count
            if over_budget or over_count:
                dropped.append(
                    DroppedContextItem(
                        item_id=item.id,
                        section=layer,
                        reason="over_budget" if over_budget else "low_relevance",
                        source=item.source,
                    )
                )
                continue
            kept.append(item)
            token_total += item.estimated_tokens
        selected[layer] = kept
        sections.append(
            ContextSectionReport(
                name=layer,
                budget_tokens=layer_budget,
                candidate_items=len(items),
                selected_items=len(kept),
                estimated_tokens_before=sum(item.estimated_tokens for item in items),
                estimated_tokens_after=token_total,
                reduction_policy="simple_priority_budget",
            )
        )
    return selected, dropped, sections


def _task_items(task_state: Mapping[str, object] | None) -> list[ContextItem]:
    if task_state is None:
        return []
    lines = [
        f"Goal: {_task_goal(task_state)}",
        f"Mode: {_task_string(task_state, 'current_mode') or 'build'}",
        f"Verification: {_task_string(task_state, 'verification_status') or 'unknown'}",
    ]
    blocked = _task_string(task_state, "blocked_reason")
    if blocked:
        lines.append(f"Blocked: {blocked}")
    steps = task_state.get("steps")
    if isinstance(steps, list):
        for step in steps[:8]:
            if isinstance(step, Mapping):
                lines.append(
                    f"Step {step.get('id')}: {step.get('title')} [{step.get('status')}]"
                )
    return [
        _item(f"task:{index}", "task_state", line, "task_state", 75 - index)
        for index, line in enumerate(lines)
        if line.strip()
    ]


def _memory_items(recall: MemoryRecall) -> list[ContextItem]:
    return [
        _item(
            item.record.id,
            "memory",
            f"{render_memory(item.record)} [reasons={', '.join(item.reasons)}]",
            "memory_recall",
            70 - index,
        )
        for index, item in enumerate(recall.retrieved[:5])
    ]


def _conversation_items(messages: list[Message]) -> list[ContextItem]:
    items: list[ContextItem] = []
    for index, message in enumerate(messages[-8:]):
        text = _message_preview(message)
        if text:
            items.append(
                _item(
                    f"conversation:{index}",
                    "conversation",
                    f"{getattr(message, 'role', 'message')}: {text}",
                    "recent_messages",
                    40 - index,
                )
            )
    return items


def _select_messages(
    messages: list[Message],
    pressure_level: str,
    ledger: ToolArtifactLedger,
) -> list[Message]:
    keep = {"normal": len(messages), "tight": 12, "critical": 6}.get(pressure_level, 12)
    selected = list(messages[-keep:])
    if pressure_level == "normal":
        return selected
    return [
        ledger.project_tool_result(message, preserve_full=pressure_level == "tight")
        if isinstance(message, ToolResultMessage)
        else message
        for message in selected
    ]


def _compose_system_prompt(base: str, view: ContextView) -> str:
    sections = [
        ("System", view.system),
        ("Task State", view.task_state),
        ("Working Set", view.working_set),
        ("Memory", view.memory),
        ("Conversation", view.conversation),
    ]
    parts = [base.rstrip()]
    for name, lines in sections:
        if lines:
            parts.append(f"## {name}\n" + "\n".join(f"- {line}" for line in lines))
    return "\n\n".join(part for part in parts if part).strip()


def _selected_summaries(selected: dict[str, list[ContextItem]]) -> list[dict[str, object]]:
    return [
        {
            "id": item.id,
            "layer": layer,
            "kind": item.kind,
            "path": item.path or "",
            "source": item.source,
            "estimated_tokens": item.estimated_tokens,
            "freshness": item.freshness,
        }
        for layer, items in selected.items()
        for item in items
    ]


def _item(item_id: str, kind: str, content: str, source: str, priority: int) -> ContextItem:
    return ContextItem(
        id=item_id,
        kind=kind,
        content=content,
        source=source,
        trust="observed" if source != "memory_recall" else "derived",
        priority=priority,
        estimated_tokens=estimate_text_tokens(content).total,
        freshness="fresh",
    )


def _task_state_from_context(
    context: AgentContext,
    task_state_store: TaskStateStore,
) -> Mapping[str, object] | None:
    if isinstance(context.task_state, Mapping):
        return context.task_state
    return task_state_store.current()


def _task_goal(task_state: Mapping[str, object] | None) -> str:
    if task_state is None:
        return ""
    goal = task_state.get("goal")
    if isinstance(goal, Mapping) and isinstance(goal.get("value"), str):
        return str(goal["value"])
    value = task_state.get("raw_user_request")
    return value if isinstance(value, str) else ""


def _task_string(task_state: Mapping[str, object] | None, key: str) -> str | None:
    if task_state is None:
        return None
    value = task_state.get(key)
    return value if isinstance(value, str) and value else None


def _context_mode(context: AgentContext) -> str:
    text = _latest_user_text(context.messages).lower()
    if _signal_text(context.task_signal, "recent_error_code"):
        return "repair"
    if any(word in text for word in ("why", "how", "explain", "为什么", "解释")):
        return "qa"
    return "act"


def _latest_user_text(messages: list[Message]) -> str:
    for message in reversed(messages):
        if getattr(message, "role", None) == "user":
            content = getattr(message, "content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(getattr(block, "text", "") for block in content)
    return ""


def _message_preview(message: Message, *, limit: int = 240) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return " ".join(content.split())[:limit]
    if isinstance(content, list):
        text = "".join(getattr(block, "text", "") for block in content if isinstance(block, TextContent))
        return " ".join(text.split())[:limit]
    return ""


def _tool_output_tokens(messages: list[Message]) -> int:
    return sum(
        estimate_context_tokens([message], "")
        for message in messages
        if isinstance(message, ToolResultMessage)
    )


def _request_run_id(request: ContextPreparationRequest) -> str | None:
    signal = request.signal
    if isinstance(signal, Mapping):
        value = signal.get("run_id")
        return value if isinstance(value, str) else None
    return None


def _request_provider_model(request: ContextPreparationRequest) -> tuple[str | None, str | None]:
    signal = request.signal
    if not isinstance(signal, Mapping):
        return None, None
    provider = signal.get("provider")
    model = signal.get("model")
    return (
        provider if isinstance(provider, str) and provider else None,
        model if isinstance(model, str) and model else None,
    )


def _signal_text(signal: object, key: str) -> str | None:
    if not isinstance(signal, Mapping):
        return None
    value = signal.get(key)
    return value if isinstance(value, str) and value else None


def _dedupe_artifacts(items: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for item in items:
        key = str(getattr(item, "path", ""))
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


__all__ = ["ContextGovernor", "calibrate_context_usage"]
