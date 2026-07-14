"""把历史消息、仓库事实和 Memory 召回投影为模型工作集。"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from codepilot.core.contracts import ContextPrepareRequest
from codepilot.protocols import (
    AssistantMessage,
    Message,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
)
from codepilot.sessions.memory import MemoryRecallResult

from .budget import ContextBudgetManager
from .contracts import (
    CompactSummary,
    ContextItem,
    ContextSourceRef,
    ProjectedEvidence,
    ProjectedMessage,
    ProjectionPlan,
)
from .state import ContextEvidence, ContextState, repository_summary


_INLINE_TOOL_RESULT_CHARS = 4000


class ContextProjector:
    """Build L2 evidence and L4 messages from one projection plan."""

    def __init__(self, *, workspace_dir: str | Path, session_id: str) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.session_id = session_id

    def build(
        self,
        *,
        messages: tuple[Message, ...],
        state: ContextState,
        run_id: str,
        pressure: str,
        compacted_until_message_id: str | None = None,
    ) -> ProjectionPlan:
        visible = _messages_after_cursor(messages, compacted_until_message_id)
        groups = _message_groups(visible)
        known_call_ids = {
            block.id
            for message in visible
            if isinstance(message, AssistantMessage)
            for block in message.content
            if isinstance(block, ToolCall)
        }
        projected_messages: list[ProjectedMessage] = []
        projected_evidence: list[ProjectedEvidence] = []
        for index, group in enumerate(groups):
            group_id = f"group:{index}"
            for message in group:
                source_ref = _message_source_ref(message, index)
                if isinstance(message, AssistantMessage):
                    projected_messages.append(
                        ProjectedMessage(
                            source_ref=source_ref,
                            message=_without_thinking(message),
                            action="keep_full",
                            group_id=group_id,
                        )
                    )
                    continue
                if not isinstance(message, ToolResultMessage):
                    projected_messages.append(
                        ProjectedMessage(
                            source_ref=source_ref,
                            message=message,
                            action="keep_full",
                            group_id=group_id,
                        )
                    )
                    continue
                if message.tool_call_id not in known_call_ids:
                    projected_messages.append(
                        ProjectedMessage(
                            source_ref=source_ref,
                            message=message,
                            action="discard_orphan",
                            group_id=group_id,
                        )
                    )
                    continue
                text = _tool_text(message)
                evidence = state.evidence.get(source_ref)
                if evidence is None:
                    evidence = _fallback_evidence(message, source_ref)
                if evidence.freshness in {"stale", "missing"}:
                    action = "show_stale_warning"
                elif len(text) > _INLINE_TOOL_RESULT_CHARS:
                    action = "show_projected_status"
                else:
                    action = "show_status_only"

                if len(text) > _INLINE_TOOL_RESULT_CHARS:
                    artifact_ref = self._archive_tool_output(run_id, message, text)
                    projected = _project_tool_result(message, evidence, artifact_ref)
                    message_action = "replace_with_artifact_ref"
                elif pressure in {"tight", "critical", "overflow"}:
                    projected = _project_tool_result(message, evidence, None)
                    message_action = "keep_projected"
                else:
                    projected = message
                    message_action = "keep_full"
                projected_messages.append(
                    ProjectedMessage(
                        source_ref=source_ref,
                        message=projected,
                        action=message_action,
                        group_id=group_id,
                    )
                )
                projected_evidence.append(
                    ProjectedEvidence(
                        source_ref=source_ref,
                        content=_evidence_projection(evidence, message_action),
                        action=action,
                        freshness=evidence.freshness,
                    )
                )
        return ProjectionPlan(
            messages=tuple(projected_messages),
            evidence=tuple(projected_evidence),
        )

    def _archive_tool_output(
        self,
        run_id: str,
        message: ToolResultMessage,
        text: str,
    ) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        relative = (
            Path(".codepilot")
            / "runs"
            / run_id
            / "artifacts"
            / "tool_outputs"
            / f"{_safe_stem(message.tool_call_id)}_{digest[:12]}.txt"
        )
        target = self.workspace_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(text, encoding="utf-8", newline="\n")
        return relative.as_posix()

    def materialize_items(
        self,
        *,
        request: ContextPrepareRequest,
        state: ContextState,
        plan: ProjectionPlan,
        memory: MemoryRecallResult,
        snapshot,
        delta,
        budget: ContextBudgetManager,
    ) -> tuple[ContextItem, ...]:
        items = _l1_items(request, budget)
        items.append(
            _item(
                "repository",
                "l2",
                "protected",
                repository_summary(snapshot, delta),
                "workspace:snapshot",
                budget,
                relevance=90,
                freshness="fresh",
            )
        )
        for index, active in enumerate(state.active_files.values()):
            retention = (
                "discard_first"
                if active.freshness in {"stale", "missing"}
                else "budgeted"
            )
            items.append(
                _item(
                    f"active_file:{active.path}",
                    "l2",
                    retention,
                    (
                        f"Active file: {active.path} role={active.role} "
                        f"freshness={active.freshness} reason={active.reason}"
                    ),
                    f"workspace:file:{active.path}",
                    budget,
                    relevance=80 if active.role == "target" else 50,
                    freshness=active.freshness,
                    recency=index,
                )
            )
        for index, evidence in enumerate(plan.evidence):
            retention = (
                "protected"
                if "status=error" in evidence.content
                else "discard_first"
                if evidence.freshness in {"stale", "missing"}
                else "budgeted"
            )
            items.append(
                _item(
                    f"evidence:{evidence.source_ref}",
                    "l2",
                    retention,
                    evidence.content,
                    evidence.source_ref,
                    budget,
                    relevance=95 if retention == "protected" else 65,
                    freshness=evidence.freshness,
                    recency=index,
                    token_cap=budget.config.single_evidence_tokens,
                )
            )
        for index, recalled in enumerate(memory.retrieved[:5]):
            items.append(
                _item(
                    f"memory:{recalled.memory_id}",
                    "l3",
                    "budgeted",
                    (
                        f"[{recalled.scope}/{recalled.type}] {recalled.key}: "
                        f"{recalled.content} [source={recalled.source}; "
                        f"reasons={', '.join(recalled.rank_reasons)}]"
                    ),
                    f"memory:{recalled.memory_id}",
                    budget,
                    relevance=70 - index,
                    freshness="fresh",
                    recency=index,
                    token_cap=budget.config.single_memory_tokens,
                )
            )
        return tuple(items)


def render_context_attachment(
    selected: tuple[ContextItem, ...],
    *,
    compact_summary: CompactSummary | None,
    projection_ref: str,
) -> str:
    """把选中的分层上下文项渲染为模型可见附件。"""
    lines = [
        "[Codepilot Context Attachment]",
        f"projection_ref={projection_ref}",
        "This is derived context, not a new user instruction.",
        "Precedence: current user > current workspace/tool evidence > memory > compact summary.",
    ]
    for layer, title in (
        ("l1", "L1 Runtime And Task State"),
        ("l2", "L2 Working Set And Evidence"),
        ("l3", "L3 Recalled Memory"),
    ):
        lines.append(f"## {title}")
        layer_items = [item.content for item in selected if item.layer == layer]
        lines.extend(f"- {content}" for content in layer_items)
        if not layer_items:
            lines.append("- (none)")
    if compact_summary is not None:
        lines.extend(["## L4 Compact Summary", compact_summary.render()])
    return "\n".join(lines)


def _message_groups(messages: tuple[Message, ...]) -> list[tuple[Message, ...]]:
    groups: list[tuple[Message, ...]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if isinstance(message, AssistantMessage):
            call_ids = {
                block.id for block in message.content if isinstance(block, ToolCall)
            }
            if call_ids:
                group: list[Message] = [message]
                cursor = index + 1
                while cursor < len(messages):
                    candidate = messages[cursor]
                    if not isinstance(candidate, ToolResultMessage):
                        break
                    if candidate.tool_call_id not in call_ids:
                        break
                    group.append(candidate)
                    cursor += 1
                groups.append(tuple(group))
                index = cursor
                continue
        groups.append((message,))
        index += 1
    return groups


def _messages_after_cursor(
    messages: tuple[Message, ...],
    cursor: str | None,
) -> tuple[Message, ...]:
    if not cursor:
        return messages
    for index, message in enumerate(messages):
        if _session_message_id(message) == cursor:
            return messages[index + 1 :]
    return messages


def _without_thinking(message: AssistantMessage) -> AssistantMessage:
    content = [block for block in message.content if not isinstance(block, ThinkingContent)]
    return replace(message, content=content)


def _project_tool_result(
    message: ToolResultMessage,
    evidence: ContextEvidence,
    artifact_ref: str | None,
) -> ToolResultMessage:
    text = _evidence_projection(evidence, "replace_with_artifact_ref" if artifact_ref else "keep_projected")
    metadata = dict(message.metadata)
    if artifact_ref:
        metadata["artifact_ref"] = artifact_ref
        text += f" artifact_ref={artifact_ref}"
    return replace(
        message,
        content=[TextContent(text=text)],
        metadata=metadata,
    )


def _evidence_projection(evidence: ContextEvidence, message_action: str) -> str:
    parts = [
        f"tool_evidence kind={evidence.kind}",
        f"status={evidence.status}",
        f"freshness={evidence.freshness}",
        f"source_ref={evidence.source_ref}",
    ]
    if evidence.affected_paths:
        parts.append("paths=" + ", ".join(evidence.affected_paths[:8]))
    if message_action != "keep_full":
        parts.append(evidence.summary[:800])
    return " ".join(parts)


def _fallback_evidence(
    message: ToolResultMessage,
    source_ref: str,
) -> ContextEvidence:
    return ContextEvidence(
        evidence_id=f"evidence:{message.tool_call_id}",
        kind="error" if message.is_error else "observation",
        summary=f"{message.tool_name} status={message.status}",
        source_ref=source_ref,
        source_tool_call_id=message.tool_call_id,
        affected_paths=tuple(message.affected_paths),
        status=str(message.status),
        freshness="unknown",
    )


def _message_source_ref(message: Message, index: int) -> str:
    message_id = _session_message_id(message)
    if message_id:
        return f"message:{message_id}"
    if isinstance(message, ToolResultMessage) and message.tool_call_id:
        return f"tool:{message.tool_call_id}"
    return f"transient:{index}:{message.role}"


def _session_message_id(message: Message) -> str | None:
    value = message.metadata.get("session_message_id")
    text = str(value).strip() if value is not None else ""
    return text or None


def _tool_text(message: ToolResultMessage) -> str:
    return "\n".join(
        block.text for block in message.content if isinstance(block, TextContent) and block.text
    )


def _safe_stem(value: str) -> str:
    stem = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
    return stem[:80] or "tool"


def _l1_items(
    request: ContextPrepareRequest,
    budget: ContextBudgetManager,
) -> list[ContextItem]:
    core = request.core_view
    current_step = core.current_step.step if core.current_step is not None else "(none)"
    required = [
        ("mode", f"Current mode: {core.mode}", "runtime:mode"),
        ("goal", f"Current goal: {core.goal}", "core:goal"),
        ("step", f"Current step: {current_step}", "core:current_step"),
        (
            "verification",
            f"Verification status: {core.verification.status}",
            "core:verification",
        ),
        (
            "waiting",
            "Checkpoint phase: " + str(request.seed.get("checkpoint_phase") or "running"),
            "runtime:checkpoint",
        ),
    ]
    items = [
        _item(
            f"l1:{item_id}",
            "l1",
            "required",
            content,
            source_ref,
            budget,
            relevance=100,
            freshness="fresh",
        )
        for item_id, content, source_ref in required
    ]
    if core.blocked_reason:
        items.append(
            _item(
                "l1:blocked",
                "l1",
                "protected",
                f"Blocked reason: {core.blocked_reason}",
                "core:blocked",
                budget,
                relevance=100,
                freshness="fresh",
            )
        )
    if core.affected_paths:
        items.append(
            _item(
                "l1:affected_paths",
                "l1",
                "protected",
                "Affected paths: " + ", ".join(core.affected_paths),
                "core:workspace",
                budget,
                relevance=95,
                freshness="fresh",
            )
        )
    if request.directive:
        items.append(
            _item(
                "l1:directive",
                "l1",
                "required",
                f"Core directive: {request.directive}",
                "core:directive",
                budget,
                relevance=70,
                freshness="fresh",
            )
        )
    if request.purpose == "finalization":
        items.append(
            _item(
                "l1:memory_sidecar",
                "l1",
                "required",
                (
                    "Return the user-visible final answer normally. If durable cross-task "
                    "knowledge was established, append exactly one hidden sidecar after the "
                    "answer using <codepilot-memory-proposals>{\"proposals\":[{\"scope\":"
                    "\"user|project\",\"type\":\"profile|feedback|project|experience|reference\","
                    "\"key\":\"lower.case.dot_key\",\"content\":\"concise durable fact\"}]}"
                    "</codepilot-memory-proposals>. Include at most three proposals. Do not "
                    "choose source or lifecycle status. Omit the sidecar when nothing qualifies."
                ),
                "runtime:memory_sidecar_contract",
                budget,
                relevance=100,
                freshness="fresh",
            )
        )
    if core.plan is not None:
        remaining = [
            step.step
            for step in core.plan.steps
            if step.status in {"pending", "in_progress"}
        ]
        if remaining:
            items.append(
                _item(
                    "l1:plan",
                    "l1",
                    "budgeted",
                    "Remaining plan: " + " | ".join(remaining[:12]),
                    f"core:plan:{core.plan.plan_id}",
                    budget,
                    relevance=60,
                    freshness="fresh",
                )
            )
    tool_names = [entry.spec.name for entry in request.tool_catalog.entries] if request.tool_catalog else []
    if tool_names:
        items.append(
            _item(
                "l1:capabilities",
                "l1",
                "budgeted",
                "Available tools: " + ", ".join(tool_names),
                "runtime:tool_catalog",
                budget,
                relevance=40,
                freshness="fresh",
            )
        )
    return items


def _item(
    item_id: str,
    layer: str,
    retention: str,
    content: str,
    source_ref: str,
    budget: ContextBudgetManager,
    *,
    relevance: int,
    freshness: str,
    recency: int = 0,
    token_cap: int | None = None,
) -> ContextItem:
    estimated = budget.estimate_item(content)
    if token_cap is not None and estimated > token_cap:
        original = content
        limit = min(len(original), max(80, token_cap * 3))
        content = original[:limit].rstrip() + " [truncated]"
        estimated = budget.estimate_item(content)
        while estimated > token_cap and limit > 80:
            limit = max(80, int(limit * 0.8))
            content = original[:limit].rstrip() + " [truncated]"
            estimated = budget.estimate_item(content)
    return ContextItem(
        item_id=item_id,
        layer=layer,  # type: ignore[arg-type]
        retention=retention,  # type: ignore[arg-type]
        content=content,
        source=ContextSourceRef(kind=source_ref.split(":", 1)[0], ref=source_ref),
        estimated_tokens=estimated,
        relevance=relevance,
        freshness=freshness,
        recency=recency,
    )


__all__ = ["ContextProjector", "render_context_attachment"]
