from __future__ import annotations

# 新手导读：ContextProjector 把会话事实投影成 System、Task State、Working Set、Memory 和 Conversation 五层。
# 关注点：它决定最终进入模型 prompt 的信息结构。

"""ContextProjector：把 SessionSnapshot 投影成本轮 prompt 视图。"""

from dataclasses import dataclass

from codepilot.core.contracts import AgentContext
from codepilot.llm.estimation import estimate_context_tokens
from codepilot.protocols import (
    AssistantMessage,
    ContextArtifactRef,
    ContextCheckpoint,
    ContextPressure,
    ContextSectionReport,
    ContextView,
    Message,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

from .compactor import ContextCompactResult
from .ledger import ToolArtifactLedger
from .state import ActiveFile, ContextEvidence


@dataclass(frozen=True)
class ContextProjection:
    """ContextProjector 输出的 prompt 结构。"""

    view: ContextView
    messages: list[Message]
    system_prompt: str


class ContextProjector:
    """按五层上下文生成本轮决策视图。"""

    def __init__(self, *, ledger: ToolArtifactLedger) -> None:
        self.ledger = ledger

    def render_evidence(
        self,
        *,
        evidence: list[ContextEvidence],
        artifacts: list[ContextArtifactRef],
        stale_items: list[str],
    ) -> list[str]:
        lines: list[str] = []
        for item in evidence:
            if item.freshness == "fresh":
                lines.append(
                    f"[{item.kind}] {item.source}: "
                    f"{_compact_evidence_content(item.content)}"
                )
        for artifact in artifacts[-8:]:
            lines.append(f"[artifact] {artifact.summary} -> {artifact.path}")
        for item in stale_items[:8]:
            lines.append(f"[stale] {item}")
        return lines

    def project(
        self,
        *,
        context: AgentContext,
        pressure: ContextPressure,
        checkpoint: ContextCheckpoint | None,
        active_files: list[str],
        changed_files: list[str],
        memory_lines: list[str],
        evidence_lines: list[str],
        compact_result: ContextCompactResult | None = None,
        emergency_trim: bool = False,
    ) -> ContextProjection:
        view = ContextView(
            stable_rules=stable_rules(context.system_prompt),
            task_state=task_state_lines(
                context,
                checkpoint=checkpoint,
                compact_result=compact_result,
                emergency_trim=emergency_trim,
            ),
            working_set=working_set_lines(
                active_files=active_files,
                changed_files=changed_files,
                evidence_lines=evidence_lines,
                compact_result=compact_result,
                emergency_trim=emergency_trim,
            ),
            recalled_memory=compacted_memory_lines(
                memory_lines,
                compact_result=compact_result,
                emergency_trim=emergency_trim,
            ),
            conversation=compacted_conversation_lines(
                context.messages,
                pressure.level,
                compact_result=compact_result,
                emergency_trim=emergency_trim,
            ),
            tools=[tool.name for tool in context.tools],
        )
        messages = self.project_messages(
            context.messages,
            pressure.level,
            compact_result=compact_result,
            emergency_trim=emergency_trim,
        )
        return ContextProjection(
            view=view,
            messages=messages,
            system_prompt=compose_system_prompt(context.system_prompt, view),
        )

    def project_messages(
        self,
        messages: list[Message],
        pressure_level: str,
        *,
        compact_result: ContextCompactResult | None = None,
        emergency_trim: bool = False,
    ) -> list[Message]:
        if emergency_trim:
            keep_recent = 1
        elif compact_result is not None:
            keep_recent = 2
        else:
            keep_recent = {"normal": 10, "tight": 6, "critical": 4}.get(
                pressure_level,
                6,
            )
        selected = list(messages[-keep_recent:])
        out: list[Message] = []
        for message in selected:
            if isinstance(message, ToolResultMessage):
                out.append(
                    self.ledger.project_tool_result(
                        message,
                        preserve_full=pressure_level == "normal"
                        and compact_result is None
                        and not emergency_trim,
                    )
                )
            else:
                out.append(message)
        return repair_tool_pairs(out, messages)


def tool_output_tokens(messages: list[Message]) -> int:
    return sum(
        estimate_context_tokens([message], "")
        for message in messages
        if isinstance(message, ToolResultMessage)
    )


def stable_rules(system_prompt: str) -> list[str]:
    lines = [line.strip() for line in system_prompt.splitlines() if line.strip()]
    instruction_lines = [
        line
        for line in lines
        if "AGENTS" in line or "CLAUDE" in line or "rule" in line.lower()
    ]
    return instruction_lines or lines[:8]


def task_state_lines(
    context: AgentContext,
    *,
    checkpoint: ContextCheckpoint | None,
    compact_result: ContextCompactResult | None = None,
    emergency_trim: bool = False,
) -> list[str]:
    lines: list[str] = []
    if context.current_task:
        lines.append(context.current_task)
    if context.task_signal:
        signal_parts = [
            f"{key}={value}"
            for key, value in context.task_signal.items()
            if isinstance(value, str) and value
            and key in {"phase", "action_intent", "recent_error_code", "next_action"}
        ]
        if signal_parts:
            lines.append("Task signal: " + ", ".join(signal_parts))
    if checkpoint is not None:
        lines.append(f"Checkpoint goal: {checkpoint.goal}")
        if checkpoint.next_actions:
            lines.append(f"Next actions: {', '.join(checkpoint.next_actions)}")
    if compact_result is not None:
        if compact_result.recovery_summary:
            lines.append(
                "Compact summary: "
                + _compact_line(
                    compact_result.recovery_summary,
                    limit=220 if emergency_trim else 800,
                )
            )
        lines.extend(
            _limit_lines(
                compact_result.task_state_lines,
                count=1 if emergency_trim else 4,
                limit=180 if emergency_trim else 500,
            )
        )
    return lines


def working_set_lines(
    *,
    active_files: list[str],
    changed_files: list[str],
    evidence_lines: list[str],
    compact_result: ContextCompactResult | None = None,
    emergency_trim: bool = False,
) -> list[str]:
    lines: list[str] = []
    if active_files:
        limit = 4 if emergency_trim else 12
        lines.append(f"Active files: {', '.join(active_files[:limit])}")
    if changed_files:
        limit = 4 if emergency_trim else 12
        lines.append(f"Changed files: {', '.join(changed_files[:limit])}")
    if compact_result is not None and compact_result.working_set_lines:
        lines.extend(
            _limit_lines(
                compact_result.working_set_lines,
                count=2 if emergency_trim else 6,
                limit=180 if emergency_trim else 500,
            )
        )
    else:
        lines.extend(evidence_lines[: 4 if emergency_trim else 12])
    return lines


def compacted_memory_lines(
    memory_lines: list[str],
    *,
    compact_result: ContextCompactResult | None,
    emergency_trim: bool,
) -> list[str]:
    if compact_result is not None and compact_result.memory_lines:
        return _limit_lines(
            compact_result.memory_lines,
            count=1 if emergency_trim else 3,
            limit=180 if emergency_trim else 500,
        )
    return _limit_lines(
        memory_lines,
        count=1 if emergency_trim else len(memory_lines),
        limit=180 if emergency_trim else 500,
    )


def compacted_conversation_lines(
    messages: list[Message],
    pressure_level: str,
    *,
    compact_result: ContextCompactResult | None,
    emergency_trim: bool,
) -> list[str]:
    if compact_result is not None and compact_result.conversation_lines:
        return _limit_lines(
            compact_result.conversation_lines,
            count=1 if emergency_trim else 3,
            limit=180 if emergency_trim else 500,
        )
    return recent_message_lines(
        messages,
        "critical" if emergency_trim else pressure_level,
    )


def recent_message_lines(messages: list[Message], pressure_level: str) -> list[str]:
    limit = {"normal": 6, "tight": 4, "critical": 2}.get(pressure_level, 4)
    lines: list[str] = []
    for message in messages[-limit:]:
        text = message_text(message)
        if text:
            lines.append(f"{getattr(message, 'role', 'message')}: {text[:240]}")
    return lines


def compose_system_prompt(system_prompt: str, view: ContextView) -> str:
    sections = [
        ("System", view.stable_rules),
        ("Task State", view.task_state),
        ("Working Set", view.working_set),
        ("Memory", view.recalled_memory),
        ("Conversation", view.conversation),
    ]
    parts = [system_prompt.rstrip()]
    for name, lines in sections:
        if not lines:
            continue
        parts.append(f"## {name}\n" + "\n".join(f"- {line}" for line in lines))
    return "\n\n".join(part for part in parts if part).strip()


def section_reports(view: ContextView, total_budget: int) -> list[ContextSectionReport]:
    sections = {
        "system": view.stable_rules,
        "task_state": view.task_state,
        "working_set": view.working_set,
        "memory": view.recalled_memory,
        "conversation": view.conversation,
    }
    each_budget = max(1, total_budget // max(1, len(sections)))
    return [
        ContextSectionReport(
            name=name,
            budget_tokens=each_budget,
            candidate_items=len(lines),
            selected_items=len(lines),
            estimated_tokens_before=estimate_lines(lines),
            estimated_tokens_after=estimate_lines(lines),
            reduction_policy="context_governor_projection",
        )
        for name, lines in sections.items()
    ]


def tokens_by_layer(view: ContextView) -> dict[str, int]:
    return {
        "system": estimate_lines(view.stable_rules),
        "task_state": estimate_lines(view.task_state),
        "working_set": estimate_lines(view.working_set),
        "memory": estimate_lines(view.recalled_memory),
        "conversation": estimate_lines(view.conversation),
    }


def estimate_lines(lines: list[str]) -> int:
    return max(0, sum(max(1, len(line) // 4) for line in lines))


def selected_item_summaries(
    view: ContextView,
    *,
    active_files: list[str] | None = None,
    changed_files: list[str] | None = None,
    active_file_records: list[ActiveFile] | None = None,
    evidence: list[ContextEvidence] | None = None,
    artifacts: list[ContextArtifactRef] | None = None,
    memory_ids: list[str] | None = None,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(item: dict[str, object]) -> None:
        key = (
            str(item.get("id", "")),
            str(item.get("path", "")),
            str(item.get("source", "")),
        )
        if key in seen:
            return
        seen.add(key)
        out.append(item)

    for index, item in enumerate(evidence or []):
        path = item.path or ""
        kind = "test" if item.kind == "verification" else ("file" if path else "log")
        add(
            {
                "id": f"{kind}:{path}" if path else f"evidence:{index}",
                "kind": kind,
                "path": path,
                "source": item.source or "context_state",
                "estimated_tokens": max(1, len(item.content) // 4),
                "freshness": item.freshness,
                "reason": "recent_error"
                if item.kind == "verification"
                else "task_related",
            }
        )

    for item in active_file_records or []:
        add(
            {
                "id": f"file:{item.path}",
                "kind": "file",
                "path": item.path,
                "source": item.reason,
                "estimated_tokens": max(1, item.access_count),
                "freshness": item.freshness,
                "reason": "active_file",
            }
        )

    for path in changed_files or []:
        add(
            {
                "id": f"file:{path}",
                "kind": "file",
                "path": path,
                "source": path,
                "estimated_tokens": 1,
                "freshness": "fresh",
                "reason": "changed_file",
            }
        )

    for path in active_files or []:
        add(
            {
                "id": f"file:{path}",
                "kind": "file",
                "path": path,
                "source": path,
                "estimated_tokens": 1,
                "freshness": "unknown",
                "reason": "active_file",
            }
        )

    for index, artifact in enumerate(artifacts or []):
        add(
            {
                "id": f"artifact:{index}:{artifact.path}",
                "kind": "log" if artifact.kind == "tool_output" else artifact.kind,
                "path": artifact.path,
                "source": artifact.summary or artifact.path,
                "estimated_tokens": artifact.visible_tokens or artifact.original_tokens,
                "freshness": "fresh",
                "reason": "recent_error",
            }
        )

    for memory_id in memory_ids or []:
        add(
            {
                "id": memory_id,
                "kind": "memory",
                "path": "",
                "source": "memory_recall",
                "estimated_tokens": 1,
                "freshness": "fresh",
                "reason": "memory_recall",
            }
        )

    if out:
        return out

    for section in [
        "stable_rules",
        "task_state",
        "working_set",
        "recalled_memory",
        "conversation",
    ]:
        for index, value in enumerate(getattr(view, section)):
            out.append(
                {
                    "id": f"{section}:{index}",
                    "kind": section,
                    "path": "",
                    "source": "context_governor",
                    "estimated_tokens": max(1, len(value) // 4),
                    "freshness": "fresh",
                    "reason": "task_related",
                }
            )
    return out


def repair_tool_pairs(selected: list[Message], original: list[Message]) -> list[Message]:
    calls = tool_calls_by_id(original)
    seen: set[str] = set()
    repaired: list[Message] = []
    for message in selected:
        if isinstance(message, AssistantMessage):
            seen.update(block.id for block in message.content if isinstance(block, ToolCall))
        if isinstance(message, ToolResultMessage) and message.tool_call_id not in seen:
            call = calls.get(message.tool_call_id)
            if call is not None:
                repaired.append(AssistantMessage(content=[call], stop_reason="toolUse"))
                seen.add(call.id)
        repaired.append(message)
    return repaired


def tool_calls_by_id(messages: list[Message]) -> dict[str, ToolCall]:
    calls: dict[str, ToolCall] = {}
    for message in messages:
        if not isinstance(message, AssistantMessage):
            continue
        for block in message.content:
            if isinstance(block, ToolCall):
                calls.setdefault(block.id, block)
    return calls


def message_text(message: Message) -> str:
    if isinstance(message, UserMessage):
        if isinstance(message.content, str):
            return message.content
        return "".join(getattr(block, "text", "") for block in message.content)
    if isinstance(message, AssistantMessage):
        return "".join(getattr(block, "text", "") for block in message.content)
    if isinstance(message, ToolResultMessage):
        parts = [f"{message.tool_name} status={message.status}"]
        if message.affected_paths:
            parts.append(f"paths={', '.join(message.affected_paths[:6])}")
        if message.verification:
            parts.append(f"verification={message.verification}")
        if message.error_code:
            parts.append(f"error={message.error_code}")
        artifact = message.metadata.get("artifact_ref")
        if isinstance(artifact, str):
            parts.append(f"artifact={artifact}")
        return "; ".join(parts)
    return ""


def latest_user_text(messages: list[Message]) -> str:
    for message in reversed(messages):
        if isinstance(message, UserMessage):
            return message_text(message)
    return ""


def current_task_goal(context: AgentContext) -> str:
    return context.current_task or latest_user_text(context.messages)


def next_action(context: AgentContext) -> str:
    action = optional_signal(context, "action_intent")
    return action or "continue_current_task"


def verification_state(stale_items: list[str], evidence: list[str]) -> str:
    if any("verification" in item for item in stale_items):
        return "stale"
    if any("verification" in item or "failed" in item for item in evidence):
        return "observed"
    return "unknown"


def optional_signal(context: AgentContext, key: str) -> str | None:
    if not isinstance(context.task_signal, dict):
        return None
    value = context.task_signal.get(key)
    return value if isinstance(value, str) and value else None


def context_mode(context: AgentContext) -> str:
    intent = optional_signal(context, "action_intent") or ""
    recent_error = optional_signal(context, "recent_error_code")
    if recent_error or "debug" in intent or "repair" in intent:
        return "repair"
    if "verify" in intent:
        return "verify"
    text = latest_user_text(context.messages).lower()
    if any(word in text for word in ["why", "how", "explain", "为什么", "解释"]):
        return "qa"
    return "act"


def _compact_evidence_content(content: str) -> str:
    compact = " ".join(content.strip().split())
    if len(compact) <= 400:
        return compact
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    important = [
        line
        for line in lines
        if any(
            marker in line
            for marker in [
                "FAILED",
                "ERROR",
                "Traceback",
                "AssertionError",
                "Exception",
                "Exit code",
            ]
        )
    ]
    if important:
        return " | ".join(important[:6])[:400]
    return f"{len(lines)} lines, {len(content)} chars archived"


def _compact_line(value: object, *, limit: int) -> str:
    return " ".join(str(value or "").strip().split())[:limit]


def _limit_lines(values: list[str], *, count: int, limit: int) -> list[str]:
    out: list[str] = []
    for value in values[:count]:
        text = _compact_line(value, limit=limit)
        if text:
            out.append(text)
    return out


__all__ = [
    "ContextProjection",
    "ContextProjector",
    "context_mode",
    "current_task_goal",
    "latest_user_text",
    "next_action",
    "section_reports",
    "selected_item_summaries",
    "tokens_by_layer",
    "tool_output_tokens",
    "verification_state",
]
