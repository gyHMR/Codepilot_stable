from __future__ import annotations

"""Final model-request message repair used by the agent runner."""

from collections import Counter
from dataclasses import dataclass, field

from codepilot.protocols import (
    AssistantMessage,
    Message,
    RunnerPreflightReport,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


TOOL_RESULT_MAX_CHARS = 30_000
TOOL_RESULT_TRUNCATION_NOTICE = "\n...<content truncated>..."


@dataclass(frozen=True)
class ContextPreflightResult:
    messages: list[Message]
    report: RunnerPreflightReport = field(default_factory=RunnerPreflightReport)


def prepare_messages_for_model(
    messages: list[Message],
    *,
    max_messages: int | None = None,
    tool_result_max_chars: int = TOOL_RESULT_MAX_CHARS,
) -> ContextPreflightResult:
    """Return provider-safe messages without mutating the persisted transcript."""

    repaired, dropped_orphans, backfilled = repair_tool_boundaries(messages)
    snipped, outputs_snipped = snip_tool_outputs(
        repaired,
        max_chars=tool_result_max_chars,
    )
    compacted, snipped_messages = keep_legal_tail(snipped, max_messages=max_messages)
    return ContextPreflightResult(
        messages=compacted,
        report=RunnerPreflightReport(
            orphan_tool_results_dropped=dropped_orphans,
            missing_tool_results_backfilled=backfilled,
            tool_outputs_snipped=outputs_snipped,
            snipped_messages=snipped_messages,
        ),
    )


def repair_tool_boundaries(messages: list[Message]) -> tuple[list[Message], int, int]:
    """Drop orphan tool results and synthesize missing results for retained calls."""

    remaining_tool_results = Counter(
        message.tool_call_id
        for message in messages
        if isinstance(message, ToolResultMessage) and message.tool_call_id
    )
    pending: dict[str, ToolCall] = {}
    result: list[Message] = []
    dropped_orphans = 0
    backfilled = 0

    for message in messages:
        if isinstance(message, ToolResultMessage):
            if message.tool_call_id:
                remaining_tool_results[message.tool_call_id] -= 1
            if message.tool_call_id not in pending:
                dropped_orphans += 1
                continue
            pending.pop(message.tool_call_id, None)
            result.append(message)
            continue

        if isinstance(message, AssistantMessage):
            message = _assistant_with_available_tool_calls(
                message,
                remaining_tool_results=remaining_tool_results,
            )
            if not message.content:
                continue
            tool_calls = [
                block
                for block in message.content
                if isinstance(block, ToolCall) and block.id
            ]
            for call in tool_calls:
                pending[call.id] = call
            result.append(message)
            continue

        result.append(message)

    for call in pending.values():
        result.append(
            ToolResultMessage(
                tool_call_id=call.id,
                tool_name=call.name,
                content=[
                    TextContent(
                        text="Error: task was interrupted before this tool returned."
                    )
                ],
                status="error",
                is_error=True,
                error_code="tool_result_missing",
            )
        )
        backfilled += 1

    return result, dropped_orphans, backfilled


def snip_tool_outputs(
    messages: list[Message],
    *,
    max_chars: int,
) -> tuple[list[Message], int]:
    out: list[Message] = []
    snipped = 0
    for message in messages:
        if not isinstance(message, ToolResultMessage):
            out.append(message)
            continue
        total_chars = sum(
            len(block.text)
            for block in message.content
            if isinstance(block, TextContent)
        )
        if total_chars <= max_chars:
            out.append(message)
            continue
        out.append(_snipped_tool_result(message, max_chars=max_chars))
        snipped += 1
    return out, snipped


def keep_legal_tail(
    messages: list[Message],
    *,
    max_messages: int | None,
) -> tuple[list[Message], int]:
    if max_messages is None or max_messages <= 0 or len(messages) <= max_messages:
        return messages, 0

    tail = messages[-max_messages:]
    while tail and isinstance(tail[0], ToolResultMessage):
        tail = tail[1:]
    repaired, dropped, backfilled = repair_tool_boundaries(tail)
    # If cutting the head caused synthetic backfills, keep a slightly larger tail
    # rather than giving the model fabricated results when the real pair is nearby.
    if backfilled and max_messages < len(messages):
        return keep_legal_tail(messages, max_messages=max_messages + backfilled)
    return repaired, len(messages) - len(tail) + dropped


def _assistant_with_available_tool_calls(
    message: AssistantMessage,
    *,
    remaining_tool_results: Counter[str],
) -> AssistantMessage:
    tool_calls = [
        block for block in message.content if isinstance(block, ToolCall) and block.id
    ]
    if not tool_calls:
        return message
    content = [
        block
        for block in message.content
        if not isinstance(block, ToolCall) or remaining_tool_results[block.id] > 0
    ]
    if len(content) == len(message.content):
        return message
    return AssistantMessage(
        role=message.role,
        content=content,
        api=message.api,
        provider=message.provider,
        model=message.model,
        usage=message.usage,
        stop_reason=(
            message.stop_reason
            if any(isinstance(block, ToolCall) for block in content)
            else "stop"
        ),
        response_id=message.response_id,
        error_message=message.error_message,
        error_info=message.error_info,
        timestamp=message.timestamp,
        metadata=dict(message.metadata),
    )


def _snipped_tool_result(
    message: ToolResultMessage,
    *,
    max_chars: int,
) -> ToolResultMessage:
    content = []
    remaining = max_chars
    for block in message.content:
        if isinstance(block, TextContent):
            if remaining <= 0:
                continue
            text = block.text
            if len(text) > remaining:
                content.append(
                    TextContent(text=text[:remaining] + TOOL_RESULT_TRUNCATION_NOTICE)
                )
                remaining = 0
            else:
                content.append(block)
                remaining -= len(text)
            continue
        content.append(block)
    return ToolResultMessage(
        role=message.role,
        tool_call_id=message.tool_call_id,
        tool_name=message.tool_name,
        content=content,
        status=message.status,
        is_error=message.is_error,
        approved=message.approved,
        approval_id=message.approval_id,
        error_code=message.error_code,
        exit_code=message.exit_code,
        affected_paths=list(message.affected_paths),
        workspace_changed=message.workspace_changed,
        diff_summary=message.diff_summary,
        verification=dict(message.verification) if message.verification else None,
        details=message.details,
        timestamp=message.timestamp,
        metadata={**message.metadata, "context_snipped": True},
    )


__all__ = [
    "ContextPreflightResult",
    "TOOL_RESULT_MAX_CHARS",
    "TOOL_RESULT_TRUNCATION_NOTICE",
    "keep_legal_tail",
    "prepare_messages_for_model",
    "repair_tool_boundaries",
    "snip_tool_outputs",
]
