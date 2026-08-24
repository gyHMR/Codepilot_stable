"""提供对话记录中未结工具调用和最后助手消息的纯查询函数。"""

from __future__ import annotations

from collections.abc import Sequence

from codepilot.protocols import AssistantMessage, Message, ToolCall, ToolResultMessage


def message_groups(messages: Sequence[Message]) -> tuple[tuple[Message, ...], ...]:
    """Group one assistant tool-call batch with its contiguous tool results."""

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
    return tuple(groups)


def tool_batch_call_ids(group: Sequence[Message]) -> frozenset[str]:
    """Return ToolCall IDs when a group starts with an assistant tool batch."""

    if not group or not isinstance(group[0], AssistantMessage):
        return frozenset()
    return frozenset(
        block.id for block in group[0].content if isinstance(block, ToolCall)
    )


def is_closed_tool_batch(group: Sequence[Message]) -> bool:
    """Return whether every ToolCall in a grouped batch has exactly one result."""

    call_ids = tool_batch_call_ids(group)
    if not call_ids:
        return False
    result_ids = [
        message.tool_call_id
        for message in group[1:]
        if isinstance(message, ToolResultMessage)
    ]
    return len(result_ids) == len(call_ids) and frozenset(result_ids) == call_ids


def latest_unconsumed_tool_batch(
    groups: Sequence[Sequence[Message]],
) -> int | None:
    """Find the newest closed tool batch with no later assistant response."""

    has_later_assistant = False
    for index in range(len(groups) - 1, -1, -1):
        group = groups[index]
        if is_closed_tool_batch(group) and not has_later_assistant:
            return index
        if any(isinstance(message, AssistantMessage) for message in group):
            has_later_assistant = True
    return None


def unsettled_tool_calls(messages: Sequence[Message]) -> tuple[ToolCall, ...]:
    """返回尚未被 ToolResultMessage 结算的工具调用。"""
    calls: dict[str, ToolCall] = {}
    settled: set[str] = set()
    for message in messages:
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolCall):
                    calls[block.id] = block
        elif isinstance(message, ToolResultMessage):
            settled.add(message.tool_call_id)
    return tuple(call for call_id, call in calls.items() if call_id not in settled)


def last_assistant_message(messages: Sequence[Message]) -> AssistantMessage | None:
    """返回对话记录中的最后一条助手消息。"""
    return next(
        (
            message
            for message in reversed(messages)
            if isinstance(message, AssistantMessage)
        ),
        None,
    )


__all__ = [
    "is_closed_tool_batch",
    "last_assistant_message",
    "latest_unconsumed_tool_batch",
    "message_groups",
    "tool_batch_call_ids",
    "unsettled_tool_calls",
]
