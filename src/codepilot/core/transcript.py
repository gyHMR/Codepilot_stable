"""提供对话记录中未结工具调用和最后助手消息的纯查询函数。"""

from __future__ import annotations

from collections.abc import Sequence

from codepilot.protocols import AssistantMessage, Message, ToolCall, ToolResultMessage


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


__all__ = ["last_assistant_message", "unsettled_tool_calls"]
