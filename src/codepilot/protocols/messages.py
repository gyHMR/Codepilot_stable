from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from .content import ImageContent, TextContent, ThinkingContent
from .errors import LLMErrorInfo
from .llm import Api, Provider, StopReason, Usage
from .tools import Tool, ToolCall, ToolResultStatus


AssistantBlock = Union[TextContent, ThinkingContent, ToolCall]
UserBlock = Union[TextContent, ImageContent]
ToolResultBlock = Union[TextContent, ImageContent]


@dataclass
class UserMessage:
    """User message."""

    role: str = "user"
    content: Union[str, list[UserBlock]] = ""
    timestamp: int = 0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class AssistantMessage:
    """Normalized assistant message produced by providers."""

    role: str = "assistant"
    content: list[AssistantBlock] = field(default_factory=list)
    api: Api = ""
    provider: Provider = ""
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    stop_reason: StopReason = "stop"
    response_id: Optional[str] = None
    error_message: Optional[str] = None
    error_info: LLMErrorInfo | None = None
    timestamp: int = 0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class ToolResultMessage:
    """Tool execution result message appended back into model context."""

    role: str = "toolResult"
    tool_call_id: str = ""
    tool_name: str = ""
    content: list[ToolResultBlock] = field(default_factory=list)
    status: ToolResultStatus = "success"
    is_error: bool = False
    details: object = None
    timestamp: int = 0
    metadata: dict[str, object] = field(default_factory=dict)


Message = Union[UserMessage, AssistantMessage, ToolResultMessage]


@dataclass
class Context:
    """Request context: system prompt, message history, and available tools."""

    messages: list[Message]
    system_prompt: Optional[str] = None
    tools: Optional[list[Tool]] = None


__all__ = [
    "AssistantBlock",
    "AssistantMessage",
    "Context",
    "Message",
    "ToolResultBlock",
    "ToolResultMessage",
    "UserBlock",
    "UserMessage",
]
