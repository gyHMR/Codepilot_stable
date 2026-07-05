from __future__ import annotations

# 新手导读：conversation.py 定义模型与工具之间能看见的对话事实。
# 关注点：这里描述消息、内容块和模型发出的工具调用意图，不执行工具。

"""Conversation-level protocol DTOs shared across layers."""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, TYPE_CHECKING, Union

from .errors import LLMErrorInfo
from .llm import Api, Provider, StopReason, Usage

if TYPE_CHECKING:
    from .tools import Tool, ToolResultStatus


@dataclass
class TextContent:
    """Plain text content block."""

    type: Literal["text"] = "text"
    text: str = ""
    text_signature: str | None = None


@dataclass
class ThinkingContent:
    """Reasoning/thinking content block emitted by capable models."""

    type: Literal["thinking"] = "thinking"
    thinking: str = ""
    thinking_signature: str | None = None
    redacted: bool = False


@dataclass
class ImageContent:
    """Base64 encoded image content block."""

    type: Literal["image"] = "image"
    data: str = ""
    mime_type: str = "image/png"
    source: str | None = None


@dataclass
class ToolCall:
    """Normalized tool-call intent emitted by the model."""

    type: Literal["toolCall"] = "toolCall"
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str | None = None
    index: int | None = None
    provider: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


ContentBlock = Union[TextContent, ThinkingContent, ImageContent]
AssistantBlock = Union[TextContent, ThinkingContent, ToolCall]
UserBlock = Union[TextContent, ImageContent]
ToolResultBlock = Union[TextContent, ImageContent]


@dataclass
class UserMessage:
    """User message in the canonical transcript."""

    role: str = "user"
    content: Union[str, list[UserBlock]] = ""
    timestamp: int = 0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class AssistantMessage:
    """Normalized assistant message returned by a model provider."""

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
    """Tool observation appended back to the transcript for model reasoning."""

    role: str = "toolResult"
    tool_call_id: str = ""
    tool_name: str = ""
    content: list[ToolResultBlock] = field(default_factory=list)
    status: ToolResultStatus = "success"
    is_error: bool = False
    approved: bool = True
    approval_id: str | None = None
    error_code: str | None = None
    exit_code: int | None = None
    affected_paths: list[str] = field(default_factory=list)
    workspace_changed: bool | None = None
    diff_summary: str | None = None
    verification: dict[str, object] | None = None
    details: object = None
    timestamp: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from .tools import ensure_tool_result_status

        ensure_tool_result_status(self.status)
        if self.is_error and self.status == "success":
            self.status = "error"
        elif self.status != "success":
            self.is_error = True


Message = Union[UserMessage, AssistantMessage, ToolResultMessage]


@dataclass
class Context:
    """Model request context: transcript, system prompt, and visible tools."""

    messages: list[Message]
    system_prompt: Optional[str] = None
    tools: Optional[list[Tool]] = None


__all__ = [
    "AssistantBlock",
    "AssistantMessage",
    "ContentBlock",
    "Context",
    "ImageContent",
    "Message",
    "TextContent",
    "ThinkingContent",
    "ToolCall",
    "ToolResultBlock",
    "ToolResultMessage",
    "UserBlock",
    "UserMessage",
]
