"""定义模型、Core、Tools 与 Sessions 共享的规范化对话契约。

本模块描述消息、内容块以及模型产生的工具调用意图，是持久化记录和 Provider 转换共同
使用的边界对象；它不执行工具，也不拥有 Run 或 Session 状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, TYPE_CHECKING, Union

from .errors import LLMErrorInfo
from .llm import Api, Provider, StopReason, Usage

if TYPE_CHECKING:
    from .tools import Tool, ToolResultStatus


@dataclass
class TextContent:
    """普通文本内容块，可出现在用户、助手或工具结果消息中。"""

    type: Literal["text"] = "text"
    text: str = ""
    text_signature: str | None = None


@dataclass
class ThinkingContent:
    """支持推理能力的模型产生的思考内容块。"""

    type: Literal["thinking"] = "thinking"
    thinking: str = ""
    thinking_signature: str | None = None
    redacted: bool = False


@dataclass
class ImageContent:
    """图片内容块；``data`` 保存 Base64 数据，``mime_type`` 声明媒体类型。"""

    type: Literal["image"] = "image"
    data: str = ""
    mime_type: str = "image/png"
    source: str | None = None


@dataclass
class ToolCall:
    """Provider 输出经规范化后的工具调用意图。

    ``id`` 是贯穿模型消息、工具执行和 ToolResult 的 ``tool_call_id``；本对象只表达
    调用意图，不代表工具已经通过校验、审批或执行。
    """

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
    """规范对话记录中的用户消息。"""

    role: str = "user"
    content: Union[str, list[UserBlock]] = ""
    timestamp: int = 0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class AssistantMessage:
    """模型 Provider 返回并规范化后的助手消息。

    消息同时携带模型身份、用量、停止原因和结构化错误，是 LLM 层交给 Core 的最终
    消息事实，而不是界面专用展示对象。
    """

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
    """追加回规范对话记录、供模型继续推理的工具观察结果。"""

    role: str = "toolResult"
    tool_call_id: str = ""
    tool_name: str = ""
    content: list[ToolResultBlock] = field(default_factory=list)
    status: ToolResultStatus = "success"
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

        self.status = ensure_tool_result_status(self.status)

    @property
    def is_error(self) -> bool:
        """返回该结果是否属于非成功终态。"""

        return self.status != "success"


Message = Union[UserMessage, AssistantMessage, ToolResultMessage]


@dataclass
class Context:
    """传给 Provider 的模型请求上下文，包含消息链、系统提示词和可见工具。"""

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
