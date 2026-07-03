from __future__ import annotations

# 新手导读：messages.py 定义 User/Assistant/ToolResult 等跨层消息结构。
# 关注点：理解 Agent 对话历史，先从这些消息类型开始。

"""
消息类型定义。

定义了 Agent 对话中涉及的三种消息类型：
- UserMessage: 用户发送的消息（文本 + 可选图片）
- AssistantMessage: LLM 生成的助手回复（文本 + 思考 + 工具调用）
- ToolResultMessage: 工具执行结果消息（回传给 LLM 继续推理）

以及上下文容器 Context 和各消息的内容块联合类型。
"""

from dataclasses import dataclass, field
from typing import Optional, Union

from .content import ImageContent, TextContent, ThinkingContent
from .errors import LLMErrorInfo
from .llm import Api, Provider, StopReason, Usage
from .tools import Tool, ToolCall, ToolResultStatus, ensure_tool_result_status


# ── 内容块联合类型 ──────────────────────────────────────────────

# 助手消息可包含的内容块：文本、思考、工具调用
AssistantBlock = Union[TextContent, ThinkingContent, ToolCall]
# 用户消息可包含的内容块：文本、图片
UserBlock = Union[TextContent, ImageContent]
# 工具结果消息可包含的内容块：文本、图片
ToolResultBlock = Union[TextContent, ImageContent]


@dataclass
class UserMessage:
    """用户消息。

    由用户发送的消息，可以是纯文本字符串，也可以是包含文本和图片的内容块列表。

    Attributes:
        role: 角色标识，固定为 "user"。
        content: 消息内容，可以是纯字符串或内容块列表。
        timestamp: 消息时间戳（毫秒）。
        metadata: 附加元数据字典。
    """

    role: str = "user"
    content: Union[str, list[UserBlock]] = ""
    timestamp: int = 0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class AssistantMessage:
    """助手消息（归一化格式）。

    由各 provider 生成后统一转换为此格式，包含 LLM 的完整回复信息：
    文本内容、思考过程、工具调用请求，以及用量和费用统计。

    Attributes:
        role: 角色标识，固定为 "assistant"。
        content: 内容块列表（文本/思考/工具调用的混合）。
        api: 使用的 API 协议标识。
        provider: 使用的 provider 名称。
        model: 使用的模型 ID。
        usage: 本次调用的 token 用量统计。
        stop_reason: 模型停止生成的原因。
        response_id: provider 返回的响应 ID（可选）。
        error_message: 错误消息（stop_reason 为 "error" 时设置）。
        error_info: 结构化错误信息（stop_reason 为 "error" 时设置）。
        timestamp: 消息时间戳（毫秒）。
        metadata: 附加元数据字典。
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
    """工具执行结果消息。

    工具执行完成后，将结果以此消息格式追加回模型上下文中，
    供 LLM 继续推理或生成最终回答。

    Attributes:
        role: 角色标识，固定为 "toolResult"。
        tool_call_id: 对应的工具调用 ID（与 ToolCall.id 匹配）。
        tool_name: 工具名称。
        content: 结果内容块列表（文本/图片）。
        status: 执行状态（success/error/denied/approval_required/cancelled）。
        is_error: 是否为错误结果。
        error_code: 错误代码（可选）。
        exit_code: 进程退出码（可选，适用于命令行工具）。
        affected_paths: 受影响的文件路径列表。
        workspace_changed: 是否修改了工作区文件。
        diff_summary: 变更摘要（可选）。
        verification: 验证结果字典（可选）。
        details: 附加详情（可选）。
        timestamp: 消息时间戳（毫秒）。
        metadata: 附加元数据字典。
    """

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
        ensure_tool_result_status(self.status)
        if self.is_error and self.status == "success":
            self.status = "error"
        elif self.status != "success":
            self.is_error = True


# 消息联合类型：对话中的任意一条消息
Message = Union[UserMessage, AssistantMessage, ToolResultMessage]


@dataclass
class Context:
    """请求上下文：封装发送给 LLM 的完整请求信息。

    Attributes:
        messages: 消息列表（对话历史）。
        system_prompt: 系统提示词（可选）。
        tools: 可用工具定义列表（可选）。
    """

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
