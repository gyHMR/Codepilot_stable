"""
Protocols 子包统一导出模块。

本包定义了 Codepilot 系统中所有核心数据结构的协议/类型，
是整个项目的"类型契约层"，各模块依赖此包进行类型交互。

子模块分工：
- content.py: 内容块类型（文本、图片、思考）
- messages.py: 消息类型（用户、助手、工具结果）和上下文
- tools.py: 工具定义、工具调用、工具结果
- llm.py: 模型配置、用量统计、流式事件
- events.py: 运行时事件类型定义
- runs.py: 运行结果和状态
- errors.py: 错误信息结构
"""

from .content import ContentBlock, ImageContent, TextContent, ThinkingContent
from .errors import ErrorInfo, ErrorSource, LLMErrorInfo, LLMErrorKind
from .events import (
    AgentEndEvent,
    AgentEvent,
    AgentEventBase,
    AgentEventSink,
    AgentStartEvent,
    ErrorEvent,
    EventEnvelope,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ModelRetryStartEvent,
    RuntimeEvent,
    RuntimeEventType,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from .llm import (
    Api,
    Cost,
    LLMStreamEvent,
    LLMStreamEventType,
    Model,
    ModelCapabilities,
    Provider,
    SimpleStreamOptions,
    StopReason,
    StreamOptions,
    ThinkingLevel,
    Usage,
)
from .messages import (
    AssistantBlock,
    AssistantMessage,
    Context,
    Message,
    ToolResultBlock,
    ToolResultMessage,
    UserBlock,
    UserMessage,
)
from .runs import (
    AgentRunCounters,
    AgentRunResult,
    AgentRunStatus,
    AgentRunStopReason,
    RunVerification,
    RunVerificationStatus,
)
from .tools import (
    Tool,
    ToolCall,
    ToolMetadata,
    ToolResult,
    ToolResultStatus,
    ToolRiskLevel,
    ToolSpec,
)


__all__ = [
    # ── 事件相关 ──
    "AgentEvent",
    "AgentEventBase",
    "AgentEventSink",
    "AgentStartEvent",
    "AgentEndEvent",
    "ErrorEvent",
    "EventEnvelope",
    "MessageEndEvent",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "ModelRetryStartEvent",
    "RuntimeEvent",
    "RuntimeEventType",
    "ToolExecutionEndEvent",
    "ToolExecutionStartEvent",
    "ToolExecutionUpdateEvent",
    "TurnEndEvent",
    "TurnStartEvent",
    # ── 运行结果相关 ──
    "AgentRunCounters",
    "AgentRunResult",
    "AgentRunStatus",
    "AgentRunStopReason",
    "RunVerification",
    "RunVerificationStatus",
    # ── LLM 相关 ──
    "Api",
    "Cost",
    "LLMStreamEvent",
    "LLMStreamEventType",
    "Model",
    "ModelCapabilities",
    "Provider",
    "SimpleStreamOptions",
    "StopReason",
    "StreamOptions",
    "ThinkingLevel",
    "Usage",
    # ── 消息相关 ──
    "AssistantBlock",
    "AssistantMessage",
    "Context",
    "Message",
    "ToolResultBlock",
    "ToolResultMessage",
    "UserBlock",
    "UserMessage",
    # ── 内容块相关 ──
    "ContentBlock",
    "ImageContent",
    "TextContent",
    "ThinkingContent",
    # ── 工具相关 ──
    "Tool",
    "ToolCall",
    "ToolMetadata",
    "ToolResult",
    "ToolResultStatus",
    "ToolRiskLevel",
    "ToolSpec",
    # ── 错误相关 ──
    "ErrorInfo",
    "ErrorSource",
    "LLMErrorInfo",
    "LLMErrorKind",
]
