"""
Protocols 子包公共索引。

本包是 Codepilot 的类型契约层，只放跨层共享的稳定数据结构。
这里不放业务逻辑、文件读写、模型调用、工具执行或持久化实现。

子模块分工：
- content.py: 内容块类型（文本、图片、思考）
- messages.py: 消息类型（用户、助手、工具结果）和上下文
- tools.py: 工具定义、工具调用、工具结果
- llm.py: 模型配置、用量统计、流式事件
- events.py: 运行时事件类型定义
- runs.py: 运行结果和状态
- errors.py: 错误信息结构

使用建议：
- 常用稳定协议可以从 codepilot.protocols 直接导入；
- 细分事件、上下文治理等较专门的类型，优先从对应子模块导入。
"""

from .content import ContentBlock, ImageContent, TextContent, ThinkingContent
from .context import (
    ContextFreshness,
    ContextItem,
    ContextReport,
    ContextSectionReport,
    ContextTrust,
    DroppedContextItem,
    DroppedContextReason,
    RepositoryDelta,
    RepositorySnapshot,
)
from .errors import ErrorInfo, ErrorSource, LLMErrorInfo, LLMErrorKind
from .events import (
    AgentEndEvent,
    AgentEvent,
    AgentEventBase,
    AgentEventSink,
    AgentStartEvent,
    ErrorEvent,
    EventEnvelope,
    ensure_runtime_event_type,
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
    TaskSummary,
)
from .tools import (
    Tool,
    ToolCall,
    ToolMetadata,
    ToolResult,
    ToolResultStatus,
    ToolRiskLevel,
)


__all__ = [
    # ── 消息与内容 ──
    "AssistantBlock",
    "AssistantMessage",
    "ContentBlock",
    "Context",
    "ImageContent",
    "Message",
    "TextContent",
    "ThinkingContent",
    "ToolResultBlock",
    "ToolResultMessage",
    "UserBlock",
    "UserMessage",
    # ── 工具 ──
    "Tool",
    "ToolCall",
    "ToolMetadata",
    "ToolResult",
    "ToolResultStatus",
    "ToolRiskLevel",
    # ── LLM ──
    "Api",
    "Cost",
    "Model",
    "ModelCapabilities",
    "Provider",
    "SimpleStreamOptions",
    "StopReason",
    "StreamOptions",
    "ThinkingLevel",
    "Usage",
    # ── Run 结果 ──
    "AgentRunCounters",
    "AgentRunResult",
    "AgentRunStatus",
    "AgentRunStopReason",
    "RunVerification",
    "RunVerificationStatus",
    "TaskSummary",
    # ── 通用事件入口 ──
    "AgentEvent",
    "AgentEventSink",
    "EventEnvelope",
    "ensure_runtime_event_type",
    "RuntimeEvent",
    "RuntimeEventType",
    # ── LLM 流式事件 ──
    "LLMStreamEvent",
    "LLMStreamEventType",
    # ── 错误 ──
    "ErrorInfo",
    "LLMErrorInfo",
    "LLMErrorKind",
]
