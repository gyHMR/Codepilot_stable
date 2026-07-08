# 新手导读：包门面文件：集中导出本层最常用的类型和入口，降低学习时的导入成本。
# 关注点：protocols 层是全项目共享的数据契约，原则上不依赖任何内部实现。

"""
Protocols 子包公共索引。

本包是 Codepilot 的类型契约层，只放跨层共享的稳定数据结构。
这里不放业务逻辑、文件读写、模型调用、工具执行或持久化实现。

子模块分工：
- conversation.py: 内容块、消息、上下文和模型工具调用意图
- tools.py: 模型可见工具定义和工具结果
- llm.py: 模型配置、能力和用量统计
- runtime.py: 运行结果、运行状态和运行时事件
- errors.py: 错误信息结构

使用建议：
- 常用稳定协议可以从 codepilot.protocols 直接导入；
- 细分事件、上下文治理等较专门的类型，优先从对应子模块导入。
"""

from .conversation import (
    AssistantBlock,
    AssistantMessage,
    ContentBlock,
    Context,
    ImageContent,
    Message,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultBlock,
    ToolResultMessage,
    UserBlock,
    UserMessage,
)
from .context import (
    ContextArtifactRef,
    ContextCheckpoint,
    ContextFreshness,
    ContextItem,
    ContextPressure,
    ContextPressureLevel,
    ContextReport,
    ContextSectionReport,
    ContextTrust,
    ContextView,
    DroppedContextItem,
    DroppedContextReason,
    RepositoryDelta,
    RepositorySnapshot,
    RunnerPreflightReport,
)
from .commands import (
    AfterToolCallContext,
    AfterToolCallResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    CommandHandler,
    CommandSource,
    LifecycleHook,
    RegisteredCommand,
    SessionCommandContext,
    SessionCommandView,
    SessionLifecycleContext,
    SessionLifecycleView,
    ToolHookContextSnapshot,
)
from .errors import ErrorInfo, ErrorSource, LLMErrorInfo, LLMErrorKind
from .runtime import (
    AgentEndEvent,
    AgentEvent,
    AgentEventBase,
    AgentEventSink,
    AgentRunCounters,
    AgentRunResult,
    AgentRunStatus,
    AgentRunStopReason,
    AgentStartEvent,
    ErrorEvent,
    EventEnvelope,
    ensure_runtime_event_type,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ModelRetryStartEvent,
    PlanSummary,
    RuntimeEvent,
    RuntimeEventType,
    RunSignalsSummary,
    RunSignalsVerificationStatus,
    RunVerification,
    RunVerificationStatus,
    ToolFinishedEvent,
    ToolStartedEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from .llm import (
    Api,
    Cost,
    Model,
    ModelCapabilities,
    Provider,
    StopReason,
    ThinkingLevel,
    Usage,
)
from .tools import (
    PLAN_ITEM_LIMIT,
    Tool,
    ToolResult,
    ToolResultStatus,
    ToolRiskLevel,
    UPDATE_PLAN_TOOL,
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
    "ContextArtifactRef",
    "ContextCheckpoint",
    "ContextFreshness",
    "ContextItem",
    "ContextPressure",
    "ContextPressureLevel",
    "ContextReport",
    "ContextSectionReport",
    "ContextTrust",
    "ContextView",
    "DroppedContextItem",
    "DroppedContextReason",
    "RepositoryDelta",
    "RepositorySnapshot",
    "RunnerPreflightReport",
    # ── 命令与生命周期能力 ──
    "CommandHandler",
    "CommandSource",
    "LifecycleHook",
    "RegisteredCommand",
    "SessionCommandContext",
    "SessionCommandView",
    "SessionLifecycleContext",
    "SessionLifecycleView",
    # ── 工具 ──
    "Tool",
    "ToolCall",
    "ToolResult",
    "ToolResultStatus",
    "ToolRiskLevel",
    "UPDATE_PLAN_TOOL",
    "PLAN_ITEM_LIMIT",
    "AfterToolCallContext",
    "AfterToolCallResult",
    "BeforeToolCallContext",
    "BeforeToolCallResult",
    "ToolHookContextSnapshot",
    # ── LLM ──
    "Api",
    "Cost",
    "Model",
    "ModelCapabilities",
    "Provider",
    "StopReason",
    "ThinkingLevel",
    "Usage",
    # ── Run 结果 ──
    "AgentRunCounters",
    "AgentRunResult",
    "AgentRunStatus",
    "AgentRunStopReason",
    "PlanSummary",
    "RunSignalsSummary",
    "RunSignalsVerificationStatus",
    "RunVerification",
    "RunVerificationStatus",
    # ── 通用事件入口 ──
    "AgentEvent",
    "AgentEventSink",
    "EventEnvelope",
    "ensure_runtime_event_type",
    "RuntimeEvent",
    "RuntimeEventType",
    # ── 错误 ──
    "ErrorInfo",
    "LLMErrorInfo",
    "LLMErrorKind",
]
