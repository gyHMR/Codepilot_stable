from __future__ import annotations

"""
agent_core 的类型定义。

这一层关注“编排”而不是“具体 provider 实现”：
1) 维护 Agent 状态；
2) 定义循环配置；
3) 引用 tools/protocols 拥有的跨层类型，保持依赖方向清晰。
"""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Optional

from codepilot.protocols import (
    AssistantMessage,
    ContextReport,
    ImageContent,
    Message,
    Model,
    TextContent,
    ThinkingLevel,
    ToolCall,
    ToolResultMessage,
)
from codepilot.tools.types import (
    AgentTool,
    AgentToolResult,
)


ToolExecutionMode = Literal["sequential", "parallel"]

# 当前阶段只支持 LLM 消息类型，后续可以扩展 custom message。
AgentMessage = Message


@dataclass
class AgentContext:
    system_prompt: str
    messages: list[AgentMessage]
    tools: list[AgentTool] = field(default_factory=list)
    current_task: str | None = None
    recovered_task: dict[str, object] | None = None


@dataclass(frozen=True)
class ContextPreparationRequest:
    session_id: str | None
    model_context_window: int
    model_max_output_tokens: int
    signal: Any | None = None


@dataclass
class PreparedAgentContext:
    system_prompt: str
    messages: list[AgentMessage]
    tools: list[AgentTool]
    report: ContextReport


PrepareContextFn = Callable[
    [AgentContext, ContextPreparationRequest],
    PreparedAgentContext | Awaitable[PreparedAgentContext],
]


@dataclass
class BeforeToolCallResult:
    block: bool = False
    reason: Optional[str] = None


@dataclass
class AfterToolCallResult:
    content: Optional[list[TextContent | ImageContent]] = None
    details: Any = None
    is_error: Optional[bool] = None


@dataclass
class BeforeToolCallContext:
    assistant_message: AssistantMessage
    tool_call: ToolCall
    args: dict[str, Any]
    context: AgentContext


@dataclass
class AfterToolCallContext:
    assistant_message: AssistantMessage
    tool_call: ToolCall
    args: dict[str, Any]
    result: AgentToolResult
    is_error: bool
    context: AgentContext


@dataclass
class AgentLoopConfig:
    model: Model
    convert_to_llm: Callable[[list[AgentMessage]], list[Message] | Awaitable[list[Message]]]
    transform_context: Optional[
        Callable[[list[AgentMessage], Any | None], list[AgentMessage] | Awaitable[list[AgentMessage]]]
    ] = None
    prepare_context: PrepareContextFn | None = None
    get_api_key: Optional[Callable[[str], str | None | Awaitable[str | None]]] = None
    get_steering_messages: Optional[Callable[[], list[AgentMessage] | Awaitable[list[AgentMessage]]]] = None
    get_follow_up_messages: Optional[Callable[[], list[AgentMessage] | Awaitable[list[AgentMessage]]]] = None
    tool_execution: ToolExecutionMode = "parallel"
    before_tool_call: Optional[
        Callable[[BeforeToolCallContext, Any | None], BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None]]
    ] = None
    after_tool_call: Optional[
        Callable[[AfterToolCallContext, Any | None], AfterToolCallResult | None | Awaitable[AfterToolCallResult | None]]
    ] = None
    reasoning: Optional[ThinkingLevel] = None
    session_id: Optional[str] = None
    max_tool_iterations: int = 12
    max_tool_calls_per_turn: Optional[int] = 8
    allow_unmanaged_tools: bool = False
    repeated_tool_call_limit: int = 3
    retry_enabled: bool = True
    max_model_retries: int = 2
    retry_base_delay_ms: int = 1200
    task_control_enabled: bool = True


@dataclass
class AgentState:
    system_prompt: str
    model: Model
    thinking_level: Literal["off", "minimal", "low", "medium", "high", "xhigh"] = "off"
    tools: list[AgentTool] = field(default_factory=list)
    messages: list[AgentMessage] = field(default_factory=list)
    is_streaming: bool = False
    stream_message: AgentMessage | None = None
    pending_tool_calls: set[str] = field(default_factory=set)
    error: str | None = None
