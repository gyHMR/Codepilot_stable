from __future__ import annotations

"""
agent_core 的类型定义。

这一层关注“编排”而不是“具体 provider 实现”：
1) 维护 Agent 状态；
2) 定义循环配置；
3) 引用 tools/protocols 拥有的跨层类型，保持依赖方向清晰。
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Optional, cast

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
    UserMessage,
)
from codepilot.tools.types import (
    AgentTool,
    AgentToolResult,
)


ToolExecutionMode = Literal["sequential", "parallel"]
_TOOL_EXECUTION_MODES = frozenset({"sequential", "parallel"})
_THINKING_LEVELS = frozenset({"minimal", "low", "medium", "high", "xhigh"})
_AGENT_THINKING_LEVELS = frozenset({"off", *_THINKING_LEVELS})

# 当前阶段只支持 LLM 消息类型，后续可以扩展 custom message。
AgentMessage = Message


@dataclass
class AgentContext:
    system_prompt: str
    messages: list[AgentMessage]
    tools: list[AgentTool] = field(default_factory=list)
    current_task: str | None = None
    task_recovery_projection: dict[str, object] | None = None
    task_signal: dict[str, object] | None = None

    def __post_init__(self) -> None:
        self.system_prompt = _clean_core_text(self.system_prompt)
        self.messages = _copy_messages(self.messages, field_name="messages")
        self.tools = _copy_tools(self.tools, field_name="tools")
        self.current_task = _optional_core_text(self.current_task)
        self.task_recovery_projection = _copy_optional_dict(
            self.task_recovery_projection,
            field_name="task_recovery_projection",
        )
        self.task_signal = _copy_optional_dict(self.task_signal, field_name="task_signal")


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
    task_planner_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model, Model):
            raise TypeError("AgentLoopConfig model must be Model")
        if not callable(self.convert_to_llm):
            raise TypeError("AgentLoopConfig convert_to_llm must be callable")
        _ensure_optional_callable(self.transform_context, "transform_context")
        _ensure_optional_callable(self.prepare_context, "prepare_context")
        _ensure_optional_callable(self.get_api_key, "get_api_key")
        _ensure_optional_callable(self.get_steering_messages, "get_steering_messages")
        _ensure_optional_callable(self.get_follow_up_messages, "get_follow_up_messages")
        _ensure_optional_callable(self.before_tool_call, "before_tool_call")
        _ensure_optional_callable(self.after_tool_call, "after_tool_call")
        self.tool_execution = _ensure_tool_execution_mode(self.tool_execution)
        self.reasoning = _ensure_optional_thinking_level(self.reasoning)
        self.session_id = _optional_core_text(self.session_id)
        self.max_tool_iterations = _ensure_positive_int(
            self.max_tool_iterations,
            field_name="max_tool_iterations",
        )
        self.max_tool_calls_per_turn = _ensure_optional_positive_int(
            self.max_tool_calls_per_turn,
            field_name="max_tool_calls_per_turn",
        )
        self.allow_unmanaged_tools = _ensure_bool(
            self.allow_unmanaged_tools,
            field_name="allow_unmanaged_tools",
        )
        self.repeated_tool_call_limit = _ensure_non_negative_int(
            self.repeated_tool_call_limit,
            field_name="repeated_tool_call_limit",
        )
        self.retry_enabled = _ensure_bool(self.retry_enabled, field_name="retry_enabled")
        self.max_model_retries = _ensure_non_negative_int(
            self.max_model_retries,
            field_name="max_model_retries",
        )
        self.retry_base_delay_ms = _ensure_non_negative_int(
            self.retry_base_delay_ms,
            field_name="retry_base_delay_ms",
        )
        self.task_control_enabled = _ensure_bool(
            self.task_control_enabled,
            field_name="task_control_enabled",
        )
        self.task_planner_enabled = _ensure_bool(
            self.task_planner_enabled,
            field_name="task_planner_enabled",
        )


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

    def __post_init__(self) -> None:
        if not isinstance(self.model, Model):
            raise TypeError("AgentState model must be Model")
        self.system_prompt = _clean_core_text(self.system_prompt)
        self.thinking_level = _ensure_agent_thinking_level(self.thinking_level)
        self.tools = _copy_tools(self.tools, field_name="tools")
        self.messages = _copy_messages(self.messages, field_name="messages")
        self.is_streaming = _ensure_bool(self.is_streaming, field_name="is_streaming")
        if self.stream_message is not None and not isinstance(
            self.stream_message,
            (UserMessage, AssistantMessage, ToolResultMessage),
        ):
            raise TypeError("AgentState stream_message must be AgentMessage or None")
        self.pending_tool_calls = _copy_text_set(
            self.pending_tool_calls,
            field_name="pending_tool_calls",
        )
        self.error = _optional_core_text(self.error)


def _clean_core_text(value: object) -> str:
    return str(value) if value is not None else ""


def _optional_core_text(value: object) -> str | None:
    text = _clean_core_text(value).strip()
    return text or None


def _copy_messages(value: object, *, field_name: str) -> list[AgentMessage]:
    if not isinstance(value, list):
        raise TypeError(f"AgentContext {field_name} must be a list")
    messages: list[AgentMessage] = []
    for message in value:
        if not isinstance(message, (UserMessage, AssistantMessage, ToolResultMessage)):
            raise TypeError(f"AgentContext {field_name} entries must be AgentMessage")
        messages.append(message)
    return messages


def _copy_tools(value: object, *, field_name: str) -> list[AgentTool]:
    if not isinstance(value, list):
        raise TypeError(f"AgentContext {field_name} must be a list")
    tools: list[AgentTool] = []
    for tool in value:
        if not isinstance(tool, AgentTool):
            raise TypeError(f"AgentContext {field_name} entries must be AgentTool")
        tools.append(tool)
    return tools


def _copy_optional_dict(
    value: object,
    *,
    field_name: str,
) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"AgentContext {field_name} must be a dict or None")
    return deepcopy(value)


def _ensure_tool_execution_mode(value: object) -> ToolExecutionMode:
    text = str(value).strip() if value is not None else ""
    if text not in _TOOL_EXECUTION_MODES:
        raise ValueError(f"Unknown tool_execution mode: {value}")
    return cast(ToolExecutionMode, text)


def _ensure_optional_thinking_level(value: object) -> ThinkingLevel | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("AgentLoopConfig reasoning must be a thinking level or None")
    text = str(value).strip()
    if text not in _THINKING_LEVELS:
        raise ValueError(f"Unknown reasoning level: {value}")
    return cast(ThinkingLevel, text)


def _ensure_agent_thinking_level(value: object) -> Literal["off", "minimal", "low", "medium", "high", "xhigh"]:
    if isinstance(value, bool):
        raise TypeError("thinking_level must be a valid agent thinking level")
    text = str(value).strip() if value is not None else ""
    if text not in _AGENT_THINKING_LEVELS:
        raise ValueError(f"Unknown thinking_level: {value}")
    return cast(Literal["off", "minimal", "low", "medium", "high", "xhigh"], text)


def _copy_text_set(value: object, *, field_name: str) -> set[str]:
    if not isinstance(value, set):
        raise TypeError(f"AgentState {field_name} must be a set")
    return {text for item in value if (text := str(item).strip())}


def _ensure_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"AgentLoopConfig {field_name} must be bool")
    return value


def _ensure_optional_callable(value: object, field_name: str) -> None:
    if value is not None and not callable(value):
        raise TypeError(f"{field_name} must be callable or None")


def _ensure_non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"AgentLoopConfig {field_name} must be int")
    if value < 0:
        raise ValueError(f"AgentLoopConfig {field_name} must be non-negative")
    return value


def _ensure_positive_int(value: object, *, field_name: str) -> int:
    integer = _ensure_non_negative_int(value, field_name=field_name)
    if integer <= 0:
        raise ValueError(f"AgentLoopConfig {field_name} must be positive")
    return integer


def _ensure_optional_positive_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _ensure_positive_int(value, field_name=field_name)
