from __future__ import annotations

# 新手导读：core/types.py 定义 AgentContext、AgentLoopConfig、AgentState 和工具 hook 上下文。
# 关注点：这些类型是 core 主循环的输入面板，读懂它们就能理解一次 run 能拿到哪些能力。

"""
agent_core 的类型定义模块
========================

本模块是 Codepilot Agent 核心层的类型基础，定义了 Agent 运行所需的
所有数据结构和配置类型。

设计原则：
    这一层关注”编排”而不是”具体 provider 实现”：
    1) 维护 Agent 状态（AgentState）；
    2) 定义循环配置（AgentLoopConfig）；
    3) 定义上下文结构（AgentContext）；
    4) 引用 tools/protocols 拥有的跨层类型，保持依赖方向清晰。

主要类型：
    - AgentContext: Agent 执行上下文，包含系统提示词、消息列表、工具列表等
    - AgentLoopConfig: Agent 循环配置，控制重试、工具执行模式等行为
    - AgentState: Agent 运行时状态，跟踪流式消息、待处理工具调用等
    - AgentMessage: 消息类型别名，当前等同于 Message
    - BeforeToolCallContext / AfterToolCallContext: 工具调用钩子上下文
    - BeforeToolCallResult / AfterToolCallResult: 工具调用钩子结果

辅助函数：
    - _ensure_*: 类型校验函数，确保数据类字段在赋值时类型正确
    - _copy_*: 防御性拷贝函数，避免外部修改影响内部状态
    - _clean_* / _optional_*: 文本清理函数
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
from codepilot.tools.contracts import (
    AgentTool,
    AgentToolResult,
)
from .task_control.contracts import (
    PlanningBudgetProfile,
    TaskMode,
    ensure_planning_budget_profile,
    ensure_task_mode,
)


# ── 类型别名与常量 ──────────────────────────────────────────────

# 工具执行模式：串行（sequential）或并行（parallel）
ToolExecutionMode = Literal["sequential", "parallel"]

# 合法的工具执行模式集合（用于校验）
_TOOL_EXECUTION_MODES = frozenset({"sequential", "parallel"})

# LLM 推理思考级别（不包含 “off”，用于 AgentLoopConfig.reasoning）
_THINKING_LEVELS = frozenset({"minimal", "low", "medium", "high", "xhigh"})

# Agent 状态中的思考级别（包含 “off”，表示关闭推理）
_AGENT_THINKING_LEVELS = frozenset({"off", *_THINKING_LEVELS})

# 消息类型别名：当前阶段只支持 LLM 消息类型，后续可以扩展 custom message
AgentMessage = Message


@dataclass
class AgentContext:
    """Agent 执行上下文：在一次 Run 的生命周期内，封装所有 LLM 调用所需的数据。

    这是 Agent 核心层与 LLM 之间的数据桥梁，每次调用 LLM 前都会构建一个
    AgentContext 快照，确保循环过程中外部修改不会影响正在进行的推理。

    Attributes:
        system_prompt: 系统提示词，定义 Agent 的角色和行为准则。
            示例: "你是一个有帮助的编程助手。"
        messages: 消息列表，包含历史对话和本次运行的消息。
            消息类型包括: UserMessage（用户消息）、AssistantMessage（助手消息）、
            ToolResultMessage（工具执行结果）。
        tools: 可用工具列表，定义 Agent 可以调用的工具及其参数格式。
            每个 AgentTool 包含 name、description、parameters 和 execute 函数。
        current_task: 当前任务上下文（Markdown 格式），由 TaskController 渲染。
            包含任务目标、步骤进度、验收标准等信息，注入系统提示词供 LLM 参考。
        task_recovery_projection: 会话持久化的任务恢复投影。
            当会话中断后恢复时，用于重建 TaskState，避免丢失任务进度。
        task_signal: 任务控制信号，由 TaskController 输出。
            包含当前步骤、阶段、下一步动作等轻量级信息，供上下文和记忆模块使用。
    """
    system_prompt: str                                    # 系统提示词
    messages: list[AgentMessage]                          # 消息列表（历史 + 本次）
    tools: list[AgentTool] = field(default_factory=list)  # 可用工具列表
    current_task: str | None = None                       # 当前任务上下文（Markdown）
    task_recovery_projection: dict[str, object] | None = None  # 任务恢复投影
    task_signal: dict[str, object] | None = None          # 任务控制信号

    def __post_init__(self) -> None:
        """初始化后校验：对所有字段进行类型检查和防御性拷贝。"""
        # 清理系统提示词（确保为字符串）
        self.system_prompt = _clean_core_text(self.system_prompt)
        # 拷贝消息列表（避免外部引用共享可变对象）
        self.messages = _copy_messages(self.messages, field_name="messages")
        # 拷贝工具列表
        self.tools = _copy_tools(self.tools, field_name="tools")
        # 清理可选文本字段
        self.current_task = _optional_core_text(self.current_task)
        # 深拷贝可选字典（避免外部修改影响内部状态）
        self.task_recovery_projection = _copy_optional_dict(
            self.task_recovery_projection,
            field_name="task_recovery_projection",
        )
        self.task_signal = _copy_optional_dict(self.task_signal, field_name="task_signal")


@dataclass(frozen=True)
class ContextPreparationRequest:
    """上下文准备请求：携带模型能力信息，供 prepare_context 回调使用。

    prepare_context 回调可以根据这些信息决定如何裁剪消息列表、
    是否需要压缩上下文等。

    Attributes:
        session_id: 当前会话标识符。
        model_context_window: 模型的上下文窗口大小（token 数）。
        model_max_output_tokens: 模型的最大输出 token 数。
        signal: 可选的取消信号，用于中断长时间的上下文准备操作。
    """
    session_id: str | None                      # 会话标识符
    model_context_window: int                   # 模型上下文窗口大小
    model_max_output_tokens: int                # 模型最大输出 token 数
    signal: Any | None = None                   # 取消信号


@dataclass
class PreparedAgentContext:
    """已准备的上下文：prepare_context 回调的返回值。

    经过上下文准备后，系统提示词、消息列表和工具列表可能已经被裁剪或变换，
    同时附带一份报告（ContextReport）说明做了哪些处理。

    Attributes:
        system_prompt: 处理后的系统提示词。
        messages: 处理后的消息列表（可能已裁剪或压缩）。
        tools: 处理后的工具列表。
        report: 上下文准备报告，记录裁剪了多少消息、节省了多少 token 等。
    """
    system_prompt: str                          # 处理后的系统提示词
    messages: list[AgentMessage]                # 处理后的消息列表
    tools: list[AgentTool]                      # 处理后的工具列表
    report: ContextReport                       # 上下文准备报告


# 上下文准备函数类型：接收原始上下文和准备请求，返回准备后的上下文
# 支持同步和异步两种调用方式
PrepareContextFn = Callable[
    [AgentContext, ContextPreparationRequest],
    PreparedAgentContext | Awaitable[PreparedAgentContext],
]


@dataclass
class BeforeToolCallResult:
    """工具调用前钩子的返回值：决定是否拦截本次工具调用。

    Attributes:
        block: True 表示拦截（阻止工具执行），False 表示放行。
        reason: 拦截原因（仅 block=True 时有意义），会传递给用户。
    """
    block: bool = False                # 是否拦截
    reason: Optional[str] = None       # 拦截原因


@dataclass
class AfterToolCallResult:
    """工具调用后钩子的返回值：可修改工具执行结果。

    所有字段都是可选的，只修改非 None 的字段。

    Attributes:
        content: 替换工具结果的文本/图片内容。
        details: 替换工具结果的详细信息。
        is_error: 覆盖工具结果的错误状态。
    """
    content: Optional[list[TextContent | ImageContent]] = None  # 替换内容
    details: Any = None                                          # 替换详情
    is_error: Optional[bool] = None                              # 覆盖错误状态


@dataclass
class BeforeToolCallContext:
    """工具调用前钩子的上下文：包含本次工具调用的完整信息。

    传递给 before_tool_call 回调，用于：
    - 项目规则检查（如禁止某些工具）
    - 扩展策略（如记录审计日志）
    - 临时禁用特定工具

    Attributes:
        assistant_message: 触发工具调用的助手消息。
        tool_call: 即将执行的工具调用信息（名称、参数等）。
        args: 解析后的工具参数字典。
        context: 当前 Agent 上下文快照。
    """
    assistant_message: AssistantMessage   # 触发工具调用的助手消息
    tool_call: ToolCall                   # 工具调用信息
    args: dict[str, Any]                  # 解析后的参数
    context: AgentContext                  # 当前上下文快照


@dataclass
class AfterToolCallContext:
    """工具调用后钩子的上下文：包含工具调用的输入和执行结果。

    传递给 after_tool_call 回调，用于：
    - 结果后处理（如脱敏、格式化）
    - 审计日志记录
    - 错误恢复策略

    Attributes:
        assistant_message: 触发工具调用的助手消息。
        tool_call: 已执行的工具调用信息。
        args: 工具参数字典。
        result: 工具执行结果。
        is_error: 工具执行是否出错。
        context: 当前 Agent 上下文快照。
    """
    assistant_message: AssistantMessage   # 触发工具调用的助手消息
    tool_call: ToolCall                   # 工具调用信息
    args: dict[str, Any]                  # 工具参数
    result: AgentToolResult               # 工具执行结果
    is_error: bool                        # 是否出错
    context: AgentContext                  # 当前上下文快照


@dataclass
class AgentLoopConfig:
    """Agent 循环配置：控制 Agent 执行循环的所有行为参数。

    这是 Agent 核心循环的"控制面板"，由 Agent._start_run() 在每次运行时构建，
    传递给 run_agent_loop / run_agent_loop_continue 使用。

    配置分为以下几组：
    1. 模型与消息转换: model, convert_to_llm, transform_context
    2. 上下文准备: prepare_context, get_api_key
    3. 消息注入: get_steering_messages, get_follow_up_messages
    4. 工具执行: tool_execution, before_tool_call, after_tool_call
    5. 推理控制: reasoning（思考深度）
    6. 安全限制: max_tool_iterations, max_tool_calls_per_turn, repeated_tool_call_limit
    7. 重试策略: retry_enabled, max_model_retries, retry_base_delay_ms
    8. 任务控制: task_control_enabled, task_mode, max_task_replans_per_run

    Attributes:
        model: 使用的 LLM 模型信息（包含 provider、api、context_window 等）。
        convert_to_llm: 消息转换函数，将 AgentMessage 列表转换为 LLM 可消费的格式。
        transform_context: 上下文变换钩子，发送前对消息进行裁剪、脱敏等处理。
        prepare_context: 上下文准备回调，在每次 LLM 调用前执行（如压缩长对话）。
        get_api_key: API Key 获取回调，支持动态获取密钥（如从环境变量或密钥管理器）。
        get_steering_messages: 引导消息获取回调，在每轮迭代开始前调用。
            引导消息用于在工具调用过程中向 Agent 提供额外指示或纠正方向。
        get_follow_up_messages: 后续消息获取回调，在当前轮次结束后调用。
            后续消息用于补充信息或触发下一轮对话。
        tool_execution: 工具执行模式，"parallel"（并行）或 "sequential"（串行）。
        before_tool_call: 工具调用前钩子，可拦截工具执行（用于项目规则、扩展策略）。
        after_tool_call: 工具调用后钩子，可修改工具结果（用于后处理、审计）。
        reasoning: LLM 推理思考级别（minimal/low/medium/high/xhigh），None 表示关闭。
        session_id: 会话标识符，用于关联日志和持久化数据。
        max_tool_iterations: 单次运行允许的最大工具迭代次数（防止无限循环）。
        max_tool_calls_per_turn: 单轮允许的最大工具调用数量，None 表示不限制。
        allow_unmanaged_tools: 是否允许执行非 ToolRuntime 托管的工具。
        repeated_tool_call_limit: 连续重复相同工具调用的允许次数（检测死循环）。
        retry_enabled: 是否启用 LLM 调用失败重试。
        max_model_retries: LLM 调用最大重试次数。
        retry_base_delay_ms: 重试基础延迟（毫秒），实际延迟按指数退避计算。
        task_control_enabled: 是否启用任务控制器（TaskController）。
        task_mode: 用户选择的任务模式（read/edit/plan）。
        max_task_replans_per_run: 单次任务运行允许的最大局部重新规划次数。
    """

    # ── 模型与消息转换 ──────────────────────────────────────────
    model: Model                                                              # LLM 模型信息
    convert_to_llm: Callable[[list[AgentMessage]], list[Message] | Awaitable[list[Message]]]  # 消息转换函数
    transform_context: Optional[
        Callable[[list[AgentMessage], Any | None], list[AgentMessage] | Awaitable[list[AgentMessage]]]
    ] = None                                                                  # 上下文变换钩子

    # ── 上下文准备 ──────────────────────────────────────────────
    prepare_context: PrepareContextFn | None = None                           # 上下文准备回调
    get_api_key: Optional[Callable[[str], str | None | Awaitable[str | None]]] = None  # API Key 获取回调

    # ── 消息注入 ────────────────────────────────────────────────
    get_steering_messages: Optional[Callable[[], list[AgentMessage] | Awaitable[list[AgentMessage]]]] = None  # 引导消息回调
    get_follow_up_messages: Optional[Callable[[], list[AgentMessage] | Awaitable[list[AgentMessage]]]] = None  # 后续消息回调

    # ── 工具执行 ────────────────────────────────────────────────
    tool_execution: ToolExecutionMode = "parallel"                            # 工具执行模式
    before_tool_call: Optional[
        Callable[[BeforeToolCallContext, Any | None], BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None]]
    ] = None                                                                  # 工具调用前钩子
    after_tool_call: Optional[
        Callable[[AfterToolCallContext, Any | None], AfterToolCallResult | None | Awaitable[AfterToolCallResult | None]]
    ] = None                                                                  # 工具调用后钩子

    # ── 推理控制 ────────────────────────────────────────────────
    reasoning: Optional[ThinkingLevel] = None                                 # 推理思考级别

    # ── 会话标识 ────────────────────────────────────────────────
    session_id: Optional[str] = None                                          # 会话标识符

    # ── 安全限制 ────────────────────────────────────────────────
    max_tool_iterations: int = 12                                             # 最大工具迭代次数
    max_tool_calls_per_turn: Optional[int] = 8                                # 单轮最大工具调用数
    allow_unmanaged_tools: bool = False                                       # 是否允许非托管工具
    repeated_tool_call_limit: int = 3                                         # 重复工具调用限制

    # ── 重试策略 ────────────────────────────────────────────────
    retry_enabled: bool = True                                                # 是否启用重试
    max_model_retries: int = 2                                                # 最大重试次数
    retry_base_delay_ms: int = 1200                                           # 重试基础延迟（毫秒）

    # ── 任务控制 ────────────────────────────────────────────────
    task_control_enabled: bool = True                                         # 是否启用任务控制
    task_mode: TaskMode = "edit"                                              # 用户任务模式
    planning_budget_profile: PlanningBudgetProfile = "balanced"               # plan discovery 预算档位
    max_task_replans_per_run: int = 2                                         # 最大任务重规划次数

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
        self.task_mode = ensure_task_mode(self.task_mode)
        self.planning_budget_profile = ensure_planning_budget_profile(
            self.planning_budget_profile
        )
        self.max_task_replans_per_run = _ensure_positive_int(
            self.max_task_replans_per_run,
            field_name="max_task_replans_per_run",
        )


@dataclass
class AgentState:
    """Agent 运行时状态：跟踪 Agent 在整个生命周期内的可变状态。

    与 AgentLoopConfig（每次运行时构建的不可变配置）不同，
    AgentState 在 Agent 实例的整个生命周期内持续存在，
    记录当前的消息历史、流式状态、待处理工具调用等信息。

    典型使用：
        - 外部通过 Agent.state 属性只读访问
        - Agent 内部在运行过程中更新状态
        - 事件分发时根据事件类型同步更新状态

    Attributes:
        system_prompt: 系统提示词，可通过 Agent.set_system_prompt() 更新。
        model: 当前使用的 LLM 模型。
        thinking_level: 推理思考级别，"off" 表示关闭推理。
        tools: 当前可用的工具列表，可通过 Agent.set_tools() 更新。
        messages: 消息历史列表，运行结束后新消息会被追加到此列表。
        is_streaming: 是否正在流式运行（同一时刻只有一个流式任务）。
        stream_message: 当前正在流式生成的消息（用于实时显示）。
        pending_tool_calls: 待处理的工具调用 ID 集合（用于跟踪工具执行进度）。
        error: 当前错误信息（如有），成功运行后会被清除。
    """
    system_prompt: str                                                        # 系统提示词
    model: Model                                                              # LLM 模型
    thinking_level: Literal["off", "minimal", "low", "medium", "high", "xhigh"] = "off"  # 推理级别
    tools: list[AgentTool] = field(default_factory=list)                      # 可用工具列表
    messages: list[AgentMessage] = field(default_factory=list)                # 消息历史
    is_streaming: bool = False                                                # 是否正在流式运行
    stream_message: AgentMessage | None = None                                # 当前流式消息
    pending_tool_calls: set[str] = field(default_factory=set)                 # 待处理工具调用 ID
    error: str | None = None                                                  # 错误信息

    def __post_init__(self) -> None:
        """初始化后校验：确保所有字段类型正确。"""
        if not isinstance(self.model, Model):
            raise TypeError("AgentState model must be Model")
        # 清理系统提示词
        self.system_prompt = _clean_core_text(self.system_prompt)
        # 校验思考级别
        self.thinking_level = _ensure_agent_thinking_level(self.thinking_level)
        # 防御性拷贝工具和消息列表
        self.tools = _copy_tools(self.tools, field_name="tools")
        self.messages = _copy_messages(self.messages, field_name="messages")
        # 校验布尔类型
        self.is_streaming = _ensure_bool(self.is_streaming, field_name="is_streaming")
        # 校验流式消息类型
        if self.stream_message is not None and not isinstance(
            self.stream_message,
            (UserMessage, AssistantMessage, ToolResultMessage),
        ):
            raise TypeError("AgentState stream_message must be AgentMessage or None")
        # 拷贝待处理工具调用集合
        self.pending_tool_calls = _copy_text_set(
            self.pending_tool_calls,
            field_name="pending_tool_calls",
        )
        # 清理错误信息
        self.error = _optional_core_text(self.error)


# ── 文本清理辅助函数 ─────────────────────────────────────────────

def _clean_core_text(value: object) -> str:
    """将任意值转换为字符串，None 转为空字符串。"""
    return str(value) if value is not None else ""


def _optional_core_text(value: object) -> str | None:
    """将任意值转换为可选字符串：空字符串返回 None。"""
    text = _clean_core_text(value).strip()
    return text or None


# ── 防御性拷贝辅助函数 ──────────────────────────────────────────

def _copy_messages(value: object, *, field_name: str) -> list[AgentMessage]:
    """拷贝消息列表：校验每个元素的类型，返回新的列表副本。"""
    if not isinstance(value, list):
        raise TypeError(f"AgentContext {field_name} must be a list")
    messages: list[AgentMessage] = []
    for message in value:
        if not isinstance(message, (UserMessage, AssistantMessage, ToolResultMessage)):
            raise TypeError(f"AgentContext {field_name} entries must be AgentMessage")
        messages.append(message)
    return messages


def _copy_tools(value: object, *, field_name: str) -> list[AgentTool]:
    """拷贝工具列表：校验每个元素的类型，返回新的列表副本。"""
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
    """深拷贝可选字典：None 返回 None，否则返回深拷贝副本。"""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"AgentContext {field_name} must be a dict or None")
    return deepcopy(value)


# ── 类型校验辅助函数 ─────────────────────────────────────────────
# 这些函数用于 dataclass 的 __post_init__ 中，确保字段值的类型正确。
# 采用防御式编程：在赋值时就捕获类型错误，而不是等到使用时才发现。

def _ensure_tool_execution_mode(value: object) -> ToolExecutionMode:
    """校验工具执行模式：必须是 "sequential" 或 "parallel"。"""
    text = str(value).strip() if value is not None else ""
    if text not in _TOOL_EXECUTION_MODES:
        raise ValueError(f"Unknown tool_execution mode: {value}")
    return cast(ToolExecutionMode, text)


def _ensure_optional_thinking_level(value: object) -> ThinkingLevel | None:
    """校验可选的推理思考级别：None 或 ThinkingLevel 枚举值。"""
    if value is None:
        return None
    # bool 是 int 的子类，需要排除
    if isinstance(value, bool):
        raise TypeError("AgentLoopConfig reasoning must be a thinking level or None")
    text = str(value).strip()
    if text not in _THINKING_LEVELS:
        raise ValueError(f"Unknown reasoning level: {value}")
    return cast(ThinkingLevel, text)


def _ensure_agent_thinking_level(value: object) -> Literal["off", "minimal", "low", "medium", "high", "xhigh"]:
    """校验 Agent 思考级别：包含 "off" 选项（表示关闭推理）。"""
    if isinstance(value, bool):
        raise TypeError("thinking_level must be a valid agent thinking level")
    text = str(value).strip() if value is not None else ""
    if text not in _AGENT_THINKING_LEVELS:
        raise ValueError(f"Unknown thinking_level: {value}")
    return cast(Literal["off", "minimal", "low", "medium", "high", "xhigh"], text)


def _copy_text_set(value: object, *, field_name: str) -> set[str]:
    """拷贝文本集合：校验类型并清理空白字符串。"""
    if not isinstance(value, set):
        raise TypeError(f"AgentState {field_name} must be a set")
    return {text for item in value if (text := str(item).strip())}


def _ensure_bool(value: object, *, field_name: str) -> bool:
    """校验布尔类型：Python 中 bool 是 int 的子类，需要显式检查。"""
    if not isinstance(value, bool):
        raise TypeError(f"AgentLoopConfig {field_name} must be bool")
    return value


def _ensure_optional_callable(value: object, field_name: str) -> None:
    """校验可选的可调用对象：None 或可调用对象。"""
    if value is not None and not callable(value):
        raise TypeError(f"{field_name} must be callable or None")


def _ensure_non_negative_int(value: object, *, field_name: str) -> int:
    """校验非负整数：排除 bool（bool 是 int 的子类），值 >= 0。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"AgentLoopConfig {field_name} must be int")
    if value < 0:
        raise ValueError(f"AgentLoopConfig {field_name} must be non-negative")
    return value


def _ensure_positive_int(value: object, *, field_name: str) -> int:
    """校验正整数：值 > 0。"""
    integer = _ensure_non_negative_int(value, field_name=field_name)
    if integer <= 0:
        raise ValueError(f"AgentLoopConfig {field_name} must be positive")
    return integer


def _ensure_optional_positive_int(value: object, *, field_name: str) -> int | None:
    """校验可选的正整数：None 或正整数。"""
    if value is None:
        return None
    return _ensure_positive_int(value, field_name=field_name)
