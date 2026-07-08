from __future__ import annotations

"""
可执行工具契约 —— 由工具层 (tools layer) 拥有和维护。

本模块定义了 Codepilot 系统中所有与工具相关的核心数据模型、类型别名、
协议接口和辅助函数。这些契约贯穿工具的注册、调度、执行、审批和安全策略
等完整生命周期，是工具子系统与运行时 (runtime)、会话 (sessions) 等其他
子系统之间通信的基础。

模块主要内容：
- 类型别名：ToolScope（工具作用域）、ToolObservationStatus（工具观察状态）、
  ToolInvocationSource（工具调用来源）
- 执行协议：ToolExecuteFn（工具执行函数签名）
- 核心数据类：ToolMetadata（运行时元数据）、ToolDefinition（工具定义）、
  ToolCallRequest（工具调用请求）、ToolInvocation（工具调用输入）、
  ToolObservation（工具执行观察输出）、ToolInterruption（审批中断）、
  ToolResumeDecision（审批恢复决策）、ToolPolicyContext（策略评估上下文）、
  ToolCatalogItem / ToolCatalogView（工具目录对外视图）、
  PreparedToolCall / PreparedToolCallResult（工具准备结果）、
  ToolRiskView（风险展示）
- 协议接口：ToolPort（工具端口 —— 工具子系统对外暴露的主接口）
- 辅助函数：tool_call_from_invocation（从调用输入构造 ToolCall）、
  error_result（构造错误结果）、_clean_text / _require_text / _optional_text / _clean_scopes（内部校验与清洗）
"""

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Literal, Protocol, TypeAlias, cast

from codepilot.protocols import (
    AssistantMessage,
    ContentBlock,
    RunVerification,
    TextContent,
    Tool,
    ToolCall,
    ToolHookContextSnapshot,
)
from codepilot.protocols.tools import ToolResult, ToolResultStatus, ToolRiskLevel

# ============================================================================
# 类型别名 (Type Aliases)
# ============================================================================

# ToolScope: 工具的作用域/可见范围。
# - "read":      只读操作（如读取文件、浏览目录）
# - "plan":      规划模式下的工具
# - "build":     构建模式下的工具（默认模式，允许修改文件）
# - "memory":    记忆相关操作
# - "extension": 扩展工具（在任何模式下都可见）
ToolScope: TypeAlias = Literal["read", "plan", "build", "memory", "extension"]

# ToolObservationStatus: 工具执行后的观察状态。
# - "success":           执行成功
# - "error":             执行出错
# - "denied":            被安全策略拒绝
# - "approval_required": 需要人工审批
# - "cancelled":         被取消
ToolObservationStatus: TypeAlias = Literal[
    "success",
    "error",
    "denied",
    "approval_required",
    "cancelled",
]

# ToolInvocationSource: 工具调用的来源。
# - "agent":           由 AI 智能体发起
# - "approval_resume": 由审批恢复流程发起（用户批准后重新执行）
ToolInvocationSource: TypeAlias = Literal["agent", "approval_resume"]

# ToolResumeDecisionValue: 审批恢复时的决策值。
# - "approve": 批准执行
# - "deny":    拒绝执行
ToolResumeDecisionValue: TypeAlias = Literal["approve", "deny"]

# 合法状态值集合（用于运行时校验）
_OBSERVATION_STATUSES = frozenset(
    {"success", "error", "denied", "approval_required", "cancelled"}
)
_RESUME_DECISIONS = frozenset({"approve", "deny"})
_RISK_LEVELS = frozenset({"low", "medium", "high"})
_SCOPES = frozenset({"read", "plan", "build", "memory", "extension"})

# ToolUpdateCallback: 工具执行过程中的更新回调类型。
# 接收一个 ToolResult 参数，用于在执行过程中向调用方推送中间状态。
ToolUpdateCallback: TypeAlias = Callable[[ToolResult], None]


# ============================================================================
# 执行函数协议 (Execution Function Protocol)
# ============================================================================

class ToolExecuteFn(Protocol):
    """
    工具执行函数的协议签名。

    任何实现了此签名的可调用对象都可以作为工具的 execute 函数。
    该函数接收一个具体的 ToolCallRequest，可选的信号对象 (signal)
    和可选的更新回调 (on_update)，返回 ToolResult 或 Awaitable[ToolResult]
    （支持同步和异步两种返回方式）。
    """
    def __call__(
        self,
        request: "ToolCallRequest",
        signal: Any | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> Awaitable[ToolResult] | ToolResult:
        ...


# ============================================================================
# 核心数据类 (Core Dataclasses)
# ============================================================================

@dataclass(frozen=True)
class ToolMetadata:
    """
    工具运行时元数据 —— 用于工具对外暴露、调度决策和安全策略评估。

    每个工具在注册时都附带一份元数据，描述其类别、风险等级、并发特性、
    是否需要审批等静态属性。这些信息在工具生命周期中保持不变。
    """

    # 工具唯一名称
    name: str
    # 工具类别标签（如 "files", "shell", "network" 等）
    category: str
    # 是否为只读操作（只读工具通常无需审批，安全风险低）
    read_only: bool
    # 是否并发安全（可同时运行多个实例而不会互相干扰）
    concurrency_safe: bool
    # 是否排他执行（执行时需要独占锁，其他工具必须等待）
    exclusive: bool
    # 是否需要人工审批
    requires_approval: bool
    # 风险等级: "low"（低）、"medium"（中）、"high"（高）
    risk_level: ToolRiskLevel
    # 可见作用域列表：控制工具在哪些模式下可见
    scopes: tuple[str, ...]
    # 是否需要网络访问权限
    network_access: bool = False
    # 是否需要凭据
    credential_required: bool = False
    # 额外元数据（用于扩展，存放任意键值对）
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        创建后的校验逻辑：
        1. 校验 name、category 为非空文本
        2. 校验所有布尔字段的类型
        3. 校验 risk_level 在合法值集合中
        4. 校验 scopes 列表合法且去重
        5. 校验 extra 是 dict 类型
        """
        object.__setattr__(self, "name", _require_text(self.name, "metadata.name"))
        object.__setattr__(
            self,
            "category",
            _require_text(self.category, "metadata.category"),
        )
        for field_name in (
            "read_only",
            "concurrency_safe",
            "exclusive",
            "requires_approval",
            "network_access",
            "credential_required",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"ToolMetadata {field_name} must be bool")
        risk_level = _clean_text(self.risk_level)
        if risk_level not in _RISK_LEVELS:
            raise ValueError(f"Unknown tool risk level: {self.risk_level}")
        object.__setattr__(self, "risk_level", cast(ToolRiskLevel, risk_level))
        object.__setattr__(self, "scopes", tuple(_clean_scopes(self.scopes)))
        if not isinstance(self.extra, dict):
            raise TypeError("ToolMetadata extra must be a dict")
        object.__setattr__(self, "extra", deepcopy(self.extra))

    def visible_in(self, current_mode: str) -> bool:
        """
        判断工具在指定的当前模式下是否可见。

        工具在以下任一条件下可见：
        1. current_mode 在工具的 scopes 列表中
        2. 工具的 scopes 中包含 "extension"（扩展工具在所有模式下均可见）

        返回 True 表示可见，False 表示不可见。
        """
        mode = _clean_text(current_mode)
        return mode in self.scopes or "extension" in self.scopes


@dataclass
class ToolDefinition:
    """
    工具定义 —— 一个可调用的工具，包含模型可见的 schema 和运行时元数据。

    ToolDefinition 是工具注册表中的核心条目。它将工具的"描述面"
    （name、description、parameters——供 LLM 理解和调用）与"执行面"
    （execute 函数、metadata 元数据）绑定在一起。

    注意：这是可变数据类 (frozen=False)，因为 execute 属性在注册后
    可能被动态替换或包装。
    """

    # 工具唯一名称（与 metadata.name 必须一致）
    name: str
    # 工具显示标签（面向用户的可读名称）
    label: str
    # 工具功能描述（面向 LLM，告诉模型何时以及如何使用该工具）
    description: str
    # JSON Schema 格式的参数定义（定义工具接受的输入参数结构）
    parameters: dict[str, Any]
    # 工具运行时元数据
    metadata: ToolMetadata
    # 工具执行函数（符合 ToolExecuteFn 协议的可调用对象）
    execute: ToolExecuteFn

    def __post_init__(self) -> None:
        """
        创建后的校验逻辑：
        1. 校验 name、label、description 为非空文本
        2. 校验 parameters 为 dict
        3. 校验 metadata 为 ToolMetadata 实例
        4. 校验 metadata.name == self.name（名称一致性）
        5. 校验 execute 可调用
        6. 深拷贝 parameters 以防止外部修改
        """
        self.name = _require_text(self.name, "tool.name")
        self.label = _require_text(self.label, "tool.label")
        self.description = _require_text(self.description, "tool.description")
        if not isinstance(self.parameters, dict):
            raise TypeError("ToolDefinition parameters must be a dict")
        if not isinstance(self.metadata, ToolMetadata):
            raise TypeError("ToolDefinition metadata must be ToolMetadata")
        if self.metadata.name != self.name:
            raise ValueError(
                f"Tool metadata name must match tool name: {self.metadata.name} != {self.name}"
            )
        if not callable(self.execute):
            raise TypeError("ToolDefinition execute must be callable")
        self.parameters = deepcopy(self.parameters)

    def to_spec(self) -> Tool:
        """
        将工具定义转换为纯协议对象 Tool（供 LLM API 使用）。

        Tool 对象包含 name、description、parameters 三个字段，
        是对外发送给 LLM 的工具描述，不包含执行逻辑和元数据。
        """
        return Tool(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


@dataclass(frozen=True)
class ToolCatalogItem:
    """
    工具目录条目 —— 工具目录中的一个条目。

    将工具的 spec（供 LLM 消费）和 metadata（供运行时代理消费）
    打包在一起，作为目录查询的基本单元。

    注意：ToolCatalogItem 不暴露 execute 函数，外部使用者只能看到
    工具的描述信息，不能直接执行。
    """
    spec: Tool
    metadata: ToolMetadata


@dataclass(frozen=True)
class ToolCatalogView:
    """
    工具目录视图 —— 工具目录的不可变快照。

    封装了当前可用的全部工具条目列表，提供便捷的迭代和查询方法。
    此视图是只读的（frozen=True），确保在传递过程中不会被意外修改。

    主要用途：
    1. 向 LLM 提供当前可用工具列表（通过 tools 属性）
    2. 供运行时筛选特定模式下的工具
    """
    items: tuple[ToolCatalogItem, ...] = field(default_factory=tuple)

    @property
    def tools(self) -> tuple[Tool, ...]:
        """提取所有条目的 Tool spec，返回纯协议对象元组。"""
        return tuple(item.spec for item in self.items)

    def __iter__(self) -> Iterator[Tool]:
        """迭代目录视图时默认迭代 Tool spec 列表。"""
        return iter(self.tools)

    def __len__(self) -> int:
        """返回目录中工具的数量。"""
        return len(self.items)


@dataclass(frozen=True)
class ToolPolicyContext:
    """
    工具策略评估上下文 —— 在执行策略检查时提供运行环境信息。

    策略模块在执行工具前需要根据上下文信息（如会话 ID、会话元数据）
    来决定是否允许执行、是否需要审批等。

    字段说明：
    - session_id: 当前会话的唯一标识（可选）
    - metadata: 与策略评估相关的额外键值对（如用户角色、权限级别等），
      内部存储为 MappingProxyType 以确保不可变
    """
    session_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """清洗 session_id（None 或空字符串统一为 None），将 metadata 转为不可变映射。"""
        object.__setattr__(self, "session_id", _optional_text(self.session_id))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(deepcopy(dict(self.metadata))),
        )


@dataclass(frozen=True)
class ToolCallRequest:
    """
    工具调用请求 —— 描述一次完整的工具调用所需的所有信息。

    当 AI 智能体决定调用某个工具时，会生成一个 ToolCallRequest，包含：
    - 运行标识（run_id、tool_call_id）：用于追踪和关联
    - 工具名称和参数（name、arguments）：告诉系统调用哪个工具以及传什么参数
    - 元数据和安全上下文（metadata、policy_context、current_mode、source）：
      用于策略评估和权限检查
    - 消息上下文（assistant_message、context）：提供调用的原始消息上下文

    这是整个工具执行链路的入口数据结构。
    """
    # 当前运行 ID（一次 run 可能包含多个 tool call）
    run_id: str
    # 工具调用 ID（全局唯一的调用标识）
    tool_call_id: str
    # 要调用的工具名称
    name: str
    # 工具调用参数（JSON 可序列化的键值对）
    arguments: dict[str, Any] = field(default_factory=dict)
    # 工具元数据（可选，通常由准备阶段填充）
    metadata: ToolMetadata | None = None
    # 当前运行模式（决定哪些工具可用）
    current_mode: str = "build"
    # 调用来源
    source: ToolInvocationSource = "agent"
    # 策略评估上下文
    policy_context: ToolPolicyContext = field(default_factory=ToolPolicyContext)
    # 触发此工具调用的 AI 助手消息（可选）
    assistant_message: AssistantMessage | None = None
    # 工具钩子上下文快照（可选，用于工具执行前后的拦截处理）
    context: ToolHookContextSnapshot | None = None

    def __post_init__(self) -> None:
        """清洗并校验所有字段：确保 run_id、tool_call_id、name 非空，arguments 为 dict，
        current_mode 非空，source 值合法。"""
        object.__setattr__(self, "run_id", _require_text(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "tool_call_id",
            _require_text(self.tool_call_id, "tool_call_id"),
        )
        object.__setattr__(self, "name", _require_text(self.name, "tool.name"))
        if not isinstance(self.arguments, dict):
            raise TypeError("ToolCallRequest arguments must be a dict")
        object.__setattr__(self, "arguments", deepcopy(self.arguments))
        object.__setattr__(self, "current_mode", _require_text(self.current_mode, "current_mode"))
        source = _clean_text(self.source)
        if source not in {"agent", "approval_resume"}:
            raise ValueError(f"Unknown tool invocation source: {self.source}")
        object.__setattr__(self, "source", cast(ToolInvocationSource, source))


@dataclass(frozen=True)
class PreparedToolCall:
    """
    已准备的工具调用 —— 将 ToolCallRequest 与对应的 ToolDefinition 绑定。

    准备阶段 (preparation phase) 的输出：根据请求中的工具名称查找到
    对应的工具定义后，将两者打包为 PreparedToolCall，供后续执行使用。
    """
    # 匹配到的工具定义
    definition: ToolDefinition
    # 原始调用请求
    request: ToolCallRequest

    @property
    def metadata(self) -> ToolMetadata:
        """快捷属性：直接从 definition 获取工具元数据。"""
        return self.definition.metadata


@dataclass(frozen=True)
class PreparedToolCallResult:
    """
    工具准备结果 —— 准备阶段的返回结构。

    如果准备成功：call 字段包含有效的 PreparedToolCall，error_code 为 None。
    如果准备失败（如工具未找到、模式不匹配等）：error_code 包含错误代码，
    message 包含错误描述，recovery_hint 包含恢复建议。
    """
    # 已准备好的工具调用（成功时非 None）
    call: PreparedToolCall | None = None
    # 错误代码（成功时为 None）
    error_code: str | None = None
    # 错误或成功消息
    message: str = ""
    # 恢复提示（告诉调用方如何修复问题）
    recovery_hint: str = ""

    @property
    def valid(self) -> bool:
        """
        判断准备结果是否有效。

        当 call 不为 None 且 error_code 为 None 时返回 True，
        表示准备成功，可以继续进行工具执行。
        """
        return self.call is not None and self.error_code is None


@dataclass(frozen=True)
class ToolRiskView:
    """
    工具风险视图 —— 以人类可读的方式展示工具的风险等级。

    用于在审批中断时将风险信息呈现给用户，帮助用户做出批准/拒绝的决策。

    字段说明：
    - level: 风险等级字符串（如 "low", "medium", "high", "unknown"）
    - summary: 风险摘要描述（说明为什么这个工具有该风险等级）
    """
    level: str
    summary: str = ""


@dataclass(frozen=True)
class ToolInterruption:
    """
    工具中断 —— 当工具需要人工审批时产生的中断信号。

    当工具的 requires_approval 为 True 或策略评估认为需要审批时，
    系统会生成一个 ToolInterruption 并暂停执行流程，等待用户做出
    批准或拒绝的决策。用户决策通过 ToolResumeDecision 返回。

    字段说明：
    - approval_id: 审批请求的唯一 ID（用于后续恢复时关联）
    - run_id: 当前运行 ID
    - tool_call_id: 被中断的工具调用 ID
    - tool_name: 被中断的工具名称
    - arguments: 工具调用的参数（供用户审查）
    - reason: 中断原因（说明为什么需要审批）
    - risk: 风险视图（展示风险信息给用户）
    """
    approval_id: str
    run_id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, object] = field(default_factory=dict)
    reason: str = ""
    risk: ToolRiskView = field(default_factory=lambda: ToolRiskView(level="unknown"))


@dataclass(frozen=True)
class ToolInvocation:
    """
    工具调用输入 —— 传递给 ToolPort.execute() 的标准化输入结构。

    与 ToolCallRequest 类似但更精简：去掉了 metadata 字段（因为此时
    工具定义已匹配完成），保留执行所需的核心信息。

    同时去除了 source 字段中的调用来源区分，统一作为执行入口参数。
    """
    run_id: str
    tool_call_id: str
    name: str
    arguments: dict[str, object] = field(default_factory=dict)
    current_mode: str = "build"
    source: ToolInvocationSource = "agent"
    policy_context: ToolPolicyContext = field(default_factory=ToolPolicyContext)
    assistant_message: AssistantMessage | None = None
    context: ToolHookContextSnapshot | None = None


@dataclass(frozen=True)
class ToolObservation:
    """
    工具执行观察 —— 工具执行完成后产生的输出/结果。

    ToolObservation 是工具执行链路的终点数据结构。它记录了：
    - 执行状态（成功、错误、被拒绝、需要审批、已取消）
    - 输出内容（content：多模态内容块元组）
    - 副作用信息（affected_paths：受影响的文件路径，workspace_changed：工作区是否改变）
    - 验证结果（verification：运行后验证项）
    - 中断信息（interruption：如果需要审批，包含中断详情）
    - 额外元数据（metadata：工具自定义的附加信息）

    注意：content 使用 tuple 是因为 dataclass 的 frozen=True 需要不可变类型。
    """
    # 工具调用 ID
    tool_call_id: str
    # 工具名称
    name: str
    # 执行状态
    status: ToolObservationStatus
    # 输出内容（多模态内容块的不可变元组）
    content: tuple[ContentBlock, ...] = field(default_factory=tuple)
    # 受影响的文件路径列表
    affected_paths: tuple[str, ...] = field(default_factory=tuple)
    # 工作区是否发生了变化
    workspace_changed: bool = False
    # 运行后验证结果列表
    verification: tuple[RunVerification, ...] = field(default_factory=tuple)
    # 中断信息（如果需要审批）
    interruption: ToolInterruption | None = None
    # 额外元数据
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """清洗 status 字段，确保其值在合法的观察状态集合中。"""
        status = _clean_text(self.status)
        if status not in _OBSERVATION_STATUSES:
            raise ValueError(f"Unknown tool observation status: {self.status}")
        object.__setattr__(self, "status", cast(ToolObservationStatus, status))


@dataclass(frozen=True)
class ToolResumeDecision:
    """
    工具恢复决策 —— 用户在审批中断后做出的决策。

    当工具执行被中断等待审批时，用户通过此数据结构
    传达批准 (approve) 或拒绝 (deny) 的决定，系统根据此决定
    恢复或终止工具执行。

    字段说明：
    - approval_id: 对应的审批请求 ID（与 ToolInterruption.approval_id 对应）
    - decision: "approve"（批准执行）或 "deny"（拒绝执行）
    - reason: 决策理由（可选，用于记录审计日志）
    """
    approval_id: str
    decision: ToolResumeDecisionValue
    reason: str = ""

    def __post_init__(self) -> None:
        """清洗并校验：确保 approval_id 非空，decision 在合法决策值集合中，
        reason 为清洗后的文本。"""
        decision = _clean_text(self.decision)
        if decision not in _RESUME_DECISIONS:
            raise ValueError(f"Unknown tool resume decision: {self.decision}")
        object.__setattr__(self, "approval_id", _require_text(self.approval_id, "approval_id"))
        object.__setattr__(self, "decision", cast(ToolResumeDecisionValue, decision))
        object.__setattr__(self, "reason", _clean_text(self.reason))


# ============================================================================
# 协议接口 (Protocol Interface)
# ============================================================================

class ToolPort(Protocol):
    """
    工具端口 —— 工具子系统对外暴露的主协议接口。

    ToolPort 定义了工具子系统的三个核心能力：
    1. catalog():  查询当前可用的工具目录
    2. execute():  执行一个工具调用
    3. resume():   在审批中断后恢复执行（批准或拒绝）

    任何实现了此协议的对象都可以作为工具子系统与外部（如运行时）
    进行交互。这种端口/适配器架构使得工具子系统可以独立替换和测试。
    """

    def catalog(self, current_mode: str = "build") -> ToolCatalogView:
        """
        获取当前模式下的工具目录视图。

        参数 current_mode 指定当前运行模式（如 "read"、"plan"、"build"），
        返回的 ToolCatalogView 仅包含在该模式下可见的工具。
        """
        ...

    async def execute(self, invocation: ToolInvocation) -> ToolObservation:
        """
        执行一个工具调用。

        接收 ToolInvocation 作为输入，执行后返回 ToolObservation 作为结果。
        这是一个异步方法，因为工具执行可能涉及 I/O 操作（如文件读写、网络请求等）。
        """
        ...

    async def resume(self, decision: ToolResumeDecision) -> ToolObservation:
        """
        在审批中断后恢复工具执行。

        接收用户的 ToolResumeDecision（批准或拒绝），
        恢复之前被中断的工具执行流程，返回最终的 ToolObservation。
        如果用户拒绝，返回状态为 "denied" 的观察结果。
        """
        ...


# ============================================================================
# 辅助函数 (Helper Functions)
# ============================================================================

def tool_call_from_invocation(invocation: ToolInvocation) -> ToolCall:
    """
    从 ToolInvocation 构造标准的 ToolCall 对象。

    ToolCall 是协议层定义的简单数据结构（包含 id、name、arguments），
    供 LLM API 和历史记录使用。此函数桥接了工具层的 ToolInvocation
    和协议层的 ToolCall 之间的数据转换。

    参数:
        invocation: 工具调用输入

    返回:
        ToolCall: 协议层的工具调用对象
    """
    return ToolCall(
        id=invocation.tool_call_id,
        name=invocation.name,
        arguments=dict(invocation.arguments),
    )


def error_result(message: str, *, status: ToolResultStatus = "error", error_code: str) -> ToolResult:
    """
    构造一个错误工具结果 (ToolResult)。

    当工具执行失败时，使用此函数快速创建统一的错误响应。
    自动设置 is_error=True，并将错误消息包装为 TextContent。

    参数:
        message:   错误描述消息
        status:    结果状态（默认为 "error"，也可用于 "denied" 等其他非成功状态）
        error_code: 机器可读的错误代码（如 "TOOL_NOT_FOUND"、"PERMISSION_DENIED"）

    返回:
        ToolResult: 包含错误信息的工具结果对象
    """
    return ToolResult(
        content=[TextContent(text=message)],
        status=status,
        is_error=status != "success",
        error_code=error_code,
    )


# ============================================================================
# 内部工具函数 (Internal Helper Functions)
# ============================================================================

def _clean_scopes(values: tuple[str, ...]) -> list[str]:
    """
    清洗并校验作用域列表。

    对输入的每个值进行文本清洗，校验其是否在合法作用域集合中，
    去除重复值（保持第一次出现的顺序），确保结果非空。

    参数:
        values: 原始作用域元组

    返回:
        list[str]: 去重后的合法作用域列表

    异常:
        ValueError: 如果某个值不在合法集合中，或结果为空
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        if text not in _SCOPES:
            raise ValueError(f"Unknown tool scope: {value}")
        if text not in seen:
            cleaned.append(text)
            seen.add(text)
    if not cleaned:
        raise ValueError("ToolMetadata scopes cannot be empty")
    return cleaned


def _require_text(value: object, field_name: str) -> str:
    """
    要求文本值非空 —— 清洗后如果为空白字符串则抛出异常。

    用于校验必填的字符串字段（如 name、id 等），确保它们有实际内容。

    参数:
        value:      待校验的值
        field_name: 字段名（用于异常消息）

    返回:
        str: 清洗后的非空字符串

    异常:
        ValueError: 如果 value 为空或清洗后为空
    """
    text = _clean_text(value)
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text


def _optional_text(value: object) -> str | None:
    """
    可选文本值清洗 —— 清洗后如果为空白字符串则返回 None。

    用于校验可选的字符串字段（如 session_id），空值统一为 None。

    参数:
        value: 待清洗的值

    返回:
        str | None: 清洗后的非空字符串，或 None
    """
    text = _clean_text(value)
    return text or None


def _clean_text(value: object) -> str:
    """
    基础文本清洗 —— 将任意值转为去除首尾空白的字符串。

    如果 value 为 None，返回空字符串。
    如果 value 为其他类型，调用 str() 转换后去除首尾空白。

    参数:
        value: 任意值

    返回:
        str: 清洗后的字符串
    """
    return str(value).strip() if value is not None else ""


# ============================================================================
# 模块导出列表
# ============================================================================

__all__ = [
    "PreparedToolCall",
    "PreparedToolCallResult",
    "ToolCallRequest",
    "ToolCatalogItem",
    "ToolCatalogView",
    "ToolDefinition",
    "ToolExecuteFn",
    "ToolInterruption",
    "ToolInvocation",
    "ToolInvocationSource",
    "ToolMetadata",
    "ToolObservation",
    "ToolObservationStatus",
    "ToolPolicyContext",
    "ToolPort",
    "ToolResumeDecision",
    "ToolResumeDecisionValue",
    "ToolResult",
    "ToolRiskView",
    "ToolScope",
    "ToolUpdateCallback",
    "error_result",
    "tool_call_from_invocation",
]
