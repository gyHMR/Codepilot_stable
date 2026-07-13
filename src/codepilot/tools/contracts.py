from __future__ import annotations

"""工具拥有者、Registry、Runtime 和 Core 共享的规范定义。

本文件是整个工具子系统的"契约层"，定义了所有核心类型和协议接口。
理解 tools 包的关键就是理解这里的类型体系。
"""

import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    Awaitable,
    Callable,
    Generic,
    Literal,
    Protocol,
    TYPE_CHECKING,
    TypeAlias,
    TypeVar,
    cast,
)

from .security import (
    ApprovalChallenge,
    ApprovalResponse,
    ToolAccessResolution,
    ToolMode,
    ToolPolicy,
)

if TYPE_CHECKING:
    from .registry import ToolCatalogSnapshot
    from .results import ToolResult
    from .state import InteractionResponse


# ── 类型别名 ──────────────────────────────────────────────────────────────────
#
# ToolCategory: 工具的功能分类
#   - filesystem:  文件系统操作（read/write/edit/ls）
#   - search:      搜索（grep/glob）
#   - command:     受控命令（受限制的 git/python 等）
#   - delegation:  代理/子代理
#   - plan:        计划相关
#   - interaction: 用户交互（弹框询问）
#   - external:    外部工具（MCP 等）
#
# ToolSource: 工具的来源（决定其信任级别和替换规则）
#   - builtin:   内置工具（不可被外部覆盖）
#   - caller:    调用方注册
#   - skill:     技能注册
#   - extension: Python 扩展插件注册
#   - mcp:       MCP 服务器代理

ToolCategory: TypeAlias = Literal[
    "filesystem",
    "search",
    "command",
    "delegation",
    "plan",
    "interaction",
    "external",
]
ToolSource: TypeAlias = Literal["builtin", "caller", "skill", "extension", "mcp"]

# 工具名正则：字母开头，后续字母/数字/下划线/连字符，最长 64 字符
_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_TOOL_CATEGORIES = frozenset(
    {"filesystem", "search", "command", "delegation", "plan", "interaction", "external"}
)
_TOOL_SOURCES = frozenset({"builtin", "caller", "skill", "extension", "mcp"})
_TOOL_MODES = frozenset({"plan", "execute", "unrestricted"})


@dataclass(frozen=True)
class ToolSpec:
    """工具规格 —— 对 LLM 可见的工具定义。

    这是工具定义中会暴露给模型的部分，包含名称、描述和输入/输出 JSON Schema。
    LLM 通过这个 spec 了解有哪些工具可用以及如何调用。

    参数:
        name: 工具名称（必须匹配 [A-Za-z][A-Za-z0-9_-]{0,63}）
        description: 工具描述（LLM 理解此工具用途的文本）
        input_schema: 输入参数的 JSON Schema（Draft 2020-12）
        output_schema: 输出结果的 JSON Schema（可选）
        schema_version: schema 版本号（默认为 1，必须为正整数）
    """

    name: str
    description: str
    input_schema: Mapping[str, object]
    output_schema: Mapping[str, object] | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        # 验证名称格式
        name = _require_text(self.name, "tool name")
        if _TOOL_NAME_PATTERN.fullmatch(name) is None:
            raise ValueError(f"Invalid tool name: {self.name}")
        # 验证版本号
        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version, int
        ):
            raise TypeError("schema_version must be int")
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self, "description", _require_text(self.description, "description")
        )
        # 冻结 input_schema 防止运行时篡改
        object.__setattr__(
            self,
            "input_schema",
            _freeze_json_mapping(self.input_schema, "input_schema"),
        )
        if self.output_schema is not None:
            object.__setattr__(
                self,
                "output_schema",
                _freeze_json_mapping(self.output_schema, "output_schema"),
            )


@dataclass(frozen=True)
class ToolExecutionRequest:
    """工具执行请求 —— 从 model step 创建的具体执行请求。

    包含模型发出的工具调用的完整上下文：
    - run_id / session_id: 标识所属的运行和会话
    - tool_call_id: LLM 为该工具调用分配的 ID
    - tool_name: 调用的工具名称
    - arguments: LLM 传入的参数
    - mode: 当前运行模式（plan / execute / unrestricted）
    - registration_id: 模型看到的注册快照 ID（用于检测过期）
    - deadline_at_ms: 可选的执行截止时间戳
    - raw_arguments / argument_parse_error: 参数解析失败时的原始内容

    返回值:
        此对象被传递给 ToolRuntime，由它负责查找注册、验证权限并执行。
    """

    run_id: str
    session_id: str
    tool_call_id: str
    tool_name: str
    arguments: Mapping[str, object]
    mode: ToolMode
    registration_id: str
    deadline_at_ms: int | None = None
    raw_arguments: str | None = None
    argument_parse_error: str | None = None

    def __post_init__(self) -> None:
        # 所有 ID 字段不能为空
        for name in (
            "run_id",
            "session_id",
            "tool_call_id",
            "tool_name",
            "registration_id",
        ):
            object.__setattr__(self, name, _require_text(getattr(self, name), name))
        mode = _clean_text(self.mode)
        if mode not in _TOOL_MODES:
            raise ValueError(f"Unknown tool mode: {self.mode}")
        if self.deadline_at_ms is not None and (
            isinstance(self.deadline_at_ms, bool)
            or not isinstance(self.deadline_at_ms, int)
        ):
            raise TypeError("deadline_at_ms must be int or None")
        object.__setattr__(self, "mode", cast(ToolMode, mode))
        object.__setattr__(
            self, "arguments", _freeze_json_mapping(self.arguments, "arguments")
        )
        object.__setattr__(self, "raw_arguments", _optional_text(self.raw_arguments))
        object.__setattr__(
            self,
            "argument_parse_error",
            _optional_text(self.argument_parse_error),
        )


@dataclass(frozen=True)
class ToolBatchPreparation:
    """Opaque prepared batch handle or side-effect-free preparation results."""

    batch_id: str | None = None
    results: tuple["ToolResult", ...] = ()

    def __post_init__(self) -> None:
        from .results import ToolResult

        batch_id = _optional_text(self.batch_id)
        results = tuple(self.results)
        if any(not isinstance(result, ToolResult) for result in results):
            raise TypeError(
                "ToolBatchPreparation results must contain ToolResult values"
            )
        if (batch_id is None) == (not results):
            raise ValueError(
                "ToolBatchPreparation requires exactly one of batch_id or results"
            )
        object.__setattr__(self, "batch_id", batch_id)
        object.__setattr__(self, "results", results)


class CancellationToken(Protocol):
    """取消令牌 —— 工具处理器检查取消状态的协议接口。

    工具处理器在长时间操作中应定期检查 cancelled 属性
    或调用 raise_if_cancelled() 以响应取消信号。
    """

    @property
    def cancelled(self) -> bool: ...

    def raise_if_cancelled(self) -> None: ...


class ProgressReporter(Protocol):
    """进度报告器 —— 工具处理器发送进度事件的协议接口。

    工具处理器可以在执行过程中调用 report() 发送中间状态事件，
    这些事件会被转发到上层（REPL 显示进度、RPC 发送事件帧等）。
    """

    async def report(
        self,
        kind: str,
        *,
        message: str = "",
        data: Mapping[str, object] | None = None,
    ) -> None: ...


class EffectReporter(Protocol):
    """效果报告器 —— 工具处理器报告副作用的协议接口。

    工具处理器通过此接口记录每次操作的文件系统/网络等副作用，
    这些记录会被用于权限验证（实际效果不超过授权范围）和审计日志。
    """

    def report(self, effect: object) -> None: ...


CleanupCallback: TypeAlias = Callable[[], Awaitable[None] | None]


class CleanupStack(Protocol):
    """清理栈 —— 注册和运行工具清理回调的协议接口。

    工具处理器可以通过 push() 注册清理回调（如关闭临时文件句柄），
    工具执行完成后这些回调会被逆序执行。
    """

    def push(self, callback: CleanupCallback) -> None: ...


@dataclass(frozen=True)
class ToolExecutionContext:
    """工具执行上下文 —— 运行时提供给工具处理器的完整上下文。

    当工具处理器被调用时，此对象包含了执行所需的所有基础设施：
    - request: 原始的 ToolExecutionRequest（含调用 ID、参数等）
    - cancellation: 取消令牌（检查是否被取消）
    - deadline_at_ms: 执行截止时间戳
    - progress: 进度报告器（发送中间状态事件）
    - effects: 效果报告器（记录副作用操作）
    - cleanup: 清理栈（注册清理回调）

    工具处理器不直接创建此对象，由 ToolRuntime 在执行时构造并传入。
    """

    request: ToolExecutionRequest
    cancellation: CancellationToken
    deadline_at_ms: int | None
    progress: ProgressReporter
    effects: EffectReporter
    cleanup: CleanupStack


TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class ToolHandlerError(Exception):
    """工具处理器错误 —— 工具处理器抛出的预期领域异常。

    与普通的 Exception 不同，ToolHandlerError 被视为"预期的"失败，
    不会触发崩溃恢复逻辑。它包含结构化的错误码和可选的详细信息，
    帮助上层理解失败原因并决定是否重试。

    参数:
        code: 错误码（如 "write.exists"、"read.not_file"）
        message: 人类可读的错误描述
        retryable: 此错误是否可重试（如超时可重试，权限拒绝不可重试）
        details: 额外错误数据的映射
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: Mapping[str, object] | None = None,
    ) -> None:
        self.code = _require_text(code, "tool handler error code")
        self.message = _require_text(message, "tool handler error message")
        self.retryable = bool(retryable)
        if details is not None and not isinstance(details, Mapping):
            raise TypeError("tool handler error details must be a mapping")
        self.details = deepcopy(dict(details or {}))
        super().__init__(self.message)


class ToolCodec(Protocol, Generic[TInput]):
    """编解码器协议 —— 工具值的序列化和反序列化接口。

    每个工具注册时都需要提供 input_codec 和 output_codec，
    负责在"原始 JSON 对象"和"类型化的 Python 对象"之间转换。
    具体的实现有 DataclassCodec（基于 dataclass 类型转换）和
    JsonObjectCodec（纯 JSON Schema 验证）。

    类型参数 TInput: 解码后的 Python 类型（如 WriteInput dataclass）。
    """

    @property
    def json_schema(self) -> Mapping[str, object] | None: ...

    # 返回 JSON Schema，None 表示不校验（仅 UnverifiedJsonCodec 使用）

    def decode(self, value: object) -> TInput: ...

    # 将 JSON 对象解码为类型化的 Python 值
    # 参数 value: 来自 LLM 的原始 JSON 参数
    # 返回: 解码后的类型化对象

    def encode(self, value: TInput) -> object: ...

    # 将类型化的 Python 值编码回 JSON 对象
    # 参数 value: 工具处理器的返回值
    # 返回: 编码后的 JSON 对象


class ToolHandler(Protocol, Generic[TInput, TOutput]):
    """工具处理器协议 —— 实际执行工具逻辑的接口。

    工具的所有业务逻辑都在 handler 中实现。
    handler 是一个异步可调用对象，接收解码后的输入和上下文，
    返回类型化的输出，然后由 output_codec 编码为 JSON。

    类型参数:
        TInput: 解码后的输入类型
        TOutput: 原始的返回类型（在编码前）
    """

    async def __call__(
        self, input: TInput, context: ToolExecutionContext
    ) -> TOutput: ...


class ToolAccessResolver(Protocol, Generic[TInput]):
    """访问解析器协议 —— 将工具输入解析为访问权限请求的接口。

    在工具执行前被调用，负责：
    1. 对输入参数进行路径安全解析（如将相对路径转为工作区绝对路径）
    2. 构造 ToolAccessRequest（包含受影响资源、预期效果、风险级别）
    3. 返回 ToolAccessResolution 供审批流程使用

    不同类型的工具有不同的访问解析逻辑：
    - 文件工具：解析路径、检查是否在工作区内、分类读写
    - Shell 工具：解析命令、评估命令分类（只读/变更/高风险）
    """

    def resolve(
        self,
        input: TInput,
        request: ToolExecutionRequest,
    ) -> ToolAccessResolution[TInput]: ...


class ToolOutputRenderer(Protocol):
    """输出渲染器协议 —— 将工具输出数据转为 LLM 可消费的内容块。

    render() 接收编码后的输出字典，返回内容元素元组（TextContent、
    ImageContent、ArtifactContent），这些内容将作为 ToolResult 的
    content 字段返回给 LLM。
    """

    def render(self, data: Mapping[str, object]) -> tuple[object, ...]: ...


@dataclass(frozen=True)
class ToolRegistration:
    """工具注册 —— 工具拥有者提供的完整定义，在 Registry 物化之前使用。

    这是工具创建者需要填充的"登记表"，包含工具的所有方面：
    - 身份信息（名称、版本、分类、来源、拥有者）
    - 模型可见定义（spec）
    - 安全策略（policy）
    - 输入输出编解码（codecs）
    - 业务逻辑（handler）
    - 权限解析（access_resolver）
    - 结果渲染（renderer）

    __post_init__ 会对所有字段做严格校验，确保注册信息的完整性。
    """

    #: 工具版本号（字符串，如 "1.0.0"）
    version: str
    #: 实现版本号（用于检测 handler 实现是否变化）
    implementation_version: str
    #: 工具规格（对 LLM 可见的名称、描述、Schema）
    spec: ToolSpec
    #: 工具功能分类
    category: ToolCategory
    #: 工具来源
    source: ToolSource
    #: 工具的拥有者标识（用于权限管理和卸载）
    owner: str
    #: 安全策略（并发、超时、权限模式、审批规则等）
    policy: ToolPolicy
    #: 输入编解码器（JSON → 类型化对象）
    input_codec: ToolCodec[object]
    #: 输出编解码器（类型化对象 → JSON）
    output_codec: ToolCodec[object]
    #: 工具处理器（实际业务逻辑）
    handler: ToolHandler[object, object]
    #: 输出渲染器（输出字典 → LLM 内容块）
    renderer: ToolOutputRenderer
    #: 访问解析器（输入参数 → 权限审批请求）
    access_resolver: ToolAccessResolver[object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _require_text(self.version, "tool version"))
        object.__setattr__(
            self,
            "implementation_version",
            _require_text(self.implementation_version, "implementation_version"),
        )
        object.__setattr__(self, "owner", _require_text(self.owner, "tool owner"))
        if not isinstance(self.spec, ToolSpec):
            raise TypeError("spec must be ToolSpec")
        category = _clean_text(self.category)
        if category not in _TOOL_CATEGORIES:
            raise ValueError(f"Unknown tool category: {self.category}")
        source = _clean_text(self.source)
        if source not in _TOOL_SOURCES:
            raise ValueError(f"Unknown tool source: {self.source}")
        if not isinstance(self.policy, ToolPolicy):
            raise TypeError("policy must be ToolPolicy")
        if not callable(self.handler):
            raise TypeError("handler must be callable")
        if not callable(getattr(self.input_codec, "decode", None)):
            raise TypeError("input_codec must implement decode")
        if not callable(getattr(self.output_codec, "encode", None)):
            raise TypeError("output_codec must implement encode")
        if not callable(getattr(self.renderer, "render", None)):
            raise TypeError("renderer must implement render")
        if not callable(getattr(self.access_resolver, "resolve", None)):
            raise TypeError("access_resolver must implement resolve")
        object.__setattr__(self, "category", cast(ToolCategory, category))
        object.__setattr__(self, "source", cast(ToolSource, source))


class ToolPort(Protocol):
    """工具端口 —— ToolRuntime 对外暴露的标准服务接口。

    ToolPort 是 runtime 层与 core 层之间的边界接口协议。
    Core 通过此端口与工具子系统交互，不直接依赖 ToolRuntime 实现。

    提供的方法包括：
    - catalog_snapshot(): 获取当前可用工具的快照（给 LLM 生成调用）
    - execute(): 执行单个工具调用
    - execute_batch(): 批量执行工具调用（自动并行化）
    - cancel(): 取消正在执行的工具
    - pending_challenges(): 获取所有待审批的挑战
    - approval_challenge(): 获取指定审批挑战详情
    - resume(): 恢复被挂起的工具执行（审批通过或用户输入后）
    """

    def catalog_snapshot(
        self, *, mode: ToolMode | None = None
    ) -> ToolCatalogSnapshot: ...

    def pending_challenges(self) -> tuple[ApprovalChallenge, ...]: ...

    async def execute(self, request: ToolExecutionRequest) -> ToolResult: ...

    async def execute_batch(
        self,
        requests: tuple[ToolExecutionRequest, ...] | list[ToolExecutionRequest],
    ) -> list[ToolResult]: ...

    def prepare_batch(
        self,
        requests: tuple[ToolExecutionRequest, ...] | list[ToolExecutionRequest],
    ) -> ToolBatchPreparation: ...

    async def execute_prepared(self, batch_id: str) -> tuple[ToolResult, ...]: ...

    async def cancel(self, attempt_id: str) -> bool: ...

    def approval_challenge(self, approval_id: str): ...

    async def resume(
        self, response: ApprovalResponse | InteractionResponse
    ) -> ToolResult: ...

    def checkpoint_state(
        self,
        *,
        intent: Mapping[str, object] | None = None,
    ) -> dict[str, object] | None: ...

    def restore_checkpoint_state(self, state: Mapping[str, object]) -> None: ...


# ── 内部辅助函数 ──────────────────────────────────────────────────────────────


def _freeze_json_mapping(
    value: Mapping[str, object], field_name: str
) -> Mapping[str, object]:
    """递归冻结一个 JSON 兼容的映射为不可变视图（MappingProxyType）。

    使用 MappingProxyType 包装确保数据在运行时不可变，
    防止代码意外修改本应只读的数据结构。"""
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return cast(Mapping[str, object], _freeze_json_value(dict(value)))


def _freeze_json_value(value: object) -> object:
    """递归冻结 JSON 值，将 dict → MappingProxyType，list/tuple → tuple。"""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    return deepcopy(value)


def _clean_text(value: object) -> str:
    """清理文本：None → ""，其他转 str 并去除首尾空格。"""
    return str(value).strip() if value is not None else ""


def _require_text(value: object, field_name: str) -> str:
    """要求值必须有文本内容，否则抛出 ValueError。"""
    text = _clean_text(value)
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text


def _optional_text(value: object) -> str | None:
    """将值转为可选的清理后文本。"""
    return _clean_text(value) or None


__all__ = [
    "CancellationToken",
    "CleanupStack",
    "EffectReporter",
    "ProgressReporter",
    "ToolAccessResolver",
    "ToolBatchPreparation",
    "ToolCategory",
    "ToolCodec",
    "ToolExecutionContext",
    "ToolExecutionRequest",
    "ToolHandler",
    "ToolHandlerError",
    "ToolOutputRenderer",
    "ToolPort",
    "ToolRegistration",
    "ToolSource",
    "ToolSpec",
]
