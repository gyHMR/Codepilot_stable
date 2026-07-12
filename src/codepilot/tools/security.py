"""规范的工具策略、资源、副作用和权限引擎值对象。

本文件是工具子系统的"安全层"，定义了所有安全相关的数据模型：
1. 类型别名体系 —— ToolMode（运行模式）、RiskLevel（风险等级）等
2. 策略类 —— TimeoutPolicy、ConcurrencyPolicy、OutputLimits、ToolPolicy
3. 资源类 —— ToolResource（受影响资源）、ToolEffect（副作用记录）
4. 访问请求类 —— ToolAccessRequest、ToolAccessResolution
5. 权限引擎 —— PermissionRule、PermissionEngine（决策核心）
6. 审批链 —— ApprovalChallenge、ApprovalResponse、ApprovalGrant
7. 工具函数 —— 审批指纹、挑战构建、授权发放
"""

import fnmatch
import hashlib
import json
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Generic, Literal, Mapping, TypeAlias, TypeVar, cast


# ── 类型别名 ──────────────────────────────────────────────────────────────────

# ToolMode: 工具的可用模式
#   - plan:         计划模式（只允许只读工具）
#   - execute:      执行模式（允许读写工具）
#   - unrestricted: 无限制模式（跳过部分安全检查，仅用于特定场景）
ToolMode: TypeAlias = Literal["plan", "execute", "unrestricted"]

# RiskLevel: 风险等级（用于决定是否需要审批）
#   - low:     低风险（如 read、ls）
#   - medium:  中等风险（如 write、edit）
#   - high:    高风险（如 bash、网络访问）
#   - critical: 严重风险（如删除、格式化）
RiskLevel: TypeAlias = Literal["low", "medium", "high", "critical"]

# ApprovalPolicy: 审批策略（工具注册时声明）
#   - never:    从不审批（仅用于绝对安全的只读操作）
#   - on_risk:  按风险等级决定（当 risk >= medium 时审批）
#   - always:   总是需要审批
ApprovalPolicy: TypeAlias = Literal["never", "on_risk", "always"]

# ToolEffectKind: 工具副作用类型
#   - filesystem_read:      读取文件系统
#   - filesystem_write:     写入文件系统
#   - filesystem_delete:    删除文件系统内容
#   - process_spawn:        创建子进程
#   - network_access:       网络访问
#   - credential_access:    凭据访问
#   - external_state_read:  读取外部状态
#   - external_state_write: 写入外部状态
#   - session_state_read:   读取会话状态
#   - session_state_write:  写入会话状态
ToolEffectKind: TypeAlias = Literal[
    "filesystem_read",
    "filesystem_write",
    "filesystem_delete",
    "process_spawn",
    "network_access",
    "credential_access",
    "external_state_read",
    "external_state_write",
    "session_state_read",
    "session_state_write",
]

# EffectStatus: 副作用的最终状态
#   - started:   已开始（可能尚未完成）
#   - completed: 已完成
#   - partial:   部分完成
#   - unknown:   未知
EffectStatus: TypeAlias = Literal["started", "completed", "partial", "unknown"]

# EffectCertainty: 副作用的确信度
#   - observed: 实际观察到（最可靠）
#   - reported: 工具报告了（可靠但非直接观察）
#   - inferred: 推断的（根据工具类型推测）
EffectCertainty: TypeAlias = Literal["observed", "reported", "inferred"]

# ContentTrust: 内容可信度（用于防止提示注入）
#   - trusted:   来自内置工具的可信内容
#   - untrusted: 来自外部源的内容
ContentTrust: TypeAlias = Literal["trusted", "untrusted"]

# PermissionEffect: 权限决策效果
#   - allow: 允许
#   - deny:  拒绝
#   - ask:   需要询问（审批或交互）
PermissionEffect: TypeAlias = Literal["allow", "deny", "ask"]

# RuleSource: 权限规则来源
#   - configuration: 配置文件
#   - project:       项目设置
#   - runtime:       运行时动态生成
RuleSource: TypeAlias = Literal["configuration", "project", "runtime"]

# ApprovalScope: 审批范围（一次审批的有效期）
#   - once:    仅当次调用
#   - session: 整个会话期间有效
#   - project: 整个项目期间有效
ApprovalScope: TypeAlias = Literal["once", "session", "project"]

# ApprovalDecisionValue: 审批决策值
#   - approve: 批准
#   - deny:    拒绝
ApprovalDecisionValue: TypeAlias = Literal["approve", "deny"]

# 合法值集合（运行时校验用）
_TOOL_MODES = frozenset({"plan", "execute", "unrestricted"})
_RISK_LEVELS = frozenset({"low", "medium", "high", "critical"})
_APPROVAL_POLICIES = frozenset({"never", "on_risk", "always"})
_EFFECT_KINDS = frozenset(
    {
        "filesystem_read",
        "filesystem_write",
        "filesystem_delete",
        "process_spawn",
        "network_access",
        "credential_access",
        "external_state_read",
        "external_state_write",
        "session_state_read",
        "session_state_write",
    }
)
_EFFECT_STATUSES = frozenset({"started", "completed", "partial", "unknown"})
_EFFECT_CERTAINTIES = frozenset({"observed", "reported", "inferred"})
_CONTENT_TRUST = frozenset({"trusted", "untrusted"})
_PERMISSION_EFFECTS = frozenset({"allow", "deny", "ask"})
_RULE_SOURCES = frozenset({"configuration", "project", "runtime"})
_APPROVAL_SCOPES = frozenset({"once", "session", "project"})
_APPROVAL_DECISIONS = frozenset({"approve", "deny"})
_RISK_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}

# 模型授权禁用键 —— LLM 不能在工具参数中通过这些键试图绕过安全机制
_FORBIDDEN_MODEL_AUTH_KEYS = frozenset(
    {
        "allow_dangerous",
        "bypass_approval",
        "ignore_workspace_boundary",
        "trusted",
        "sudo",
        "force_without_approval",
    }
)


# ── 策略类 ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TimeoutPolicy:
    """超时策略 —— 定义工具执行的超时限制。

    参数:
        default_execution_ms: 默认执行超时（毫秒），未指定时使用的值
        max_execution_ms: 最大允许执行超时（毫秒），用户指定的超时不能超过此值
        cleanup_grace_ms: 清理操作的宽限时间（毫秒），默认 5000ms
    """

    default_execution_ms: int
    max_execution_ms: int
    cleanup_grace_ms: int = 5_000

    def __post_init__(self) -> None:
        _require_positive_int(self.default_execution_ms, "default_execution_ms")
        _require_positive_int(self.max_execution_ms, "max_execution_ms")
        if self.max_execution_ms < self.default_execution_ms:
            raise ValueError("max_execution_ms cannot be less than default_execution_ms")
        _require_non_negative_int(self.cleanup_grace_ms, "cleanup_grace_ms")


@dataclass(frozen=True)
class ConcurrencyPolicy:
    """并发策略 —— 定义工具的并发执行方式。

    参数:
        mode: 并发模式
            - "parallel": 可并行执行
            - "serial": 串行执行（受 group 限制）
        group: 串行/限流分组的名称（serial 模式必须提供）
        max_parallel: 同一组内最大并行数（可选，提供时限制并发度）
    """

    mode: Literal["parallel", "serial"]
    group: str | None = None
    max_parallel: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"parallel", "serial"}:
            raise ValueError(f"Unknown concurrency mode: {self.mode}")
        group = _optional_text(self.group)
        if self.mode == "serial" and group is None:
            raise ValueError("serial concurrency requires a group")
        if self.max_parallel is not None:
            _require_positive_int(self.max_parallel, "max_parallel")
            if group is None:
                raise ValueError("max_parallel concurrency requires a group")
        object.__setattr__(self, "group", group)


@dataclass(frozen=True)
class OutputLimits:
    """输出限制 —— 定义工具输出的最大字节数等限制。

    参数:
        max_data_bytes: 结构化的 data 部分最大字节数（默认 256KB）
        max_content_bytes: content（Text/Image）部分最大字节数（默认 128KB）
        max_artifacts: 最多允许的工件数量（默认 16）
        max_artifact_bytes: 单工件最大字节数（默认 16MB）
    """

    max_data_bytes: int = 256_000
    max_content_bytes: int = 128_000
    max_artifacts: int = 16
    max_artifact_bytes: int = 16_000_000

    def __post_init__(self) -> None:
        for name in (
            "max_data_bytes",
            "max_content_bytes",
            "max_artifacts",
            "max_artifact_bytes",
        ):
            _require_positive_int(getattr(self, name), name)


@dataclass(frozen=True)
class OutputTrustPolicy:
    """输出信任策略 —— 定义工具输出内容的信任级别。

    参数:
        default_content_trust: 默认内容可信度（"trusted" 或 "untrusted"）
        allow_structurally_validated: 是否允许仅做结构验证的输出
            （允许 UnverifiedJsonCodec 的输出通过）
    """

    default_content_trust: ContentTrust = "trusted"
    allow_structurally_validated: bool = False

    def __post_init__(self) -> None:
        trust = _clean_text(self.default_content_trust)
        if trust not in _CONTENT_TRUST:
            raise ValueError(f"Unknown content trust: {self.default_content_trust}")
        if not isinstance(self.allow_structurally_validated, bool):
            raise TypeError("allow_structurally_validated must be bool")
        object.__setattr__(self, "default_content_trust", cast(ContentTrust, trust))


@dataclass(frozen=True)
class ToolPolicy:
    """工具策略 —— 工具注册时声明的完整安全策略。

    这是工具安全策略的汇总对象，包含所有执行约束：
    - allowed_modes: 工具在哪些模式下可用（如 write 只在 execute 模式可用）
    - declared_effects: 工具声明会产生的副作用（用于审批参考）
    - required_permissions: 执行此工具需要的权限标签
    - base_risk: 基础风险等级
    - approval: 审批策略
    - timeout: 超时策略
    - concurrency: 并发策略
    - output_limits: 输出限制
    - output_trust: 输出信任策略

    参数:
        allowed_modes: 允许的运行模式集合
        declared_effects: 声明会产生的副作用集合
        required_permissions: 需要的权限标签集合
        base_risk: 基础风险等级
        approval: 审批策略
        timeout: 超时策略
        concurrency: 并发策略
        output_limits: 输出限制
        output_trust: 输出信任策略
    """

    allowed_modes: frozenset[ToolMode]
    declared_effects: frozenset[ToolEffectKind]
    required_permissions: frozenset[str]
    base_risk: RiskLevel
    approval: ApprovalPolicy
    timeout: TimeoutPolicy
    concurrency: ConcurrencyPolicy
    output_limits: OutputLimits
    output_trust: OutputTrustPolicy

    def __post_init__(self) -> None:
        modes = frozenset(_clean_text(value) for value in self.allowed_modes)
        if not modes or not modes <= _TOOL_MODES:
            raise ValueError(f"Invalid allowed tool modes: {sorted(modes)}")
        effects = frozenset(_clean_text(value) for value in self.declared_effects)
        if not effects <= _EFFECT_KINDS:
            raise ValueError(f"Invalid declared effects: {sorted(effects)}")
        permissions = frozenset(_require_text(value, "required permission") for value in self.required_permissions)
        risk = _clean_text(self.base_risk)
        if risk not in _RISK_LEVELS:
            raise ValueError(f"Unknown tool risk level: {self.base_risk}")
        approval = _clean_text(self.approval)
        if approval not in _APPROVAL_POLICIES:
            raise ValueError(f"Unknown approval policy: {self.approval}")
        for name, expected in (
            ("timeout", TimeoutPolicy),
            ("concurrency", ConcurrencyPolicy),
            ("output_limits", OutputLimits),
            ("output_trust", OutputTrustPolicy),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        object.__setattr__(self, "allowed_modes", cast(frozenset[ToolMode], modes))
        object.__setattr__(self, "declared_effects", cast(frozenset[ToolEffectKind], effects))
        object.__setattr__(self, "required_permissions", permissions)
        object.__setattr__(self, "base_risk", cast(RiskLevel, risk))
        object.__setattr__(self, "approval", cast(ApprovalPolicy, approval))


# ── 资源与副作用 ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolResource:
    """工具资源 —— 工具操作影响的目标资源标识。

    URI 格式示例：
    - 工作区文件: "workspace:///src/main.py"
    - 外部 URL:   "https://api.example.com/data"
    - 进程:       "process://python"

    参数:
        uri: 资源的唯一标识 URI
        metadata: 资源的额外元数据（如文件大小、MIME 类型等）
    """

    uri: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "uri", _require_text(self.uri, "resource uri"))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "resource metadata"))


@dataclass(frozen=True)
class ToolEffect:
    """工具副作用 —— 工具执行过程中产生的单一副作用记录。

    每个工具操作（如读取文件、写入文件、创建进程）产生一个 ToolEffect，
    这些记录用于：
    - 权限验证：实际效果不得超过授权范围
    - 审计日志：记录所有文件系统变更
    - 变更检测：记录哪些文件被修改了

    参数:
        kind: 副作用类型
        resource: 受影响的资源
        operation: 操作描述（如 "write file"、"read file"）
        status: 执行状态
        certainty: 确信度
    """

    kind: ToolEffectKind
    resource: ToolResource
    operation: str
    status: EffectStatus
    certainty: EffectCertainty

    def __post_init__(self) -> None:
        kind = _clean_text(self.kind)
        if kind not in _EFFECT_KINDS:
            raise ValueError(f"Unknown tool effect: {self.kind}")
        if not isinstance(self.resource, ToolResource):
            raise TypeError("effect resource must be ToolResource")
        status = _clean_text(self.status)
        if status not in _EFFECT_STATUSES:
            raise ValueError(f"Unknown effect status: {self.status}")
        certainty = _clean_text(self.certainty)
        if certainty not in _EFFECT_CERTAINTIES:
            raise ValueError(f"Unknown effect certainty: {self.certainty}")
        object.__setattr__(self, "kind", cast(ToolEffectKind, kind))
        object.__setattr__(self, "operation", _require_text(self.operation, "effect operation"))
        object.__setattr__(self, "status", cast(EffectStatus, status))
        object.__setattr__(self, "certainty", cast(EffectCertainty, certainty))


# ── 访问请求与解析 ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolAccessRequest:
    """工具访问请求 —— 描述工具执行需要访问的资源和操作。

    由 access_resolver 在工具执行前构造并提交给权限引擎。
    包含所有需要审批的信息：
    - 要执行的操作（如 "write"、"read"）
    - 受影响的资源（如 "workspace:///src/main.py"）
    - 预期的副作用集合
    - 风险等级
    - 安全预览（供用户确认）

    参数:
        actions: 操作名称元组（如 ("write",)）
        resources: 受影响的资源列表
        effects: 预期产生的副作用集合
        risk: 综合评估后的风险等级
        reason: 操作原因（供审批弹窗显示）
        safe_preview: 安全预览数据（如文件路径、命令文本）
        approval_scopes: 可选的审批范围（默认 once/session/project 都可用）
    """

    actions: tuple[str, ...]
    resources: tuple[ToolResource, ...]
    effects: frozenset[ToolEffectKind]
    risk: RiskLevel
    reason: str
    safe_preview: Mapping[str, object] = field(default_factory=dict)
    approval_scopes: frozenset[ApprovalScope] = frozenset(
        {"once", "session", "project"}
    )

    def __post_init__(self) -> None:
        actions = tuple(_require_text(value, "access action") for value in self.actions)
        if not actions:
            raise ValueError("access actions cannot be empty")
        resources = tuple(self.resources)
        if any(not isinstance(value, ToolResource) for value in resources):
            raise TypeError("access resources must be ToolResource values")
        effects = frozenset(_clean_text(value) for value in self.effects)
        if not effects <= _EFFECT_KINDS:
            raise ValueError(f"Invalid access effects: {sorted(effects)}")
        risk = _clean_text(self.risk)
        if risk not in _RISK_LEVELS:
            raise ValueError(f"Unknown access risk: {self.risk}")
        scopes = frozenset(_clean_text(value) for value in self.approval_scopes)
        if not scopes or not scopes <= _APPROVAL_SCOPES:
            raise ValueError(f"Invalid access approval scopes: {sorted(scopes)}")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "resources", resources)
        object.__setattr__(self, "effects", cast(frozenset[ToolEffectKind], effects))
        object.__setattr__(self, "risk", cast(RiskLevel, risk))
        object.__setattr__(self, "reason", _require_text(self.reason, "access reason"))
        object.__setattr__(self, "safe_preview", _freeze_mapping(self.safe_preview, "safe preview"))
        object.__setattr__(
            self,
            "approval_scopes",
            cast(frozenset[ApprovalScope], scopes),
        )


TInput = TypeVar("TInput")


@dataclass(frozen=True)
class ToolAccessResolution(Generic[TInput]):
    """工具访问解析结果 —— access_resolver 的返回类型。

    泛型 TInput 表示解码后的输入类型。

    参数:
        input: 解码后的工具输入
        access: 访问请求（供权限引擎决策）
        execution_timeout_ms: 执行超时覆盖值（可选）
    """

    input: TInput
    access: ToolAccessRequest
    execution_timeout_ms: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.access, ToolAccessRequest):
            raise TypeError("access must be ToolAccessRequest")
        if self.execution_timeout_ms is not None:
            _require_positive_int(self.execution_timeout_ms, "execution_timeout_ms")
        object.__setattr__(self, "input", deepcopy(self.input))


# ── 权限引擎 ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PermissionRule:
    """权限规则 —— 权限引擎中的一条规则条目。

    每条规则定义一个模式匹配条件：
    - 当工具的操作和资源匹配 action_pattern 和 resource_pattern 时
    - 且在指定的 modes 下
    - 应用 effect（allow/deny/ask）决策

    规则优先级：priority 越高越优先；模式特异性越强越优先。

    参数:
        action_pattern: 操作名称的 glob 模式（如 "write"、"read"、"*"）
        resource_pattern: 资源 URI 的 glob 模式（如 "workspace:///**"）
        effect: 规则的决策效果
        modes: 适用的工具模式（空集合表示所有模式）
        source: 规则来源
        priority: 优先级（数字越大越优先，默认 0）
    """

    action_pattern: str
    resource_pattern: str
    effect: PermissionEffect
    modes: frozenset[ToolMode] = frozenset()
    source: RuleSource = "configuration"
    priority: int = 0

    def __post_init__(self) -> None:
        effect = _clean_text(self.effect)
        source = _clean_text(self.source)
        modes = frozenset(_clean_text(value) for value in self.modes)
        if effect not in _PERMISSION_EFFECTS:
            raise ValueError(f"Unknown permission effect: {self.effect}")
        if source not in _RULE_SOURCES:
            raise ValueError(f"Unknown permission rule source: {self.source}")
        if not modes <= _TOOL_MODES:
            raise ValueError(f"Invalid permission rule modes: {sorted(modes)}")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise TypeError("permission rule priority must be int")
        object.__setattr__(self, "action_pattern", _require_text(self.action_pattern, "action pattern"))
        object.__setattr__(
            self,
            "resource_pattern",
            _require_text(self.resource_pattern, "resource pattern"),
        )
        object.__setattr__(self, "effect", cast(PermissionEffect, effect))
        object.__setattr__(self, "modes", cast(frozenset[ToolMode], modes))
        object.__setattr__(self, "source", cast(RuleSource, source))


@dataclass(frozen=True)
class PermissionDecision:
    """权限决策 —— PermissionEngine 的决策结果。

    参数:
        effect: 决策效果（allow/deny/ask）
        reason: 决策原因（如 "tool_policy_allow"、"required_permission_missing"）
        matched_rule: 命中的规则（如果有）
        details: 额外决策细节（如缺少的权限列表）
    """

    effect: PermissionEffect
    reason: str
    matched_rule: PermissionRule | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        effect = _clean_text(self.effect)
        if effect not in _PERMISSION_EFFECTS:
            raise ValueError(f"Unknown permission decision: {self.effect}")
        object.__setattr__(self, "effect", cast(PermissionEffect, effect))
        object.__setattr__(self, "reason", _require_text(self.reason, "permission reason"))
        object.__setattr__(self, "details", _freeze_mapping(self.details, "permission details"))

    @property
    def allowed(self) -> bool:
        """决策是否允许执行。"""
        return self.effect == "allow"

    @property
    def denied(self) -> bool:
        """决策是否拒绝了执行。"""
        return self.effect == "deny"

    @property
    def requires_approval(self) -> bool:
        """决策是否需要审批。"""
        return self.effect == "ask"


@dataclass(frozen=True)
class PermissionEngine:
    """权限引擎 —— 工具执行的权限决策核心。

    决策流程：
    1. 检查模型是否在参数中试图绕过安全机制（拒绝！）
    2. 检查当前运行模式是否在工具允许模式中（拒绝！）
    3. 检查是否有所需的权限标签（拒绝！）
    4. 检查声明副作用是否超出工具策略限制（拒绝！）
    5. 检查是否命中运行时拒绝效果（拒绝！）
    6. 检查是否需要运行时审批效果（要求审批）
    7. 检查规则匹配：最高优先级的规则决定策略
    8. 检查审批策略：
       - "always" → 要求审批
       - "on_risk" 且 risk >= medium → 要求审批
    9. 默认允许

    规则匹配优先级：
    1. priority 越高越优先
    2. 模式特异性（不含通配符的字符数）越高越优先
    3. 同优先级下 deny > ask > allow

    参数:
        rules: 权限规则的元组
        denied_effects: 全局禁止的副作用集合
        approval_effects: 全局需要审批的副作用集合
        granted_permissions: 已授予的权限标签集合（默认包含通配符 "*" 表示全部授予）
    """

    rules: tuple[PermissionRule, ...] = ()
    denied_effects: frozenset[ToolEffectKind] = frozenset()
    approval_effects: frozenset[ToolEffectKind] = frozenset()
    granted_permissions: frozenset[str] = frozenset({"*"})

    def __post_init__(self) -> None:
        rules = tuple(self.rules)
        if any(not isinstance(rule, PermissionRule) for rule in rules):
            raise TypeError("permission rules must be PermissionRule values")
        denied_effects = frozenset(self.denied_effects)
        approval_effects = frozenset(self.approval_effects)
        granted_permissions = frozenset(
            _require_text(value, "granted permission")
            for value in self.granted_permissions
        )
        if not denied_effects <= _EFFECT_KINDS:
            raise ValueError(f"Invalid denied effects: {sorted(denied_effects)}")
        if not approval_effects <= _EFFECT_KINDS:
            raise ValueError(f"Invalid approval effects: {sorted(approval_effects)}")
        object.__setattr__(self, "rules", rules)
        object.__setattr__(self, "denied_effects", denied_effects)
        object.__setattr__(self, "approval_effects", approval_effects)
        object.__setattr__(self, "granted_permissions", granted_permissions)

    def decide(self, request, policy: ToolPolicy, access: ToolAccessRequest) -> PermissionDecision:
        """执行权限决策 —— 对一次工具调用请求做出审批判断。

        这是权限引擎的核心方法，执行完整的决策流程。

        参数:
            request: ToolExecutionRequest（执行请求）
            policy: ToolPolicy（工具注册时声明的策略）
            access: ToolAccessRequest（访问解析器生成的请求）

        返回:
            PermissionDecision（allow/deny/ask 三种结果）
        """
        # 第 1 步：检测模型授权绕过
        if _contains_forbidden_auth(request.arguments):
            return PermissionDecision("deny", "model_authorization_forbidden")
        # 第 2 步：检查运行模式
        if request.mode not in policy.allowed_modes:
            return PermissionDecision("deny", "tool_mode_denied")
        # 第 3 步：检查权限标签
        missing_permissions = sorted(
            permission
            for permission in policy.required_permissions
            if not any(
                fnmatch.fnmatchcase(permission, pattern)
                for pattern in self.granted_permissions
            )
        )
        if missing_permissions:
            return PermissionDecision(
                "deny",
                "required_permission_missing",
                details={"missing_permissions": missing_permissions},
            )
        # 第 4 步：检查声明副作用
        if not access.effects <= policy.declared_effects:
            return PermissionDecision("deny", "tool_effect_policy_violation")
        # 第 4.5 步：Shell 高风险命令拒绝
        if access.safe_preview.get("shell_class") == "high_risk":
            return PermissionDecision("deny", "shell_high_risk_forbidden")
        # 第 5 步：运行时拒绝效果
        if access.effects & self.denied_effects:
            return PermissionDecision("deny", "runtime_effect_denied")
        # 第 6 步：运行时审批效果
        if access.effects & self.approval_effects:
            return PermissionDecision("ask", "runtime_effect_requires_approval")

        # 第 7 步：规则匹配
        matched = self._matched_rule(request.mode, access)
        if matched is not None:
            return PermissionDecision(matched.effect, f"permission_rule_{matched.effect}", matched)
        # 第 8 步：审批策略
        if policy.approval == "always":
            return PermissionDecision("ask", "tool_policy_requires_approval")
        risk = max(_RISK_RANK[policy.base_risk], _RISK_RANK[access.risk])
        if policy.approval == "on_risk" and risk >= _RISK_RANK["medium"]:
            return PermissionDecision("ask", "tool_risk_requires_approval")
        # 第 9 步：默认允许
        return PermissionDecision("allow", "tool_policy_allow")

    def _matched_rule(self, mode: ToolMode, access: ToolAccessRequest) -> PermissionRule | None:
        """查找匹配的最优规则。

        对每个操作和资源组合，找到所有匹配的规则，
        然后按优先级和特异性排序，返回最优规则。

        如果有多种效果（deny/ask/allow）的规则匹配，
        优先应用 deny，然后 ask，最后 allow。
        """
        resource_uris = tuple(resource.uri for resource in access.resources) or ("",)
        effect_rank = {"allow": 0, "ask": 1, "deny": 2}
        def rank(rule: PermissionRule) -> tuple[int, int, int]:
            return (
                rule.priority,
                _pattern_specificity(rule.action_pattern) + _pattern_specificity(rule.resource_pattern),
                effect_rank[rule.effect],
            )
        selected = []
        for action in access.actions:
            for resource_uri in resource_uris:
                matches = [
                    rule
                    for rule in self.rules
                    if (not rule.modes or mode in rule.modes)
                    and fnmatch.fnmatchcase(action, rule.action_pattern)
                    and fnmatch.fnmatchcase(resource_uri, rule.resource_pattern)
                ]
                if not matches:
                    return None
                selected.append(max(matches, key=rank))
        for effect in ("deny", "ask", "allow"):
            candidates = [rule for rule in selected if rule.effect == effect]
            if candidates:
                return max(candidates, key=rank)
        return None


# ── 审批链 ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ApprovalChallenge:
    """审批挑战 —— 需要用户审批的工具执行请求。

    当权限引擎决定 "ask" 时，构建此对象发送到上层。
    上层（CLI/RPC）会展示这些信息给用户，等待用户批准或拒绝。

    参数:
        approval_id: 审批的唯一 ID
        request_fingerprint: 请求的指纹（用于验证响应的合法性）
        run_id: 所属运行的 ID
        session_id: 所属会话的 ID
        tool_call_id: 工具调用的 ID
        tool_name: 工具名称
        registration_id: 注册 ID
        actions: 请求的操作
        resources: 受影响的资源
        effects: 预期副作用
        risk: 风险等级
        reason: 审批原因（给用户看的说明）
        safe_preview: 安全预览数据
        allowed_scopes: 允许的审批范围
        expires_at_ms: 过期时间戳（毫秒）
    """

    approval_id: str
    request_fingerprint: str
    run_id: str
    session_id: str
    tool_call_id: str
    tool_name: str
    registration_id: str
    actions: tuple[str, ...]
    resources: tuple[ToolResource, ...]
    effects: frozenset[ToolEffectKind]
    risk: RiskLevel
    reason: str
    safe_preview: Mapping[str, object]
    allowed_scopes: frozenset[ApprovalScope] = frozenset({"once"})
    expires_at_ms: int | None = None

    def __post_init__(self) -> None:
        scopes = frozenset(_clean_text(value) for value in self.allowed_scopes)
        if not scopes or not scopes <= _APPROVAL_SCOPES:
            raise ValueError(f"Invalid approval scopes: {sorted(scopes)}")
        for name in (
            "approval_id",
            "request_fingerprint",
            "run_id",
            "session_id",
            "tool_call_id",
            "tool_name",
            "registration_id",
            "reason",
        ):
            object.__setattr__(self, name, _require_text(getattr(self, name), name))
        object.__setattr__(self, "actions", tuple(_require_text(value, "approval action") for value in self.actions))
        object.__setattr__(self, "resources", tuple(self.resources))
        object.__setattr__(self, "effects", frozenset(self.effects))
        object.__setattr__(self, "allowed_scopes", cast(frozenset[ApprovalScope], scopes))
        object.__setattr__(self, "safe_preview", _freeze_mapping(self.safe_preview, "safe preview"))
        if self.expires_at_ms is not None:
            _require_non_negative_int(self.expires_at_ms, "expires_at_ms")


@dataclass(frozen=True)
class ApprovalResponse:
    """审批响应 —— 用户对审批挑战的回复。

    参数:
        approval_id: 对应审批挑战的 ID
        request_fingerprint: 请求指纹（验证指纹是否匹配）
        decision: 决策（approve/deny）
        scope: 审批范围（once/session/project）
        reason: 审批理由（可选）
    """

    approval_id: str
    request_fingerprint: str
    decision: ApprovalDecisionValue
    scope: ApprovalScope = "once"
    reason: str = ""

    def __post_init__(self) -> None:
        decision = _clean_text(self.decision)
        scope = _clean_text(self.scope)
        if decision not in _APPROVAL_DECISIONS:
            raise ValueError(f"Unknown approval decision: {self.decision}")
        if scope not in _APPROVAL_SCOPES:
            raise ValueError(f"Unknown approval scope: {self.scope}")
        object.__setattr__(self, "approval_id", _require_text(self.approval_id, "approval_id"))
        object.__setattr__(
            self,
            "request_fingerprint",
            _require_text(self.request_fingerprint, "request_fingerprint"),
        )
        object.__setattr__(self, "decision", cast(ApprovalDecisionValue, decision))
        object.__setattr__(self, "scope", cast(ApprovalScope, scope))
        object.__setattr__(self, "reason", _clean_text(self.reason))


@dataclass(frozen=True)
class ApprovalGrant:
    """审批授权 —— 用户批准后发放的授权凭证。

    用于后续的同范围自动批准（session/project scope 的缓存）。
    当用户选择 "session" 或 "project" 范围时，
    后续相同操作可以直接使用此 grant 而无需再次审批。

    参数:
        grant_id: 授权的唯一 ID
        approval_id: 对应的审批挑战 ID
        request_fingerprint: 请求指纹
        scope: 授权的适用范围
        actions: 授权允许的操作
        resources: 授权允许的资源
        effects: 授权允许的副作用
        risk: 授权允许的风险等级
        issued_at_ms: 发放时间戳
        expires_at_ms: 过期时间戳
    """

    grant_id: str
    approval_id: str
    request_fingerprint: str
    scope: ApprovalScope
    actions: tuple[str, ...]
    resources: tuple[ToolResource, ...]
    effects: frozenset[ToolEffectKind]
    risk: RiskLevel
    issued_at_ms: int
    expires_at_ms: int | None

    def __post_init__(self) -> None:
        scope = _clean_text(self.scope)
        if scope not in _APPROVAL_SCOPES:
            raise ValueError(f"Unknown approval scope: {self.scope}")
        object.__setattr__(self, "grant_id", _require_text(self.grant_id, "grant_id"))
        object.__setattr__(self, "approval_id", _require_text(self.approval_id, "approval_id"))
        object.__setattr__(
            self,
            "request_fingerprint",
            _require_text(self.request_fingerprint, "request_fingerprint"),
        )
        object.__setattr__(self, "scope", cast(ApprovalScope, scope))
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "resources", tuple(self.resources))
        effects = frozenset(_clean_text(value) for value in self.effects)
        if not effects <= _EFFECT_KINDS:
            raise ValueError(f"Invalid grant effects: {sorted(effects)}")
        risk = _clean_text(self.risk)
        if risk not in _RISK_LEVELS:
            raise ValueError(f"Unknown grant risk: {self.risk}")
        object.__setattr__(self, "effects", cast(frozenset[ToolEffectKind], effects))
        object.__setattr__(self, "risk", cast(RiskLevel, risk))
        _require_non_negative_int(self.issued_at_ms, "issued_at_ms")
        if self.expires_at_ms is not None:
            _require_non_negative_int(self.expires_at_ms, "expires_at_ms")

    def expired(self, now_ms: int | None = None) -> bool:
        """检查授权是否已过期。

        参数:
            now_ms: 当前时间戳（毫秒），默认使用系统时间

        返回:
            True 表示已过期
        """
        now = int(time.time() * 1000) if now_ms is None else now_ms
        return self.expires_at_ms is not None and now >= self.expires_at_ms


# ── 审批工具函数 ──────────────────────────────────────────────────────────────


def approval_fingerprint(request, access: ToolAccessRequest) -> str:
    """计算审批请求的指纹。

    将请求和访问的关键信息序列化为 JSON 后计算 SHA256 哈希。
    指纹用于验证审批响应对应的是否是同一个请求。

    参数:
        request: ToolExecutionRequest
        access: ToolAccessRequest

    返回:
        "sha256:" + 64 位十六进制哈希
    """
    payload = {
        "run_id": request.run_id,
        "session_id": request.session_id,
        "tool_call_id": request.tool_call_id,
        "tool_name": request.tool_name,
        "registration_id": request.registration_id,
        "arguments": _json_value(request.arguments),
        "actions": list(access.actions),
        "resources": [resource.uri for resource in access.resources],
        "effects": sorted(access.effects),
        "approval_scopes": sorted(access.approval_scopes),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_approval_challenge(
    request,
    access: ToolAccessRequest,
    *,
    reason: str,
    ttl_ms: int = 300_000,
) -> ApprovalChallenge:
    """构建审批挑战 —— 当权限引擎决定 "ask" 时调用。

    参数:
        request: ToolExecutionRequest
        access: ToolAccessRequest
        reason: 审批原因
        ttl_ms: 挑战的有效期（毫秒），默认 5 分钟

    返回:
        ApprovalChallenge 对象
    """
    now_ms = int(time.time() * 1000)
    return ApprovalChallenge(
        approval_id="approval_" + uuid.uuid4().hex[:20],
        request_fingerprint=approval_fingerprint(request, access),
        run_id=request.run_id,
        session_id=request.session_id,
        tool_call_id=request.tool_call_id,
        tool_name=request.tool_name,
        registration_id=request.registration_id,
        actions=access.actions,
        resources=access.resources,
        effects=access.effects,
        risk=access.risk,
        reason=reason,
        safe_preview=access.safe_preview,
        allowed_scopes=access.approval_scopes,
        expires_at_ms=now_ms + ttl_ms,
    )


def issue_approval_grant(
    challenge: ApprovalChallenge,
    response: ApprovalResponse,
) -> ApprovalGrant:
    """发放审批授权 —— 用户批准后从挑战和响应生成授权。

    参数:
        challenge: 原始审批挑战
        response: 用户的批准响应

    返回:
        ApprovalGrant 对象
    """
    now_ms = int(time.time() * 1000)
    return ApprovalGrant(
        grant_id="grant_" + uuid.uuid4().hex[:20],
        approval_id=challenge.approval_id,
        request_fingerprint=challenge.request_fingerprint,
        scope=response.scope,
        actions=challenge.actions,
        resources=challenge.resources,
        effects=challenge.effects,
        risk=challenge.risk,
        issued_at_ms=now_ms,
        expires_at_ms=challenge.expires_at_ms,
    )


def approval_challenge_data(challenge: ApprovalChallenge) -> dict[str, object]:
    """将审批挑战转为可序列化的字典（供 CLI/RPC 渲染）。

    参数:
        challenge: ApprovalChallenge 对象

    返回:
        纯 JSON 兼容的字典
    """
    return {
        "approval_id": challenge.approval_id,
        "request_fingerprint": challenge.request_fingerprint,
        "run_id": challenge.run_id,
        "session_id": challenge.session_id,
        "tool_call_id": challenge.tool_call_id,
        "tool_name": challenge.tool_name,
        "registration_id": challenge.registration_id,
        "actions": list(challenge.actions),
        "resources": [resource.uri for resource in challenge.resources],
        "effects": sorted(challenge.effects),
        "risk": challenge.risk,
        "reason": challenge.reason,
        "safe_preview": dict(challenge.safe_preview),
        "allowed_scopes": sorted(challenge.allowed_scopes),
        "expires_at_ms": challenge.expires_at_ms,
    }


# ── 内部辅助函数 ──────────────────────────────────────────────────────────────


def _contains_forbidden_auth(value: object) -> bool:
    """递归检查值中是否包含禁止的模型授权键。

    防止 LLM 在工具参数中试图绕过安全措施。
    检查所有嵌套的 dict 和 list。
    """
    if isinstance(value, Mapping):
        return bool(_FORBIDDEN_MODEL_AUTH_KEYS.intersection(str(key) for key in value)) or any(
            _contains_forbidden_auth(item) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_auth(item) for item in value)
    return False


def _pattern_specificity(pattern: str) -> int:
    """计算模式的特异性（不含通配符的字符数）。

    用于规则排序：更具体的模式优先于更通用的模式。
    """
    return sum(1 for char in pattern if char not in "*?[]")


def _json_value(value: object) -> object:
    """将任意值转为纯 JSON 兼容表示。"""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _freeze_mapping(value: Mapping[str, object], field_name: str) -> Mapping[str, object]:
    """递归冻结映射为不可变视图。"""
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return cast(Mapping[str, object], _freeze_value(value))


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return deepcopy(value)


def _require_positive_int(value: object, field_name: str) -> None:
    """要求值为正整数。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")


def _require_non_negative_int(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value < 0:
        raise ValueError(f"{field_name} cannot be negative")


def _clean_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _require_text(value: object, field_name: str) -> str:
    text = _clean_text(value)
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text


def _optional_text(value: object) -> str | None:
    return _clean_text(value) or None


__all__ = [
    "ApprovalChallenge",
    "ApprovalGrant",
    "ApprovalPolicy",
    "ApprovalResponse",
    "ApprovalScope",
    "ConcurrencyPolicy",
    "ContentTrust",
    "EffectCertainty",
    "EffectStatus",
    "OutputLimits",
    "OutputTrustPolicy",
    "PermissionDecision",
    "PermissionEffect",
    "PermissionEngine",
    "PermissionRule",
    "RiskLevel",
    "TimeoutPolicy",
    "ToolAccessRequest",
    "ToolAccessResolution",
    "ToolEffect",
    "ToolEffectKind",
    "ToolMode",
    "ToolPolicy",
    "ToolResource",
    "approval_challenge_data",
    "approval_fingerprint",
    "build_approval_challenge",
    "issue_approval_grant",
]