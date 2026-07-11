from __future__ import annotations

"""
权限决策模块 — 针对已准备就绪的工具调用请求，决定其执行许可。

=============================================================================
文件目的
=============================================================================
本文件是 Codepilot 工具系统的安全守门人。在 AI 模型生成工具调用请求后、
实际执行之前，由 PermissionPolicy 根据以下维度逐层裁决每个请求：

  1. 模型是否试图越权（伪造授权参数）—— 直接拒绝。
  2. 工具元数据是否存在且对当前会话模式可见 —— 不可见则拒绝。
  3. 全局权限模式（只读 / 工作区写入 / 询问）是否允许该操作。
  4. 对于 Shell 命令，还需进行命令分类（只读、验证、变更、高风险）
     并匹配黑白名单。

判决结果封装为 ToolDecision，包含三种结果：allow（允许）、deny（拒绝）、
approval_required（需用户审批）。

=============================================================================
决策链概述
=============================================================================
PermissionPolicy.decide() 按以下顺序进行短路裁决（任一命中即返回）：

  1. 禁止模型伪造授权参数  → deny
  2. 元数据缺失或不可见     → deny
  3. 只读模式下非只读工具   → deny
  4. read/plan 模式下非只读  → deny
  5. bash 命令              → 进入 _decide_shell() 子链
  6. 用户已审批恢复         → allow
  7. 元数据标记需审批       → approval_required
  8. 高风险工具 + 审批开关  → approval_required
  9. ask 模式下非只读工具   → approval_required
  10. 以上均未命中          → allow（默认放行）
"""

import re
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from .contracts import ToolCallRequest
from .sandbox import classify_shell_command, command_mentions_internal_state

# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------

# ToolDecisionKind: 工具决策的三种结果类型
#   "allow"             — 允许直接执行
#   "deny"              — 拒绝执行
#   "approval_required" — 需要用户审批后才能执行
ToolDecisionKind = Literal["allow", "deny", "approval_required"]

# ToolPermissionMode: 全局权限模式的三种级别
#   "read-only"       — 只读模式：仅允许不修改文件系统/网络的操作
#   "workspace-write" — 工作区写入模式：允许在工作区范围内进行修改
#   "ask"             — 询问模式：任何可能修改的操作都需要用户审批
ToolPermissionMode = Literal["read-only", "workspace-write", "ask"]

# 有效决策类型集合（用于运行时校验）
_DECISIONS = frozenset({"allow", "deny", "approval_required"})

# 有效权限模式集合（用于运行时校验）
_PERMISSION_MODES = frozenset({"read-only", "workspace-write", "ask"})

# 禁止模型在工具参数中使用的授权关键字
# 这些是 AI 模型可能尝试伪造以绕过权限检查的参数名，任何包含这些键的请求将被直接拒绝。
# 例如模型可能在参数中添加 {"bypass_approval": true} 试图跳过审批。
_FORBIDDEN_MODEL_AUTH_KEYS = {
    "allow_dangerous",        # 试图标记为"允许危险操作"
    "bypass_approval",        # 试图绕过审批流程
    "ignore_workspace_boundary",  # 试图忽略工作区边界
    "trusted",                # 试图自称为"可信"来源
    "sudo",                   # 试图获取超级用户权限
    "force_without_approval", # 试图强制执行而不经审批
}


# ===========================================================================
# ToolDecision — 权限决策结果
# ===========================================================================
@dataclass(frozen=True)
class ToolDecision:
    """封装对单个工具调用请求的权限决策结果。

    这是一个不可变（frozen）数据类，一经创建便不可修改，确保决策结果的
    稳定性和可追踪性。

    属性:
        kind:    决策类型 — "allow" / "deny" / "approval_required"
        reason:  决策原因的简短标识符（如 "policy_allow", "read_only_permission_mode"）
        details: 决策上下文详情字典，包含工具名、分类、风险等级等信息
    """

    kind: ToolDecisionKind
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """初始化后校验：确保 kind 是有效的决策类型，reason 是字符串，
        details 是字典。通过 object.__setattr__ 在 frozen dataclass 中
        进行规范化处理。"""
        kind = str(self.kind).strip()
        if kind not in _DECISIONS:
            raise ValueError(f"Unknown tool decision: {self.kind}")
        # frozen dataclass 不允许直接赋值，使用 object.__setattr__ 绕过限制
        object.__setattr__(self, "kind", cast(ToolDecisionKind, kind))
        object.__setattr__(self, "reason", str(self.reason).strip())
        if not isinstance(self.details, dict):
            raise TypeError("ToolDecision details must be a dict")
        object.__setattr__(self, "details", dict(self.details))

    @property
    def allowed(self) -> bool:
        """该工具调用是否被允许直接执行。"""
        return self.kind == "allow"

    @property
    def denied(self) -> bool:
        """该工具调用是否被拒绝执行。"""
        return self.kind == "deny"

    @property
    def requires_approval(self) -> bool:
        """该工具调用是否需要用户审批后方可执行。"""
        return self.kind == "approval_required"


# ===========================================================================
# PermissionPolicy — 权限策略引擎
# ===========================================================================
@dataclass(frozen=True)
class PermissionPolicy:
    """根据配置的策略规则，对已准备就绪的工具请求做出 allow/deny/approval_required 决策。

    这是整个权限系统的核心类。它整合了全局权限模式、Shell 命令分类、
    白名单/黑名单模式匹配、风险等级判定等多个维度，形成一条有序的决策链。

    配置属性:
        permission_mode:              全局权限模式（read-only / workspace-write / ask）
        block_dangerous_bash:          是否阻止高风险 Shell 命令（默认 True）
        bash_allow_patterns:           Shell 命令白名单（正则模式列表）
        bash_block_patterns:           Shell 命令黑名单（正则模式列表）
        require_approval_for_high_risk: 高风险工具是否需要审批（默认 True）
    """

    permission_mode: ToolPermissionMode = "workspace-write"
    block_dangerous_bash: bool = True
    bash_allow_patterns: list[str] | None = None
    bash_block_patterns: list[str] | None = None
    require_approval_for_high_risk: bool = True

    def __post_init__(self) -> None:
        """初始化后校验：规范化 permission_mode 并验证所有正则模式的有效性。"""
        object.__setattr__(
            self,
            "permission_mode",
            _ensure_permission_mode(self.permission_mode),
        )
        _validate_patterns(self.bash_allow_patterns)
        _validate_patterns(self.bash_block_patterns)

    # -----------------------------------------------------------------------
    # decide() — 主决策入口
    # -----------------------------------------------------------------------
    def decide(self, request: ToolCallRequest) -> ToolDecision:
        """对工具调用请求执行完整决策链，返回 ToolDecision。

        决策链按以下顺序逐层检查（短路逻辑，一旦命中即返回）：

        第 1 步 — 防越权检查
            扫描请求参数中是否包含 _FORBIDDEN_MODEL_AUTH_KEYS 中的任何键。
            AI 模型可能尝试在参数中注入 "bypass_approval" 等字段来绕过
            权限系统。一旦发现，立即拒绝，不进行后续检查。

        第 2 步 — 元数据存在性检查
            如果工具的 metadata 为 None，说明该工具未在系统中注册或配置，
            出于安全考虑直接拒绝。

        第 3 步 — 会话模式可见性检查
            每个工具定义了在哪些会话模式（如 "chat", "read", "plan"）下可见。
            如果当前模式不在可见列表中，拒绝该请求。

        第 4 步 — 只读权限模式检查
            当全局 permission_mode 为 "read-only" 时，所有非只读工具
            （metadata.read_only == False）均被拒绝。

        第 5 步 — read/plan 会话模式限制
            在 "read" 或 "plan" 会话模式下，不允许执行非只读工具。
            这与第 4 步形成双重保护。

        第 6 步 — Shell 命令特判
            如果工具名为 "bash"，转入 _decide_shell() 方法进行专门的
            命令级决策（含命令分类、黑白名单匹配等）。

        第 7 步 — 用户已审批恢复
            如果请求的来源是 "approval_resume"（用户已审批后恢复执行），
            直接放行，无需重复检查。

        第 8 步 — 工具元数据审批标记
            如果工具的 metadata.requires_approval 为 True，说明该工具
            在设计上就需要审批，返回 approval_required。

        第 9 步 — 高风险工具审批
            当 require_approval_for_high_risk 配置为 True 且工具风险等级
            为 "high" 时，要求用户审批。

        第 10 步 — ask 模式
            当全局模式为 "ask" 时，所有非只读工具都需要审批。

        第 11 步 — 默认放行
            以上所有检查均未命中，说明该请求属于安全的常规操作，直接允许。

        参数:
            request: 已准备就绪的工具调用请求，包含工具名、参数、元数据、当前模式等信息

        返回:
            ToolDecision 对象，包含 kind（allow/deny/approval_required）、
            reason 和 details
        """
        metadata = request.metadata
        # 构建决策详情字典，用于后续日志记录和调试
        details = _decision_details(request, permission_mode=self.permission_mode)

        # ---- 第 1 步：防越权检查 ----
        # 求 request.arguments 的键与禁止授权键的交集
        attempted_auth = sorted(_FORBIDDEN_MODEL_AUTH_KEYS.intersection(request.arguments))
        if attempted_auth:
            return ToolDecision(
                "deny",
                "model_authorization_forbidden",
                {**details, "forbidden_params": attempted_auth},
            )

        # ---- 第 2 步：元数据存在性检查 ----
        if metadata is None:
            return ToolDecision("deny", "tool_metadata_missing", details)

        # ---- 第 3 步：会话模式可见性检查 ----
        if not metadata.visible_in(request.current_mode):
            return ToolDecision("deny", "mode_scope_denied", details)

        # ---- 第 4 步：只读权限模式检查 ----
        if self.permission_mode == "read-only" and not metadata.read_only:
            return ToolDecision("deny", "read_only_permission_mode", details)

        # ---- 第 5 步：read/plan 会话模式限制 ----
        if request.current_mode in {"read", "plan"} and not metadata.read_only:
            return ToolDecision("deny", "mode_scope_denied", details)

        # ---- 第 6 步：Shell 命令特判 ----
        if request.name == "bash":
            return self._decide_shell(request, details)

        # ---- 第 7 步：用户已审批恢复 ----
        if request.source == "approval_resume":
            return ToolDecision("allow", "approved_by_user", details)

        # ---- 第 8 步：工具元数据审批标记 ----
        if metadata.requires_approval:
            return ToolDecision("approval_required", "tool_metadata_requires_approval", details)

        # ---- 第 9 步：高风险工具审批 ----
        if self.require_approval_for_high_risk and metadata.risk_level == "high":
            return ToolDecision("approval_required", "high_risk_tool_requires_approval", details)

        # ---- 第 10 步：ask 模式 ----
        if self.permission_mode == "ask" and not metadata.read_only:
            return ToolDecision("approval_required", "ask_mode", details)

        # ---- 第 11 步：默认放行 ----
        return ToolDecision("allow", "policy_allow", details)

    # -----------------------------------------------------------------------
    # _decide_shell() — Shell 命令子决策链
    # -----------------------------------------------------------------------
    def _decide_shell(
        self,
        request: ToolCallRequest,
        details: dict[str, Any],
    ) -> ToolDecision:
        """对 Shell（bash）命令进行专门的分类和权限决策。

        Shell 命令的决策比普通工具更细粒度，因为即使是同一个 "bash" 工具，
        不同的命令可能有完全不同的风险等级。本方法按以下顺序逐层裁决：

        1. 黑名单匹配 → deny
           如果命令文本匹配 bash_block_patterns 中的任何正则，直接拒绝。

        2. 内部状态变更检测 → deny
           如果命令涉及修改 Codepilot 内部状态文件
           （command_mentions_internal_state 返回 True），直接拒绝。

        3. 高风险命令 + 阻止开关 → deny
           如果命令被 classify_shell_command 分类为 "high_risk" 且
           block_dangerous_bash 配置为 True，直接拒绝。

        4. 用户已审批恢复 → allow
           来源为 "approval_resume" 的请求直接放行。

        5. 白名单匹配 → allow
           如果命令文本匹配 bash_allow_patterns 中的任何正则，直接放行。

        6. 验证/只读命令 → allow
           命令分类为 "verification"（验证类，如 ls、stat、git status）或
           "read_only"（只读类，如 cat、head），直接放行。

        7. 变更命令 + 工作区写入模式 → allow
           命令分类为 "mutation" 且权限模式为 "workspace-write" 时放行。

        8. 变更命令 + 非工作区写入模式 → approval_required
           需要用户审批。

        9. 兜底 → approval_required
           无法分类的命令默认要求审批。

        参数:
            request: 工具调用请求（其 arguments 中应包含 "command" 字段）
            details: 已构建的决策详情字典

        返回:
            ToolDecision 对象
        """
        # 从参数中提取命令文本
        command = str(request.arguments.get("command", "")).strip()

        # 对命令进行分类：high_risk / mutation / verification / read_only / unknown
        classification = classify_shell_command(command)

        # 构建包含命令信息的详情字典
        shell_details = {**details, "command": command, "shell_class": classification}

        # ---- 第 1 层：黑名单匹配 ----
        if _matches_any(command, self.bash_block_patterns):
            return ToolDecision("deny", "block_pattern", shell_details)

        # ---- 第 2 层：内部状态变更检测 ----
        if command_mentions_internal_state(command):
            return ToolDecision("deny", "internal_state_mutation_forbidden", shell_details)

        # ---- 第 3 层：高风险命令阻止 ----
        if classification == "high_risk" and self.block_dangerous_bash:
            return ToolDecision("deny", "dangerous_command", shell_details)

        # ---- 第 4 层：用户已审批恢复 ----
        if request.source == "approval_resume":
            return ToolDecision("allow", "approved_by_user", shell_details)

        # ---- 第 5 层：白名单匹配 ----
        if _matches_any(command, self.bash_allow_patterns):
            return ToolDecision("allow", "allow_pattern", shell_details)

        # ---- 第 6 层：验证类 / 只读类命令放行 ----
        if classification in {"verification", "read_only"}:
            return ToolDecision("allow", f"{classification}_command", shell_details)

        # ---- 第 7 层：变更类命令的分支处理 ----
        if classification == "mutation":
            if self.permission_mode == "workspace-write":
                # 工作区写入模式允许变更类命令
                return ToolDecision("allow", "workspace_mutation_command", shell_details)
            # 非工作区写入模式（read-only 或 ask）下的变更命令需审批
            return ToolDecision("approval_required", "ask_mode", shell_details)

        # ---- 第 8 层：未知命令兜底 ----
        return ToolDecision("approval_required", "unknown_shell_command", shell_details)


# ===========================================================================
# 模块级辅助函数
# ===========================================================================

def _decision_details(
    request: ToolCallRequest,
    *,
    permission_mode: str,
) -> dict[str, Any]:
    """从工具调用请求中提取决策所需的上下文详情。

    这些详情用于日志记录、调试追踪和审计。如果请求的 metadata 为 None，
    则只包含基本字段（tool, current_mode, permission_mode）；否则包含
    完整的工具元数据信息。

    参数:
        request:         工具调用请求
        permission_mode: 当前生效的全局权限模式字符串

    返回:
        包含工具标识、分类、风险等级、作用域、网络访问、凭证需求、
        当前会话模式和权限模式等信息的字典
    """
    metadata = request.metadata
    if metadata is None:
        return {
            "tool": request.name,
            "current_mode": request.current_mode,
            "permission_mode": permission_mode,
        }
    # 从 extra 字段中提取 capabilities 列表
    capabilities = metadata.extra.get("capabilities", [])
    return {
        "tool": metadata.name,
        "category": metadata.category,          # 工具分类（如 file, shell, network）
        "risk_level": metadata.risk_level,      # 风险等级（low / medium / high）
        "scopes": list(metadata.scopes),        # 工具的作用域列表
        "network_access": metadata.network_access,      # 是否需要网络访问
        "credential_required": metadata.credential_required,  # 是否需要凭证
        "current_mode": request.current_mode,   # 当前会话模式
        "permission_mode": permission_mode,     # 当前权限模式
        # 将 capabilities 转为列表（如果本身是 list 或 tuple 则直接用，否则为空）
        "capabilities": list(capabilities) if isinstance(capabilities, (list, tuple)) else [],
    }


def _ensure_permission_mode(value: object) -> ToolPermissionMode:
    """校验并规范化权限模式值。

    将输入转为字符串后去空白，检查是否在 _PERMISSION_MODES 有效集合中。
    如果不在，抛出 ValueError。

    参数:
        value: 待校验的权限模式值（字符串或其他可转字符串的对象）

    返回:
        规范化后的 ToolPermissionMode 字面量

    异常:
        ValueError: 当权限模式不在 {"read-only", "workspace-write", "ask"} 中时
    """
    text = str(value).strip()
    if text not in _PERMISSION_MODES:
        raise ValueError(f"Unknown permission mode: {value}")
    return cast(ToolPermissionMode, text)


def _matches_any(text: str, patterns: list[str] | None) -> bool:
    """检查文本是否匹配给定正则模式列表中的任意一个。

    使用 re.search（部分匹配），而非 re.fullmatch（全匹配），
    只要模式出现在文本中的任意位置即视为匹配。
    如果 patterns 为 None 或空列表，直接返回 False。

    参数:
        text:     待检查的文本
        patterns: 正则表达式模式列表（可为 None）

    返回:
        True 如果文本匹配至少一个模式；否则 False
    """
    return any(re.search(pattern, text) is not None for pattern in patterns or [])


def _validate_patterns(patterns: list[str] | None) -> None:
    """验证正则模式列表中的所有模式是否可编译。

    在 PermissionPolicy 初始化时调用，提前发现无效的正则表达式，
    避免在实际匹配时才暴露错误。

    参数:
        patterns: 正则表达式模式列表（可为 None）

    异常:
        ValueError: 当某个模式不是有效的正则表达式时
    """
    for pattern in patterns or []:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Invalid permission regex {pattern!r}: {exc}") from exc


# 模块公开接口
__all__ = [
    "PermissionPolicy",
    "ToolDecision",
    "ToolDecisionKind",
    "ToolPermissionMode",
]
