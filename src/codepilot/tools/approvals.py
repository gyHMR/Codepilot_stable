from __future__ import annotations

"""
工具审批（Approval）模块。

本模块定义了工具执行审批的数据结构和提供者接口。
当权限策略判定某个工具调用需要用户审批时，ToolRuntime 会调用
ApprovalProvider 请求审批决定。审批请求包含工具名称、参数预览、
风险级别等信息，供 CLI/Web 界面展示给用户。

核心流程:
    1. ToolRuntime 执行管线判断 decision.requires_approval == True
    2. 调用 approval_provider.request_approval(call, decision)
    3. 如果 approved=True → 直接执行工具
    4. 如果 deferred=True → 暂停执行，返回 approval_required 状态
       等待用户确认后通过 ToolRuntime.resume() 继续
    5. 如果 approved=False → 拒绝执行
"""

import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from .contracts import PreparedToolCall, ToolMetadata
from .permissions import ToolDecision


@dataclass(frozen=True)
class ApprovalDecision:
    """
    审批决策结果（不可变）。

    由 ApprovalProvider 返回，包含审批决定和元数据。

    属性:
        approved: 是否批准执行。True=允许, False=拒绝或延迟。
        reason: 决策原因，用于向用户展示为什么需要审批/为什么被拒绝。
        approval_id: 审批唯一标识，用于 resume() 时关联待审批调用。
        deferred: 是否为延迟审批。True 表示暂停等待用户操作，
                  False 表示已有最终决定（批准或拒绝）。
    """

    approved: bool
    reason: str = ""
    approval_id: str | None = None
    deferred: bool = False

    def __post_init__(self) -> None:
        """校验字段类型并规范化文本值。"""
        if not isinstance(self.approved, bool):
            raise TypeError("ApprovalDecision approved must be bool")
        if not isinstance(self.deferred, bool):
            raise TypeError("ApprovalDecision deferred must be bool")
        # 用 object.__setattr__ 绕过 frozen dataclass 的限制来设置规范化值
        object.__setattr__(self, "reason", _clean_text(self.reason))
        if self.approval_id is not None:
            object.__setattr__(self, "approval_id", _require_text(self.approval_id, "approval_id"))


@dataclass(frozen=True)
class ApprovalRequest:
    """
    审批请求（不可变），由 Runtime 构建后传递给 UI 层展示。

    包含用户审批界面需要的所有信息：哪个工具、什么参数、什么风险。

    属性:
        approval_id: 审批唯一标识（格式: approval_ + 12位UUID hex）。
        run_id: 所属运行 ID。
        tool_call_id: 工具调用 ID。
        tool_name: 工具名称（如 "bash", "write", "edit" 等）。
        params_preview: 参数预览，经过脱敏/截断处理，不同工具有不同的预览策略。
        reason: 需要审批的原因（来自权限策略的 decision.reason）。
        risk_level: 风险级别（low/medium/high）。
        capabilities: 工具能力标签列表。
    """

    approval_id: str
    run_id: str
    tool_call_id: str
    tool_name: str
    params_preview: dict[str, object]
    reason: str
    risk_level: str
    capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        """校验所有必填字段非空，规范化文本值。"""
        object.__setattr__(self, "approval_id", _require_text(self.approval_id, "approval_id"))
        object.__setattr__(self, "run_id", _require_text(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "tool_call_id",
            _require_text(self.tool_call_id, "tool_call_id"),
        )
        object.__setattr__(self, "tool_name", _require_text(self.tool_name, "tool_name"))
        if not isinstance(self.params_preview, dict):
            raise TypeError("ApprovalRequest params_preview must be a dict")
        object.__setattr__(self, "params_preview", dict(self.params_preview))
        object.__setattr__(self, "reason", _clean_text(self.reason))
        object.__setattr__(self, "risk_level", _require_text(self.risk_level, "risk_level"))
        object.__setattr__(
            self,
            "capabilities",
            tuple(_clean_unique_items(self.capabilities)),
        )


class ApprovalProvider(Protocol):
    """
    审批提供者协议（Protocol 类型，不需要显式继承）。

    由 CLI（CliApprovalProvider）、Web 接口或外部系统实现。
    ToolRuntime 不关心谁提供审批——只要实现了这个协议即可。

    方法:
        request_approval(call, decision) → ApprovalDecision
            请求对一次工具调用的审批决定。参数:
            - call: 已准备好的工具调用（包含定义和请求）。
            - decision: 权限策略的决策结果。
            返回审批决定（批准/拒绝/延迟）。
    """

    async def request_approval(
        self,
        call: PreparedToolCall,
        decision: ToolDecision,
    ) -> ApprovalDecision:
        """
        请求审批决定。

        参数:
            call: 已准备好的工具调用。
            decision: 权限策略的决策结果（包含拒绝原因等）。

        返回:
            ApprovalDecision 对象，包含 approved/deferred/approval_id。
        """
        ...


class DeferredApprovalProvider:
    """
    默认的延迟审批提供者。

    不直接批准也不直接拒绝——它总是返回 deferred=True，
    暂停执行并将审批请求交给上层（CLI/Web）处理。
    上层通过 ToolRuntime.resume() 恢复执行。

    这是"安全默认"策略：不确定时先暂停，等用户明确确认。
    """

    async def request_approval(
        self,
        call: PreparedToolCall,
        decision: ToolDecision,
    ) -> ApprovalDecision:
        """
        始终返回延迟审批决定。

        生成唯一的 approval_id 并返回 deferred=True，
        暂停执行等待用户通过 ToolRuntime.resume() 确认。
        """
        # 构建包含所有展示信息的审批请求
        request = build_approval_request(call, decision)
        return ApprovalDecision(
            approved=False,            # 不自动批准
            reason=decision.reason,    # 传递权限策略的拒绝原因
            approval_id=request.approval_id,  # 用于后续 resume
            deferred=True,             # 标记为延迟审批
        )


def build_approval_request(
    call: PreparedToolCall,
    decision: ToolDecision,
) -> ApprovalRequest:
    """
    从准备好的工具调用和权限决策构建审批请求。

    这是 ToolRuntime 和 UI 层之间的数据转换函数。
    它提取工具元数据和决策信息，生成 UI 需要的 ApprovalRequest。

    参数:
        call: 已准备好的工具调用（包含定义和请求）。
        decision: 权限策略的决策结果。

    返回:
        构建好的 ApprovalRequest，可直接传给 UI 渲染。
    """
    metadata = call.metadata
    # 从决策详情中提取能力标签
    capabilities = decision.details.get("capabilities", [])
    return ApprovalRequest(
        # 生成唯一审批 ID: approval_ + UUID hex 前 12 位
        approval_id=f"approval_{uuid.uuid4().hex[:12]}",
        run_id=call.request.run_id,
        tool_call_id=call.request.tool_call_id,
        tool_name=call.request.name,
        # 为不同工具类型生成安全的参数预览
        params_preview=_params_preview(call.request.name, call.request.arguments),
        reason=decision.reason,
        risk_level=_risk_level(metadata, decision),
        capabilities=tuple(str(item) for item in capabilities if isinstance(item, str)),
    )


def _risk_level(metadata: ToolMetadata | None, decision: ToolDecision) -> str:
    """
    确定审批显示的风险级别。

    优先从 decision.details 中读取（可能被权限策略覆盖），
    否则从工具 metadata 获取，兜底为 "medium"。
    """
    if decision.details.get("risk_level"):
        return str(decision.details["risk_level"])
    if metadata is not None:
        return metadata.risk_level
    return "medium"


def _params_preview(name: str, params: dict[str, Any]) -> dict[str, object]:
    """
    为不同工具生成安全的参数预览。

    每种工具类型有不同的预览策略，目的是:
    - 给用户足够的上下文来判断是否批准
    - 脱敏敏感信息（密钥、密码等）
    - 截断过长内容避免 UI 溢出

    工具特殊处理:
        - write: 显示 path + content 字符数 + overwrite 标志
        - edit/apply_patch: 显示除 content 外的前 8 个参数
        - bash: 显示 command（截断到 2000 字符）+ cwd + timeout
        - 其他: 显示前 12 个参数，敏感 key 自动脱敏
    """
    if name == "write":
        return {
            "path": str(params.get("path", ""))[:300],
            "content_chars": len(str(params.get("content", ""))),
            "overwrite": bool(params.get("overwrite", True)),
        }
    if name in {"edit", "apply_patch"}:
        return {
            key: _safe_preview(value)
            for key, value in list(params.items())[:8]
            if key != "content"
        }
    if name == "bash":
        return {
            "command": str(params.get("command", ""))[:2000],
            "cwd": str(params.get("cwd", "."))[:300],
            "timeout_seconds": params.get("timeout_seconds", 30),
        }
    # 通用预览：对敏感 key 名自动脱敏
    return {
        str(key): (
            "[REDACTED]"
            if any(
                marker in str(key).upper()
                for marker in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "CREDENTIAL", "COOKIE")
            )
            else _safe_preview(value)
        )
        for key, value in list(params.items())[:12]
    }


def _safe_preview(value: object) -> object:
    """
    安全地将参数值转换为预览形式。

    - 字符串: 截断到 300 字符
    - 布尔值/数字/None: 原样返回
    - 其他类型: 返回类型名（如 "<dict>"）
    """
    if isinstance(value, str):
        return value[:300]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return f"<{type(value).__name__}>"


def _clean_text(value: object) -> str:
    """安全地去除空白符, None 返回空字符串。"""
    return str(value).strip() if value is not None else ""


def _require_text(value: object, field_name: str) -> str:
    """清理文本并确保非空，否则抛出 ValueError。"""
    text = _clean_text(value)
    if not text:
        raise ValueError(f"Approval {field_name} cannot be empty")
    return text


def _clean_unique_items(values: tuple[str, ...]) -> list[str]:
    """对字符串元组去重、清理空白，保持顺序。"""
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        if text and text not in seen:
            cleaned.append(text)
            seen.add(text)
    return cleaned


__all__ = [
    "ApprovalDecision",
    "ApprovalProvider",
    "ApprovalRequest",
    "DeferredApprovalProvider",
    "build_approval_request",
]
