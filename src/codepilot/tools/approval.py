from __future__ import annotations

"""工具审批抽象层。

当前运行时尚未实现交互式审批恢复循环，但本模块定义了
ToolRuntime 和未来 CLI/Web 审批 UI 使用的边界接口。
"""

from dataclasses import dataclass
from typing import Protocol
import uuid

from .permissions import ToolDecision
from .types import ToolMetadata, ToolRuntimeRequest


@dataclass(frozen=True)
class ApprovalDecision:
    """审批决策结果。"""
    approved: bool
    reason: str = ""
    approval_id: str | None = None


@dataclass(frozen=True)
class ApprovalRequest:
    """审批请求：包含工具信息、参数预览和风险等级。"""
    approval_id: str
    tool_call_id: str
    tool_name: str
    params_preview: dict[str, object]
    reason: str
    risk_level: str
    capabilities: tuple[str, ...]


def build_approval_request(
    request: ToolRuntimeRequest,
    metadata: ToolMetadata | None,
    decision: ToolDecision,
) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=f"approval_{uuid.uuid4().hex[:12]}",
        tool_call_id=request.tool_call_id,
        tool_name=request.name,
        params_preview=_params_preview(request.name, request.params),
        reason=decision.reason,
        risk_level=(
            str(decision.details.get("risk_level"))
            if decision.details.get("risk_level")
            else metadata.risk_level if metadata else "medium"
        ),
        capabilities=tuple(
            str(item)
            for item in decision.details.get("capabilities", [])
            if isinstance(item, str)
        ),
    )


class ApprovalProvider(Protocol):
    """审批提供者协议：由 CLI/Web 等接口层实现。"""
    async def request_approval(
        self,
        request: ToolRuntimeRequest,
        metadata: ToolMetadata | None,
        decision: ToolDecision,
    ) -> ApprovalDecision: ...


class DenyApprovalProvider:
    """默认审批提供者：在 CLI/Web 审批流程接入前，所有审批请求都被拒绝。"""

    async def request_approval(
        self,
        request: ToolRuntimeRequest,
        metadata: ToolMetadata | None,
        decision: ToolDecision,
    ) -> ApprovalDecision:
        approval = build_approval_request(request, metadata, decision)
        return ApprovalDecision(
            approved=False,
            reason=decision.reason,
            approval_id=approval.approval_id,
        )


def _params_preview(name: str, params: dict[str, object]) -> dict[str, object]:
    if name == "write":
        return {
            "path": str(params.get("path", ""))[:300],
            "content_chars": len(str(params.get("content", ""))),
            "overwrite": bool(params.get("overwrite", True)),
        }
    if name == "edit":
        return {
            "path": str(params.get("path", ""))[:300],
            "old_text": str(params.get("old_text", ""))[:160],
            "new_text": str(params.get("new_text", ""))[:160],
            "replace_all": bool(params.get("replace_all", False)),
        }
    if name == "bash":
        return {
            "command": str(params.get("command", ""))[:2000],
            "cwd": str(params.get("cwd", "."))[:300],
            "timeout_seconds": params.get("timeout_seconds", 30),
        }
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
    if isinstance(value, str):
        return value[:300]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return f"<{type(value).__name__}>"


__all__ = [
    "ApprovalDecision",
    "ApprovalProvider",
    "ApprovalRequest",
    "DenyApprovalProvider",
    "build_approval_request",
]
