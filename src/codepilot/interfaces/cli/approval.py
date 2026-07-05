from __future__ import annotations

# 新手导读：CLI 审批 provider 在终端展示工具风险并等待用户确认。
# 关注点：审批结果会回到 tools/runtime，而不是在 CLI 里直接执行工具。

"""CLI 会话的交互式工具审批提供者。

当工具执行需要用户审批时，在终端显示工具信息并等待用户确认。
"""

import asyncio
from typing import Callable

from .render import format_plain_panel


class _CliApprovalDecision:
    def __init__(self, *, approved: bool, reason: str, approval_id: str) -> None:
        self.approved = approved
        self.reason = reason
        self.approval_id = approval_id


class CliApprovalProvider:
    """CLI 工具审批提供者：在终端显示审批提示并等待用户输入 y/N。"""

    def __init__(
        self,
        *,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        if not callable(input_fn):
            raise TypeError("CliApprovalProvider.input_fn must be callable")
        if not callable(output_fn):
            raise TypeError("CliApprovalProvider.output_fn must be callable")
        self.input_fn = input_fn
        self.output_fn = output_fn

    async def request_approval(
        self,
        request: object,
        metadata: object | None,
        decision: object,
    ) -> _CliApprovalDecision:
        """请求用户审批：渲染审批信息，等待用户输入，返回审批结果。"""
        approval_id = str(getattr(request, "tool_call_id", "") or "")
        self._render(request, metadata, decision)
        answer = await asyncio.to_thread(
            self.input_fn,
            "CP approve once? [y/N] ",
        )
        approved = answer.strip().lower() in {"y", "yes"}
        return _CliApprovalDecision(
            approved=approved,
            reason="user_approved" if approved else "user_denied",
            approval_id=approval_id,
        )

    def _render(self, request: object, metadata: object | None, decision: object) -> None:
        """渲染审批提示信息：工具名、原因、风险等级、能力要求和参数预览。"""
        params = getattr(request, "params", {}) or {}
        capabilities = getattr(metadata, "capabilities", ()) if metadata is not None else ()
        rows: list[tuple[str, object]] = [
            ("Tool", getattr(request, "name", "")),
            ("Reason", getattr(decision, "reason", "")),
            ("Risk", getattr(metadata, "risk_level", "unknown") if metadata is not None else "unknown"),
        ]
        if capabilities:
            rows.append(("Capabilities", ", ".join(str(item) for item in capabilities)))
        for key, value in params.items():
            rows.append((key, value))
        self.output_fn("")
        for line in format_plain_panel("Tool approval required", rows):
            self.output_fn(line)


__all__ = ["CliApprovalProvider"]
