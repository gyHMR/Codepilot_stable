from __future__ import annotations

# 新手导读：CLI 审批 provider 在终端展示工具风险并等待用户确认。
# 关注点：审批结果会回到 tools/runtime，而不是在 CLI 里直接执行工具。

"""CLI 会话的交互式工具审批提供者。

当工具执行需要用户审批时，在终端显示工具信息并等待用户确认。
"""

import asyncio
from typing import Callable

from codepilot.tools.approval import (
    ApprovalDecision,
    ApprovalRequest,
    build_approval_request,
)
from codepilot.tools.policy import ToolDecision
from codepilot.tools.contracts import ToolMetadata, ToolRuntimeRequest

from .ui import format_plain_panel


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
        request: ToolRuntimeRequest,
        metadata: ToolMetadata | None,
        decision: ToolDecision,
    ) -> ApprovalDecision:
        """请求用户审批：渲染审批信息，等待用户输入，返回审批结果。"""
        approval = build_approval_request(request, metadata, decision)
        self._render(approval)
        answer = await asyncio.to_thread(
            self.input_fn,
            "CP approve once? [y/N] ",
        )
        approved = answer.strip().lower() in {"y", "yes"}
        return ApprovalDecision(
            approved=approved,
            reason="user_approved" if approved else "user_denied",
            approval_id=approval.approval_id,
        )

    def _render(self, approval: ApprovalRequest) -> None:
        """渲染审批提示信息：工具名、原因、风险等级、能力要求和参数预览。"""
        rows: list[tuple[str, object]] = [
            ("Tool", approval.tool_name),
            ("Reason", approval.reason),
            ("Risk", approval.risk_level),
        ]
        if approval.capabilities:
            rows.append(("Capabilities", ", ".join(approval.capabilities)))
        for key, value in approval.params_preview.items():
            rows.append((key, value))
        self.output_fn("")
        for line in format_plain_panel("Tool approval required", rows):
            self.output_fn(line)


__all__ = ["CliApprovalProvider"]
