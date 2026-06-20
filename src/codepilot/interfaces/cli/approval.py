from __future__ import annotations

"""Interactive one-shot tool approval for CLI sessions."""

import asyncio
from typing import Callable

from codepilot.tools.approval import (
    ApprovalDecision,
    ApprovalRequest,
    build_approval_request,
)
from codepilot.tools.permissions import ToolDecision
from codepilot.tools.types import ToolMetadata, ToolRuntimeRequest


class CliApprovalProvider:
    def __init__(
        self,
        *,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        self.input_fn = input_fn
        self.output_fn = output_fn

    async def request_approval(
        self,
        request: ToolRuntimeRequest,
        metadata: ToolMetadata | None,
        decision: ToolDecision,
    ) -> ApprovalDecision:
        approval = build_approval_request(request, metadata, decision)
        self._render(approval)
        answer = await asyncio.to_thread(
            self.input_fn,
            "Approve once? [y/N] ",
        )
        approved = answer.strip().lower() in {"y", "yes"}
        return ApprovalDecision(
            approved=approved,
            reason="user_approved" if approved else "user_denied",
            approval_id=approval.approval_id,
        )

    def _render(self, approval: ApprovalRequest) -> None:
        self.output_fn("")
        self.output_fn("Tool approval required")
        self.output_fn(f"  Tool: {approval.tool_name}")
        self.output_fn(f"  Reason: {approval.reason}")
        self.output_fn(f"  Risk: {approval.risk_level}")
        if approval.capabilities:
            self.output_fn(f"  Capabilities: {', '.join(approval.capabilities)}")
        for key, value in approval.params_preview.items():
            self.output_fn(f"  {key}: {value}")


__all__ = ["CliApprovalProvider"]
