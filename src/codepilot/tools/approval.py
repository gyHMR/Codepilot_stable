from __future__ import annotations

"""Tool approval abstractions.

The current runtime has no interactive approval resume loop yet, but this
module defines the boundary used by ToolRuntime and future CLI/Web approval UI.
"""

from dataclasses import dataclass
from typing import Protocol

from .permissions import ToolDecision
from .types import ToolMetadata, ToolRuntimeRequest


@dataclass(frozen=True)
class ApprovalDecision:
    approved: bool
    reason: str = ""
    approval_id: str | None = None


class ApprovalProvider(Protocol):
    async def request_approval(
        self,
        request: ToolRuntimeRequest,
        metadata: ToolMetadata | None,
        decision: ToolDecision,
    ) -> ApprovalDecision: ...


class DenyApprovalProvider:
    """Default provider used until a CLI/Web approval flow is connected."""

    async def request_approval(
        self,
        request: ToolRuntimeRequest,
        metadata: ToolMetadata | None,
        decision: ToolDecision,
    ) -> ApprovalDecision:
        _ = request, metadata
        return ApprovalDecision(approved=False, reason=decision.reason)
