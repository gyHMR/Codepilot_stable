from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class ApprovalView:
    approval_id: str
    session_id: str
    run_id: str
    tool_call_id: str
    tool_name: str
    reason: str = ""
    risk_level: str = "unknown"

__all__ = ["ApprovalView"]
