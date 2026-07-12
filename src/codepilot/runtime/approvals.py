from __future__ import annotations

from dataclasses import dataclass

from codepilot.tools.security import ApprovalChallenge


@dataclass(frozen=True)
class ApprovalView:
    approval_id: str
    session_id: str
    run_id: str
    tool_call_id: str
    tool_name: str
    reason: str = ""
    risk_level: str = "unknown"


@dataclass(frozen=True)
class ApprovalTransaction:
    session_id: str
    challenge: ApprovalChallenge

    @property
    def approval_id(self) -> str:
        return self.challenge.approval_id

    def view(self) -> ApprovalView:
        return ApprovalView(
            approval_id=self.approval_id,
            session_id=self.session_id,
            run_id=self.challenge.run_id,
            tool_call_id=self.challenge.tool_call_id,
            tool_name=self.challenge.tool_name,
            reason=self.challenge.reason,
            risk_level=self.challenge.risk,
        )


class ApprovalRegistry:
    def __init__(self) -> None:
        self._items: dict[str, ApprovalTransaction] = {}

    def add(self, session_id: str, challenge: ApprovalChallenge) -> None:
        transaction = ApprovalTransaction(
            session_id=session_id,
            challenge=challenge,
        )
        self._items[transaction.approval_id] = transaction

    def get(self, approval_id: str) -> ApprovalTransaction | None:
        return self._items.get(approval_id)

    def pop(self, approval_id: str) -> ApprovalTransaction | None:
        return self._items.pop(approval_id, None)

    def remove_session(self, session_id: str) -> None:
        self._items = {
            approval_id: item
            for approval_id, item in self._items.items()
            if item.session_id != session_id
        }

    def clear(self) -> None:
        self._items.clear()

    def list(self, session_id: str | None = None) -> list[ApprovalView]:
        views = [
            item.view()
            for item in self._items.values()
            if session_id is None or item.session_id == session_id
        ]
        return sorted(views, key=lambda item: item.approval_id)
