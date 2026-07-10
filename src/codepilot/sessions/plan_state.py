from __future__ import annotations

"""Session-owned persistence for the current structured execution plan."""

from typing import Any, Mapping

from codepilot.core.plan import (
    PlanState,
    PlanValidationError,
    load_plan_state,
)


class PlanStateStore:
    """Read and write the session plan_state.json file."""

    def __init__(self, session_store: Any) -> None:
        self.session_store = session_store

    def load(self) -> dict[str, Any] | None:
        raw = self.session_store.load_plan_state()
        plan = load_plan_state(raw)
        return plan.to_dict() if plan is not None else None

    def current(self) -> dict[str, Any] | None:
        return self.load()

    def save(self, state: Mapping[str, Any] | PlanState) -> dict[str, Any]:
        plan = load_plan_state(state)
        if plan is None:
            raise PlanValidationError("plan state cannot be None")
        payload = plan.to_dict()
        self.session_store.save_plan_state(payload)
        return payload

    def approve_current(self, *, run_id: str | None = None) -> dict[str, Any] | None:
        current = load_plan_state(self.current())
        if current is None or current.status != "proposed":
            return current.to_dict() if current is not None else None
        if run_id is not None and current.owner_run_id != run_id:
            raise PlanValidationError("plan belongs to a different run")
        return self.save(current.approve())

    def reject_current(self, *, run_id: str | None = None) -> dict[str, Any] | None:
        current = load_plan_state(self.current())
        if current is None:
            return None
        if current.status != "proposed":
            return current.to_dict()
        if run_id is not None and current.owner_run_id != run_id:
            raise PlanValidationError("plan belongs to a different run")
        return self.save(current.reject())

    def abandon_current(
        self,
        *,
        run_id: str | None = None,
        source: str = "user_abandoned",
    ) -> dict[str, Any] | None:
        current = load_plan_state(self.current())
        if current is None:
            return None
        if current.status == "abandoned":
            return current.to_dict()
        if run_id is not None and current.owner_run_id != run_id:
            raise PlanValidationError("plan belongs to a different run")
        return self.save(current.abandon(source=source))

def validate_plan_state_payload(raw: object) -> dict[str, Any]:
    plan = load_plan_state(raw)
    if plan is None:
        raise PlanValidationError("plan state cannot be None")
    return plan.to_dict()


__all__ = ["PlanStateStore", "PlanValidationError", "validate_plan_state_payload"]
