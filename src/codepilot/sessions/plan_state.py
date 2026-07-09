from __future__ import annotations

"""Session-owned persistence for the current structured execution plan."""

from datetime import datetime, timezone
from typing import Any, Mapping

from codepilot.core.plan import (
    PlanState,
    PlanValidationError,
    RunMode,
    ensure_run_mode,
    load_plan_state,
)


class PlanStateStore:
    """Read and write the session plan_state.json file."""

    def __init__(self, session_store: Any) -> None:
        self.session_store = session_store

    def load(self) -> dict[str, Any] | None:
        return self.session_store.load_plan_state()

    def current(self) -> dict[str, Any] | None:
        return self.load()

    def save(self, state: Mapping[str, Any] | PlanState) -> dict[str, Any]:
        plan = load_plan_state(state)
        if plan is None:
            raise PlanValidationError("plan state cannot be None")
        payload = plan.to_dict()
        self.session_store.save_plan_state(payload)
        return payload

    def begin(
        self,
        objective: str,
        *,
        origin_mode: RunMode = "build",
        run_id: str | None = None,
    ) -> dict[str, Any]:
        if run_id is None:
            raise PlanValidationError("run_id is required")
        return PlanState.new(
            objective=objective,
            origin_mode=ensure_run_mode(origin_mode),
            run_id=run_id,
        ).to_dict()

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

    def complete_current(
        self,
        *,
        run_id: str,
        source: str = "run_finalized",
    ) -> dict[str, Any] | None:
        current = load_plan_state(self.current())
        if current is None or current.status != "active":
            return current.to_dict() if current is not None else None
        if current.owner_run_id != run_id:
            raise PlanValidationError("plan belongs to a different run")
        return self.save(current.complete(source=source))


def validate_plan_state_payload(raw: object) -> dict[str, Any]:
    plan = load_plan_state(raw)
    if plan is None:
        raise PlanValidationError("plan state cannot be None")
    return plan.to_dict()


__all__ = ["PlanStateStore", "PlanValidationError", "validate_plan_state_payload"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
