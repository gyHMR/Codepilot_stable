from __future__ import annotations

"""Session-owned persistence for soft PlanState."""

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
        return self.save(
            PlanState.new(
                objective=objective,
                origin_mode=ensure_run_mode(origin_mode),
                run_id=run_id,
            )
        )

    def approve_current(self, *, run_id: str | None = None) -> dict[str, Any] | None:
        current = load_plan_state(self.current())
        if current is None or current.status != "proposed":
            return current.to_dict() if current is not None else None
        return self.save(
            PlanState(
                plan_id=current.plan_id,
                status="active",
                approval_state="approved",
                origin_mode=current.origin_mode,
                objective=current.objective,
                items=current.items,
                explanation=current.explanation,
                created_at=current.created_at,
                updated_at=_utc_now_iso(),
                last_update_run_id=run_id or current.last_update_run_id,
            )
        )

    def reject_current(self, *, run_id: str | None = None) -> dict[str, Any] | None:
        current = load_plan_state(self.current())
        if current is None:
            return None
        if current.status != "proposed":
            return current.to_dict()
        return self.save(
            PlanState(
                plan_id=current.plan_id,
                status="rejected",
                approval_state="rejected",
                origin_mode=current.origin_mode,
                objective=current.objective,
                items=current.items,
                explanation=current.explanation,
                created_at=current.created_at,
                updated_at=_utc_now_iso(),
                last_update_run_id=run_id or current.last_update_run_id,
            )
        )

    def abandon_current(self, *, run_id: str | None = None) -> dict[str, Any] | None:
        current = load_plan_state(self.current())
        if current is None:
            return None
        if current.status == "abandoned":
            return current.to_dict()
        return self.save(
            PlanState(
                plan_id=current.plan_id,
                status="abandoned",
                approval_state=current.approval_state,
                origin_mode=current.origin_mode,
                objective=current.objective,
                items=current.items,
                explanation=current.explanation,
                created_at=current.created_at,
                updated_at=_utc_now_iso(),
                last_update_run_id=run_id or current.last_update_run_id,
            )
        )


def validate_plan_state_payload(raw: object) -> dict[str, Any]:
    plan = load_plan_state(raw)
    if plan is None:
        raise PlanValidationError("plan state cannot be None")
    return plan.to_dict()


__all__ = ["PlanStateStore", "PlanValidationError", "validate_plan_state_payload"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
