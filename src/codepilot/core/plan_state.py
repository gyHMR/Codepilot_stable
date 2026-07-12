from __future__ import annotations

from typing import Any, Mapping

from .plan import PlanState, PlanValidationError, load_plan_state


class PlanStateManager:
    """Hold the current Core plan projection for one live Runtime session."""

    def __init__(self) -> None:
        self._current: dict[str, Any] | None = None

    def current(self) -> dict[str, Any] | None:
        return dict(self._current) if self._current is not None else None

    def save(self, state: Mapping[str, Any] | PlanState) -> dict[str, Any]:
        plan = load_plan_state(state)
        if plan is None:
            raise PlanValidationError("plan state cannot be None")
        self._current = plan.to_dict()
        return dict(self._current)

    def approve_current(self, *, run_id: str | None = None) -> dict[str, Any] | None:
        _ = run_id
        current = load_plan_state(self._current)
        if current is None or current.status != "proposed":
            return current.to_dict() if current is not None else None
        return self.save(current.approve())

    def reject_current(self, *, run_id: str | None = None) -> dict[str, Any] | None:
        _ = run_id
        current = load_plan_state(self._current)
        if current is None:
            return None
        return self.save(current.reject()) if current.status == "proposed" else current.to_dict()

    def abandon_current(self, *, run_id: str | None = None, source: str = "user_abandoned") -> dict[str, Any] | None:
        _ = run_id
        current = load_plan_state(self._current)
        if current is None:
            return None
        if current.status == "abandoned":
            return current.to_dict()
        return self.save(current.abandon(source=source))


__all__ = ["PlanStateManager"]
