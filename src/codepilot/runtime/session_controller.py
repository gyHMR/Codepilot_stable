from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from codepilot.llm.ports import ModelDescriptor

from .commands import apply_session_command
from .coordinator import RunCoordinator
from codepilot.sessions.contracts import (
    SessionCommandIntent,
    SessionCommandRecord,
    SessionView,
)
from .session_coordinator import RuntimeSessionCoordinator


def create_session_controller(options: Any) -> "SessionController":
    return _bind_session_runtime(RuntimeSessionCoordinator(options))


def _bind_session_runtime(session: RuntimeSessionCoordinator) -> "SessionController":
    model = session.conversation.model
    return SessionController(
        session_id=session.session_id,
        model=ModelDescriptor(
            provider=getattr(model, "provider", "unknown"),
            model_id=getattr(model, "id", "unknown"),
        ),
        current_mode=session.current_mode,
        _session=session,
    )


@dataclass
class SessionController:
    session_id: str
    model: ModelDescriptor = field(
        default_factory=lambda: ModelDescriptor(provider="local", model_id="v2-test")
    )
    current_mode: str = "build"
    _session: RuntimeSessionCoordinator | None = None
    _derived_controllers: dict[str, "SessionController"] = field(default_factory=dict)
    _runs: RunCoordinator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self._session is None:
            raise ValueError("SessionController requires RuntimeSessionCoordinator")
        self._runs = RunCoordinator(self._session)

    @property
    def runs(self) -> RunCoordinator:
        """The single runtime entry for preparing, executing, and committing Runs."""

        return self._runs

    def describe(self) -> SessionView:
        return self._session.describe(last_run_id=self._session.session_state.last_run_id)

    async def apply_command(self, intent: SessionCommandIntent) -> SessionCommandRecord:
        record = await apply_session_command(
            self.session_id,
            intent,
            session=self._session,
            controller=self,
        )
        if "current_mode" in record.data:
            self.current_mode = str(record.data["current_mode"])
        return record

    def runtime_checkpoint(self) -> dict[str, Any] | None:
        return self._session.runtime_checkpoint()

    def component_checkpoint_state(self, owner: str) -> dict[str, object] | None:
        return self._session.component_checkpoint_state(owner)

    def current_plan_state(self) -> dict[str, Any] | None:
        return self._session.plan_state.current()

    def save_plan_state(self, state: Any) -> dict[str, Any]:
        return self._session.plan_state.save(state)

    def stage_derived_session(self, session: RuntimeSessionCoordinator) -> None:
        self._derived_controllers[session.session_id] = _bind_session_runtime(session)

    def claim_derived_controller(self, session_id: str) -> "SessionController" | None:
        return self._derived_controllers.pop(session_id, None)

    def close(self) -> None:
        self._session.close()

__all__ = ["SessionController", "create_session_controller", "_bind_session_runtime"]
