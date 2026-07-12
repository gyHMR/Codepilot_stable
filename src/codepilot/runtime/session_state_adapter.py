from __future__ import annotations

from collections.abc import Callable

from codepilot.core.contracts import CoreRunBoundary, RunStatePort
from codepilot.sessions.contracts import (
    ComponentCheckpoint,
    RunState,
    SessionState,
    WaitingState,
    WorkspaceCheckpoint,
)
from codepilot.sessions.service import CommitBoundaryRequest, SessionStateService


class RuntimeSessionStateAdapter(RunStatePort):
    """Map Core execution boundaries to authoritative Sessions v2 commits."""

    def __init__(
        self,
        service: SessionStateService,
        session: SessionState,
        run: RunState,
        context_state: Callable[[], dict[str, object]] | None = None,
        plan_state: Callable[[], dict[str, object] | None] | None = None,
        workspace_state: Callable[[dict[str, object]], WorkspaceCheckpoint] | None = None,
    ) -> None:
        self.service = service
        self.session = session
        self.run = run
        self.context_state = context_state
        self.plan_state = plan_state
        self.workspace_state = workspace_state
        self.committed_message_ids: dict[int, str] = {}

    async def commit(self, boundary: CoreRunBoundary) -> None:
        status, phase, resume_point = _boundary_state(boundary.kind)
        waiting = (
            WaitingState(
                kind=boundary.waiting.kind,
                request_id=boundary.waiting.request_id,
                payload=boundary.waiting.payload,
            )
            if boundary.waiting is not None
            else None
        )
        components: list[ComponentCheckpoint] = []
        if boundary.tool_recovery_state is not None:
            components.append(
                ComponentCheckpoint(
                    owner="tools",
                    schema_version=1,
                    state=boundary.tool_recovery_state,
                )
            )
        if self.context_state is not None:
            components.append(
                ComponentCheckpoint(
                    owner="context",
                    schema_version=1,
                    state=self.context_state(),
                )
            )
        core_state = dict(boundary.core_state)
        if self.plan_state is not None:
            current_plan = self.plan_state()
            if current_plan is not None:
                core_state["plan_state"] = current_plan
        result = self.service.commit_boundary(
            CommitBoundaryRequest(
                session_id=self.session.session_id,
                run_id=self.run.run_id,
                status=status,
                phase=phase,
                resume_point=resume_point,
                core_state=core_state,
                new_messages=boundary.new_messages,
                waiting=waiting,
                components=tuple(components),
                workspace=(
                    self.workspace_state(core_state)
                    if self.workspace_state is not None
                    else self.run.checkpoint.workspace if self.run.checkpoint else None
                ),
            ),
            expected_run_revision=self.run.revision,
            expected_session_revision=self.session.revision,
        )
        self.session = result.session
        self.run = result.run
        for message, record in zip(boundary.new_messages, result.committed_messages, strict=True):
            self.committed_message_ids[id(message)] = record.message_id


def _boundary_state(kind: str) -> tuple[str, str, str]:
    mapping = {
        "before_model": ("running", "model", "before_model"),
        "after_model": ("running", "model", "after_model"),
        "before_tools": ("running", "tools", "before_tools"),
        "after_tools": ("running", "tools", "after_tools"),
        "waiting_tool_approval": ("waiting", "tools", "before_tools"),
        "waiting_user_input": ("waiting", "model", "after_model"),
        "waiting_plan_confirmation": ("waiting", "model", "after_model"),
        "before_finalization": ("running", "finalizing", "before_finalization"),
    }
    try:
        return mapping[kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported Core boundary: {kind}") from exc


__all__ = ["RuntimeSessionStateAdapter"]
