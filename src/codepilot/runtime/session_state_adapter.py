from __future__ import annotations

import asyncio
from collections.abc import Callable

from codepilot.core.contracts import CoreBoundary
from codepilot.runtime.contracts import project_core_domain_event
from codepilot.sessions.contracts import (
    ComponentCheckpoint,
    RunState,
    SessionState,
    WaitingState,
    WorkspaceCheckpoint,
)
from codepilot.sessions.service import CommitRunBoundaryRequest, SessionStateService


class RuntimeSessionStateAdapter:
    """Map Core execution boundaries to authoritative Sessions v2 commits."""

    def __init__(
        self,
        service: SessionStateService,
        session: SessionState,
        run: RunState,
        tool_state: Callable[[], dict[str, object] | None] | None = None,
        context_state: Callable[[], dict[str, object]] | None = None,
        workspace_state: (
            Callable[[dict[str, object]], WorkspaceCheckpoint] | None
        ) = None,
    ) -> None:
        self.service = service
        self.session = session
        self.run = run
        self.tool_state = tool_state
        self.context_state = context_state
        self.workspace_state = workspace_state
        self.committed_message_ids: dict[int, str] = {}
        self._pending_events: list[dict[str, object]] = []

    def bind_tool_state(
        self,
        tool_state: Callable[[], dict[str, object] | None] | None,
    ) -> None:
        """Bind the Runtime-owned Tool checkpoint reader for target boundaries."""

        self.tool_state = tool_state

    def queue_durable_event(self, event: dict[str, object]) -> None:
        """Queue a run event until the next authoritative boundary commit."""

        payload = dict(event)
        payload.setdefault(
            "event_id",
            f"{self.run.run_id}:runtime:{self.run.revision}:{len(self._pending_events) + 1}",
        )
        payload.setdefault("run_id", self.run.run_id)
        payload.setdefault("session_id", self.session.session_id)
        self._pending_events.append(payload)

    async def commit(self, boundary: CoreBoundary) -> None:
        _, phase, resume_point = _target_boundary_state(boundary)
        wait = boundary.wait
        core_state = boundary.state.to_dict()
        new_messages = boundary.new_messages
        durable_events = tuple(
            project_core_domain_event(
                event,
                event_id=(
                    f"{self.run.run_id}:core:{self.run.revision}:"
                    f"{index + len(self._pending_events)}"
                ),
                run_id=self.run.run_id,
                session_id=self.session.session_id,
            )
            for index, event in enumerate(boundary.domain_events, start=1)
        )
        tool_checkpoint = self.tool_state() if self.tool_state is not None else None
        commit_kind = "waiting" if boundary.kind == "waiting" else "progress"
        waiting = (
            WaitingState(
                kind=wait.kind,
                request_id=wait.request_id,
                payload=dict(wait.payload),
            )
            if wait is not None
            else None
        )
        components: list[ComponentCheckpoint] = []
        if tool_checkpoint is not None:
            components.append(
                ComponentCheckpoint(
                    owner="tools",
                    schema_version=1,
                    state=tool_checkpoint,
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
        existing_owners = {component.owner for component in components}
        if self.run.checkpoint is not None:
            components.extend(
                component
                for component in self.run.checkpoint.components
                if component.owner == "rollback"
                and component.owner not in existing_owners
            )
        request = CommitRunBoundaryRequest(
            commit_id=f"{self.run.run_id}:{commit_kind}:{self.run.revision}",
            kind=commit_kind,
            session_id=self.session.session_id,
            run_id=self.run.run_id,
            expected_run_revision=self.run.revision,
            expected_session_revision=self.session.revision,
            phase=phase,
            resume_point=resume_point,
            core_state=core_state,
            new_messages=new_messages,
            durable_events=tuple([*self._pending_events, *durable_events]),
            waiting=waiting,
            components=tuple(components),
            workspace=(
                self.workspace_state(core_state)
                if self.workspace_state is not None
                else self.run.checkpoint.workspace if self.run.checkpoint else None
            ),
        )
        try:
            result = self.service.commit_run_boundary(request)
        except OSError:
            # The first attempt may have replaced run.json before an event append
            # failed.  Reusing the same commit_id lets Sessions return its receipt
            # without creating a second message/checkpoint/result.
            await asyncio.sleep(0)
            result = self.service.commit_run_boundary(request)
        self.session = result.session
        self.run = result.run
        self._pending_events.clear()
        for message, record in zip(
            new_messages, result.committed_messages, strict=True
        ):
            self.committed_message_ids[id(message)] = record.message_id


def _target_boundary_state(boundary: CoreBoundary) -> tuple[str, str, str]:
    if boundary.kind == "waiting":
        if boundary.wait is None:  # pragma: no cover - CoreBoundary validates this
            raise ValueError("Waiting Core boundary requires wait")
        if boundary.wait.kind == "tool_approval":
            return "waiting", "tools", "before_tools"
        return "waiting", "model", "after_model"
    mapping = {
        "before_model": ("running", "model", "before_model"),
        "after_model": ("running", "model", "after_model"),
        "before_tools": ("running", "tools", "before_tools"),
        "after_tools": ("running", "tools", "after_tools"),
        "before_terminal": ("running", "finalizing", "before_finalization"),
    }
    try:
        return mapping[boundary.kind]
    except KeyError as exc:  # pragma: no cover - CoreBoundary validates this
        raise ValueError(f"Unsupported Core boundary: {boundary.kind}") from exc


__all__ = ["RuntimeSessionStateAdapter"]
