from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from codepilot.observability.events import normalize_event_value
from codepilot.observability.redact import redact_artifact
from codepilot.protocols import AgentRunResult, Message, UserMessage

from .contracts import (
    ComponentCheckpoint,
    MessageCursor,
    MessageRecord,
    ModelRef,
    RecoveryBundle,
    RecoveryIssue,
    RecoveryRequest,
    RecoveryResult,
    RunCheckpoint,
    RunPhase,
    RunState,
    SessionState,
    SessionStateConflictError,
    SessionStateValidationError,
    WaitingState,
    WorkspaceCheckpoint,
    WorkspaceEffectsSnapshot,
    WorkspaceRecoveryState,
)
from .repository import FileSessionRepository
from .serde import run_state_to_dict
from .workspace import validate_workspace_checkpoint


@dataclass(frozen=True)
class CreateSessionRequest:
    workspace_root: str
    model: ModelRef
    current_mode: str
    system_prompt_hash: str
    session_id: str | None = None
    parent_session_id: str | None = None
    parent_run_id: str | None = None
    session_kind: Literal["primary", "subagent"] = "primary"


@dataclass(frozen=True)
class BeginRunRequest:
    session_id: str
    user_message: UserMessage
    initial_core_state: dict[str, object] = field(default_factory=dict)
    workspace: WorkspaceCheckpoint | None = None
    run_id: str | None = None
    message_id: str | None = None


@dataclass(frozen=True)
class BeginRunResult:
    session: SessionState
    run: RunState
    message: MessageRecord


@dataclass(frozen=True)
class CommitBoundaryRequest:
    session_id: str
    run_id: str
    status: Literal["running", "waiting"]
    phase: RunPhase
    resume_point: str
    core_state: dict[str, object]
    new_messages: tuple[Message, ...] = ()
    waiting: WaitingState | None = None
    components: tuple[ComponentCheckpoint, ...] = ()
    workspace: WorkspaceCheckpoint | None = None


@dataclass(frozen=True)
class CommitBoundaryResult:
    session: SessionState
    run: RunState
    committed_messages: tuple[MessageRecord, ...]


@dataclass(frozen=True)
class ResumeRunRequest:
    session_id: str
    run_id: str
    checkpoint_id: str
    request_id: str | None = None


@dataclass(frozen=True)
class FinishRunRequest:
    session_id: str
    run_id: str
    status: Literal["completed", "failed", "cancelled"]
    stop_reason: str
    result: AgentRunResult | None = None
    final_messages: tuple[Message, ...] = ()
    workspace_effects: WorkspaceEffectsSnapshot | None = None


@dataclass(frozen=True)
class FinishRunResult:
    session: SessionState
    run: RunState
    committed_messages: tuple[MessageRecord, ...]


class SessionStateService:
    """Commit Session and Run state through a small set of stable operations."""

    def __init__(
        self,
        workspace_dir: str | Path,
        *,
        repository: FileSessionRepository | None = None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.repository = repository or FileSessionRepository(self.workspace_dir)

    def create_session(self, request: CreateSessionRequest) -> SessionState:
        now = _utc_now_iso()
        state = SessionState(
            session_id=request.session_id or _new_id("session"),
            workspace_root=str(Path(request.workspace_root).resolve()).replace("\\", "/"),
            model=request.model,
            parent_session_id=request.parent_session_id,
            parent_run_id=request.parent_run_id,
            session_kind=request.session_kind,
            current_mode=request.current_mode,
            system_prompt_hash=request.system_prompt_hash,
            created_at=now,
            updated_at=now,
        )
        return self.repository.create_session(state)

    def begin_run(
        self,
        request: BeginRunRequest,
        *,
        expected_session_revision: int,
    ) -> BeginRunResult:
        session = self._require_session(request.session_id)
        self._require_revision(session.revision, expected_session_revision, "Session")
        if session.current_run_id is not None:
            raise SessionStateConflictError(
                f"Session already has an active run: {session.current_run_id}"
            )

        now = _utc_now_iso()
        run_id = request.run_id or _new_id("run")
        message_id = request.message_id or _new_id("msg")
        message = MessageRecord(
            message_id=message_id,
            session_id=session.session_id,
            run_id=run_id,
            parent_id=session.leaf_message_id,
            created_at=now,
            message=request.user_message,
        )
        self.repository.append_message(message)
        checkpoint = RunCheckpoint(
            checkpoint_id=_new_id("checkpoint"),
            resume_point="before_model",
            message_cursor=MessageCursor(message_id),
            workspace=request.workspace,
            created_at=now,
        )
        run = RunState(
            run_id=run_id,
            session_id=session.session_id,
            status="created",
            phase="received",
            input_message_id=message_id,
            latest_message_id=message_id,
            core_state=request.initial_core_state,
            checkpoint=checkpoint,
            created_at=now,
            updated_at=now,
        )
        self.repository.create_run(run)
        updated_session = replace(
            session,
            current_run_id=run_id,
            last_run_id=run_id,
            leaf_message_id=message_id,
            updated_at=now,
            revision=session.revision + 1,
        )
        self.repository.update_session(
            updated_session,
            expected_revision=expected_session_revision,
        )
        self._append_event("run_created", updated_session, run)
        return BeginRunResult(session=updated_session, run=run, message=message)

    def commit_boundary(
        self,
        request: CommitBoundaryRequest,
        *,
        expected_run_revision: int,
        expected_session_revision: int,
    ) -> CommitBoundaryResult:
        session, run = self._require_current_run(request.session_id, request.run_id)
        self._require_revision(run.revision, expected_run_revision, "Run")
        self._require_revision(session.revision, expected_session_revision, "Session")
        if run.checkpoint is None:
            raise SessionStateValidationError("Active run has no checkpoint")

        now = _utc_now_iso()
        parent_id = run.checkpoint.message_cursor.leaf_message_id
        committed: list[MessageRecord] = []
        for message_value in request.new_messages:
            record = MessageRecord(
                message_id=_new_id("msg"),
                session_id=session.session_id,
                run_id=run.run_id,
                parent_id=parent_id,
                created_at=now,
                message=message_value,
            )
            self.repository.append_message(record)
            committed.append(record)
            parent_id = record.message_id

        checkpoint = RunCheckpoint(
            checkpoint_id=_new_id("checkpoint"),
            resume_point=request.resume_point,  # type: ignore[arg-type]
            message_cursor=MessageCursor(parent_id),
            waiting=request.waiting,
            components=request.components,
            workspace=request.workspace,
            created_at=now,
        )
        updated_run = replace(
            run,
            status=request.status,
            phase=request.phase,
            latest_message_id=parent_id,
            core_state=request.core_state,
            checkpoint=checkpoint,
            updated_at=now,
            started_at=run.started_at or now,
            revision=run.revision + 1,
        )
        self.repository.update_run(updated_run, expected_revision=expected_run_revision)

        updated_session = session
        if parent_id != session.leaf_message_id:
            updated_session = replace(
                session,
                leaf_message_id=parent_id,
                updated_at=now,
                revision=session.revision + 1,
            )
            self.repository.update_session(
                updated_session,
                expected_revision=expected_session_revision,
            )
        self._append_event("checkpoint_committed", updated_session, updated_run)
        return CommitBoundaryResult(
            session=updated_session,
            run=updated_run,
            committed_messages=tuple(committed),
        )

    def resume_run(
        self,
        request: ResumeRunRequest,
        *,
        expected_run_revision: int,
    ) -> RunState:
        _, run = self._require_current_run(request.session_id, request.run_id)
        self._require_revision(run.revision, expected_run_revision, "Run")
        if run.status != "waiting" or run.checkpoint is None or run.checkpoint.waiting is None:
            raise SessionStateValidationError("Only waiting runs can be resumed")
        if run.checkpoint.checkpoint_id != request.checkpoint_id:
            raise SessionStateConflictError("Checkpoint changed before resume")
        if request.request_id is not None and run.checkpoint.waiting.request_id != request.request_id:
            raise SessionStateConflictError("Waiting request changed before resume")

        now = _utc_now_iso()
        checkpoint = replace(run.checkpoint, waiting=None)
        updated = replace(
            run,
            status="running",
            checkpoint=checkpoint,
            resume_count=run.resume_count + 1,
            updated_at=now,
            started_at=run.started_at or now,
            revision=run.revision + 1,
        )
        self.repository.update_run(updated, expected_revision=expected_run_revision)
        session = self._require_session(request.session_id)
        self._append_event("run_resumed", session, updated)
        return updated

    def finish_run(
        self,
        request: FinishRunRequest,
        *,
        expected_run_revision: int,
        expected_session_revision: int,
    ) -> FinishRunResult:
        session, run = self._require_current_run(request.session_id, request.run_id)
        self._require_revision(run.revision, expected_run_revision, "Run")
        self._require_revision(session.revision, expected_session_revision, "Session")
        if run.checkpoint is None:
            raise SessionStateValidationError("Active run has no checkpoint")

        now = _utc_now_iso()
        parent_id = run.checkpoint.message_cursor.leaf_message_id
        committed: list[MessageRecord] = []
        for message_value in request.final_messages:
            record = MessageRecord(
                message_id=_new_id("msg"),
                session_id=session.session_id,
                run_id=run.run_id,
                parent_id=parent_id,
                created_at=now,
                message=message_value,
            )
            self.repository.append_message(record)
            committed.append(record)
            parent_id = record.message_id

        result_ref: str | None = None
        if request.result is not None:
            result_payload = normalize_event_value(request.result)
            if not isinstance(result_payload, dict):
                raise SessionStateValidationError("AgentRunResult must serialize to an object")
            result_ref = self.repository.write_result_artifact(run.run_id, result_payload)

        updated_run = replace(
            run,
            status=request.status,
            phase="finished",
            stop_reason=request.stop_reason,
            latest_message_id=parent_id,
            checkpoint=None,
            result_ref=result_ref,
            workspace_effects=request.workspace_effects or run.workspace_effects,
            updated_at=now,
            ended_at=now,
            revision=run.revision + 1,
        )
        self.repository.update_run(updated_run, expected_revision=expected_run_revision)
        updated_session = replace(
            session,
            current_run_id=None,
            last_run_id=run.run_id,
            leaf_message_id=parent_id,
            updated_at=now,
            revision=session.revision + 1,
        )
        self.repository.update_session(
            updated_session,
            expected_revision=expected_session_revision,
        )
        self._append_event(f"run_{request.status}", updated_session, updated_run)
        return FinishRunResult(
            session=updated_session,
            run=updated_run,
            committed_messages=tuple(committed),
        )

    def inspect_recovery(self, request: RecoveryRequest) -> RecoveryResult:
        session = self.repository.load_session(request.session_id)
        if session is None:
            return RecoveryResult(
                status="not_found",
                issues=(RecoveryIssue("session.not_found", "Session not found"),),
            )
        run_id = request.run_id or session.current_run_id
        if run_id is None:
            return RecoveryResult(
                status="not_found",
                issues=(RecoveryIssue("run.not_found", "No active run"),),
            )
        run = self.repository.load_run(run_id)
        if run is None or run.session_id != session.session_id:
            return RecoveryResult(
                status="blocked",
                issues=(RecoveryIssue("run.invalid", "Run is missing or belongs to another session"),),
            )
        if run.status in {"completed", "failed", "cancelled"} or run.checkpoint is None:
            return RecoveryResult(
                status="blocked",
                issues=(RecoveryIssue("run.terminal", "Run is not recoverable"),),
            )
        if (
            request.expected_checkpoint_id is not None
            and run.checkpoint.checkpoint_id != request.expected_checkpoint_id
        ):
            return RecoveryResult(
                status="blocked",
                issues=(RecoveryIssue("checkpoint.changed", "Checkpoint no longer matches"),),
            )
        waiting_kind = run.checkpoint.waiting.kind if run.checkpoint.waiting is not None else None
        if request.expected_waiting_kind is not None and waiting_kind != request.expected_waiting_kind:
            return RecoveryResult(
                status="blocked",
                issues=(RecoveryIssue("waiting.changed", "Waiting state no longer matches"),),
            )
        try:
            messages = self.repository.load_message_chain(
                session.session_id,
                leaf_id=run.checkpoint.message_cursor.leaf_message_id,
            )
        except ValueError as exc:
            return RecoveryResult(
                status="blocked",
                issues=(RecoveryIssue("messages.invalid", str(exc)),),
            )
        workspace_status = validate_workspace_checkpoint(
            self.workspace_dir,
            run.checkpoint.workspace,
        )
        needs_validation = bool(run.checkpoint.components) or workspace_status.status != "unchanged"
        bundle = RecoveryBundle(
            session=session,
            run=run,
            messages=messages,
            workspace_status=workspace_status,
        )
        return RecoveryResult(
            status="needs_validation" if needs_validation else "ready",
            bundle=bundle,
        )

    def get_session(self, session_id: str) -> SessionState | None:
        return self.repository.load_session(session_id)

    def list_sessions(self) -> tuple[SessionState, ...]:
        return self.repository.list_sessions()

    def delete_session(self, session_id: str) -> bool:
        return self.repository.delete_session(session_id)

    def update_session_mode(
        self,
        session_id: str,
        mode: str,
        *,
        expected_revision: int,
    ) -> SessionState:
        session = self._require_session(session_id)
        self._require_revision(session.revision, expected_revision, "Session")
        updated = replace(
            session,
            current_mode=mode,
            updated_at=_utc_now_iso(),
            revision=session.revision + 1,
        )
        return self.repository.update_session(updated, expected_revision=expected_revision)

    def get_run(self, run_id: str) -> RunState | None:
        return self.repository.load_run(run_id)

    def load_messages(
        self,
        session_id: str,
        *,
        leaf_id: str | None = None,
    ) -> tuple[MessageRecord, ...]:
        return self.repository.load_message_chain(session_id, leaf_id=leaf_id)

    def list_entry_ids(self, session_id: str) -> list[str]:
        return [record.message_id for record in self.repository.load_message_records(session_id)]

    def list_entries(self, session_id: str) -> list[dict[str, Any]]:
        session = self._require_session(session_id)
        return [
            {
                "id": record.message_id,
                "parent_id": record.parent_id,
                "timestamp": record.created_at,
                "role": getattr(record.message, "role", "unknown"),
                "preview": str(record.message)[:160],
                "is_leaf": record.message_id == session.leaf_message_id,
            }
            for record in self.repository.load_message_records(session_id)
        ]

    def get_entry_path(self, session_id: str, entry_id: str) -> list[str]:
        return [
            record.message_id
            for record in self.repository.load_message_chain(session_id, leaf_id=entry_id)
        ]

    def get_session_tree(self, session_id: str) -> list[dict[str, Any]]:
        by_id = {
            item["id"]: {**item, "children": []}
            for item in self.list_entries(session_id)
        }
        roots: list[dict[str, Any]] = []
        for item in by_id.values():
            parent = item.get("parent_id")
            if isinstance(parent, str) and parent in by_id:
                by_id[parent]["children"].append(item)
            else:
                roots.append(item)
        return roots

    def append_message(
        self,
        session_id: str,
        message: Message,
        *,
        run_id: str | None = None,
    ) -> MessageRecord:
        session = self._require_session(session_id)
        now = _utc_now_iso()
        record = MessageRecord(
            message_id=_new_id("msg"),
            session_id=session_id,
            run_id=run_id,
            parent_id=session.leaf_message_id,
            created_at=now,
            message=message,
        )
        self.repository.append_message(record)
        self.repository.update_session(
            replace(
                session,
                leaf_message_id=record.message_id,
                updated_at=now,
                revision=session.revision + 1,
            ),
            expected_revision=session.revision,
        )
        return record

    def set_leaf(self, session_id: str, entry_id: str) -> SessionState:
        self.repository.load_message_chain(session_id, leaf_id=entry_id)
        session = self._require_session(session_id)
        updated = replace(
            session,
            leaf_message_id=entry_id,
            updated_at=_utc_now_iso(),
            revision=session.revision + 1,
        )
        return self.repository.update_session(updated, expected_revision=session.revision)

    def fork_session(
        self,
        session_id: str,
        new_session_id: str,
        *,
        from_entry_id: str | None = None,
    ) -> SessionState:
        source = self._require_session(session_id)
        chain = self.repository.load_message_chain(session_id, leaf_id=from_entry_id)
        now = _utc_now_iso()
        target = replace(
            source,
            session_id=new_session_id,
            parent_session_id=session_id,
            current_run_id=None,
            last_run_id=None,
            leaf_message_id=chain[-1].message_id if chain else None,
            created_at=now,
            updated_at=now,
            revision=1,
        )
        self.repository.create_session(target)
        for record in chain:
            self.repository.append_message(
                replace(record, session_id=new_session_id, run_id=None)
            )
        self.append_event(
            new_session_id,
            {"type": "session_forked", "parent_session_id": session_id},
        )
        return target

    def append_event(
        self,
        session_id: str,
        event: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        self._require_session(session_id)
        payload = redact_artifact(normalize_event_value(dict(event)))
        legacy_keys = {"eventId", "sessionId", "runId"}.intersection(payload)
        if legacy_keys:
            raise ValueError(
                "Event uses legacy field names: " + ", ".join(sorted(legacy_keys))
            )
        payload["event_id"] = str(payload.get("event_id") or _new_id("event"))
        payload["session_id"] = session_id
        effective_run_id = run_id or payload.get("run_id")
        if isinstance(effective_run_id, str) and effective_run_id:
            payload["run_id"] = effective_run_id
        else:
            payload.pop("run_id", None)
        payload.setdefault("created_at", _utc_now_iso())
        self.repository.append_event(payload)
        return payload

    def load_events(
        self,
        session_id: str,
        *,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return list(self.repository.load_events(session_id, run_id=run_id, limit=limit))

    def load_run_view(self, session_id: str, run_id: str) -> dict[str, Any]:
        run = self.repository.load_run(run_id)
        if run is None or run.session_id != session_id:
            raise FileNotFoundError(f"Run not found: {run_id}")
        payload = run_state_to_dict(run)
        rollback = self.read_rollback_metadata(run_id)
        if rollback is not None:
            payload["rollback"] = rollback
        return payload

    def load_last_run_view(self, session_id: str) -> dict[str, Any] | None:
        session = self._require_session(session_id)
        return (
            self.load_run_view(session_id, session.last_run_id)
            if session.last_run_id is not None
            else None
        )

    def write_rollback_metadata(self, run_id: str, metadata: dict[str, Any]) -> None:
        from .filesystem import atomic_write_json

        atomic_write_json(
            self.repository.layout.run_artifacts_dir(run_id) / "rollback.json",
            redact_artifact(metadata),
        )

    def read_rollback_metadata(self, run_id: str) -> dict[str, Any] | None:
        from .filesystem import read_json_object

        return read_json_object(
            self.repository.layout.run_artifacts_dir(run_id) / "rollback.json"
        )

    def _require_session(self, session_id: str) -> SessionState:
        session = self.repository.load_session(session_id)
        if session is None:
            raise FileNotFoundError(f"Session not found: {session_id}")
        return session

    def _require_current_run(self, session_id: str, run_id: str) -> tuple[SessionState, RunState]:
        session = self._require_session(session_id)
        if session.current_run_id != run_id:
            raise SessionStateConflictError(f"Run is not current for session: {run_id}")
        run = self.repository.load_run(run_id)
        if run is None or run.session_id != session_id:
            raise FileNotFoundError(f"Run not found for session: {run_id}")
        return session, run

    @staticmethod
    def _require_revision(actual: int, expected: int, name: str) -> None:
        if actual != expected:
            raise SessionStateConflictError(
                f"{name} revision conflict: expected {expected}, got {actual}"
            )

    def _append_event(self, event_type: str, session: SessionState, run: RunState) -> None:
        self.append_event(
            session.session_id,
            {
                "type": event_type,
                "session_revision": session.revision,
                "run_revision": run.revision,
            },
            run_id=run.run_id,
        )


def new_session_id() -> str:
    return _new_id("session")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "BeginRunRequest",
    "BeginRunResult",
    "CommitBoundaryRequest",
    "CommitBoundaryResult",
    "CreateSessionRequest",
    "FinishRunRequest",
    "FinishRunResult",
    "ResumeRunRequest",
    "SessionStateService",
    "new_session_id",
]
