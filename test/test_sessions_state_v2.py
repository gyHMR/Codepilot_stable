from __future__ import annotations

from dataclasses import replace

import pytest

from codepilot.protocols import UserMessage
from codepilot.sessions.contracts import (
    ComponentCheckpoint,
    MessageCursor,
    MessageRecord,
    ModelRef,
    RecoveryBundle,
    RecoveryIssue,
    RecoveryResult,
    RunCheckpoint,
    RunState,
    SessionState,
    SessionStateValidationError,
    WaitingState,
    WorkspaceRecoveryState,
    validate_run_transition,
)


NOW = "2026-07-12T10:00:00Z"


def _checkpoint(
    *,
    resume_point: str = "before_model",
    waiting: WaitingState | None = None,
    components: tuple[ComponentCheckpoint, ...] = (),
) -> RunCheckpoint:
    return RunCheckpoint(
        checkpoint_id="checkpoint_1",
        resume_point=resume_point,  # type: ignore[arg-type]
        message_cursor=MessageCursor("message_1"),
        waiting=waiting,
        components=components,
        created_at=NOW,
    )


def _run(**updates: object) -> RunState:
    values: dict[str, object] = {
        "run_id": "run_1",
        "session_id": "session_1",
        "status": "created",
        "phase": "received",
        "input_message_id": "message_1",
        "latest_message_id": "message_1",
        "core_state": {},
        "checkpoint": _checkpoint(),
        "created_at": NOW,
        "updated_at": NOW,
        "revision": 1,
    }
    values.update(updates)
    return RunState(**values)  # type: ignore[arg-type]


def _session() -> SessionState:
    return SessionState(
        session_id="session_1",
        workspace_root="E:/workspace",
        model=ModelRef(provider="test", model="test-model"),
        current_run_id="run_1",
        last_run_id="run_1",
        leaf_message_id="message_1",
        current_mode="build",
        system_prompt_hash="sha256:prompt",
        created_at=NOW,
        updated_at=NOW,
    )


def test_session_state_accepts_primary_and_subagent_identity() -> None:
    primary = _session()
    subagent = SessionState(
        session_id="session_child",
        workspace_root="E:/workspace",
        model=ModelRef(provider="test", model="test-model"),
        parent_session_id=primary.session_id,
        parent_run_id="run_parent",
        session_kind="subagent",
        current_mode="read",
        system_prompt_hash="sha256:prompt",
        created_at=NOW,
        updated_at=NOW,
    )

    assert primary.session_kind == "primary"
    assert subagent.parent_session_id == "session_1"


def test_session_state_rejects_invalid_parent_relationships() -> None:
    with pytest.raises(SessionStateValidationError, match="Primary sessions"):
        replace(_session(), parent_run_id="run_parent")

    with pytest.raises(SessionStateValidationError, match="require parent_session_id"):
        replace(
            _session(),
            session_id="session_child",
            session_kind="subagent",
            parent_session_id=None,
            parent_run_id="run_parent",
        )


def test_non_terminal_run_requires_matching_checkpoint_cursor() -> None:
    run = _run()

    assert run.checkpoint is not None
    assert run.checkpoint.message_cursor.leaf_message_id == run.latest_message_id

    with pytest.raises(SessionStateValidationError, match="require a checkpoint"):
        _run(checkpoint=None)

    with pytest.raises(SessionStateValidationError, match="must match"):
        _run(latest_message_id="message_other")


def test_waiting_run_requires_waiting_checkpoint_state() -> None:
    waiting = WaitingState(
        kind="tool_approval",
        request_id="approval_1",
        payload={"tool_call_id": "call_1"},
    )
    run = _run(
        status="waiting",
        phase="tools",
        checkpoint=_checkpoint(resume_point="before_tools", waiting=waiting),
    )

    assert run.checkpoint is not None
    assert run.checkpoint.waiting == waiting

    with pytest.raises(SessionStateValidationError, match="require waiting"):
        _run(status="waiting")

    with pytest.raises(SessionStateValidationError, match="Only waiting"):
        _run(checkpoint=_checkpoint(waiting=waiting))


def test_run_rejects_phase_and_resume_point_mismatch() -> None:
    with pytest.raises(SessionStateValidationError, match="invalid for phase"):
        _run(
            status="running",
            phase="model",
            checkpoint=_checkpoint(resume_point="before_tools"),
        )


def test_terminal_run_requires_finished_phase_and_no_checkpoint() -> None:
    completed = _run(
        status="completed",
        phase="finished",
        checkpoint=None,
        ended_at=NOW,
        result_ref="artifact:result.json",
    )

    assert completed.status == "completed"

    with pytest.raises(SessionStateValidationError, match="phase=finished"):
        _run(status="failed", phase="model", checkpoint=None, ended_at=NOW)

    with pytest.raises(SessionStateValidationError, match="cannot retain"):
        _run(status="cancelled", phase="finished", ended_at=NOW)

    with pytest.raises(SessionStateValidationError, match="require ended_at"):
        _run(status="completed", phase="finished", checkpoint=None)


def test_checkpoint_rejects_duplicate_component_owners() -> None:
    component = ComponentCheckpoint(owner="context", schema_version=1, state={})

    with pytest.raises(SessionStateValidationError, match="owners must be unique"):
        _checkpoint(components=(component, component))


def test_component_state_must_be_json_serializable() -> None:
    with pytest.raises(SessionStateValidationError, match="JSON serializable"):
        ComponentCheckpoint(owner="tools", schema_version=1, state={"value": object()})


def test_state_models_reject_unknown_schema_versions() -> None:
    with pytest.raises(SessionStateValidationError, match="session state schema_version"):
        replace(_session(), schema_version=2)

    with pytest.raises(SessionStateValidationError, match="run state schema_version"):
        replace(_run(), schema_version=2)

    with pytest.raises(SessionStateValidationError, match="run checkpoint schema_version"):
        replace(_checkpoint(), schema_version=2)


def test_run_transition_enforces_status_and_revision() -> None:
    created = _run()
    running = replace(created, status="running", revision=2, started_at=NOW)

    validate_run_transition(created, running)

    with pytest.raises(SessionStateValidationError, match="increase by exactly one"):
        validate_run_transition(created, replace(running, revision=3))

    completed = replace(
        running,
        status="completed",
        phase="finished",
        checkpoint=None,
        ended_at=NOW,
        revision=3,
    )
    validate_run_transition(running, completed)

    with pytest.raises(SessionStateValidationError, match="Invalid run transition"):
        validate_run_transition(completed, replace(running, revision=4))


def test_recovery_result_requires_bundle_for_ready_states() -> None:
    message = MessageRecord(
        message_id="message_1",
        session_id="session_1",
        run_id="run_1",
        created_at=NOW,
        message=UserMessage(content="continue"),
    )
    bundle = RecoveryBundle(
        session=_session(),
        run=_run(),
        messages=(message,),
        workspace_status=WorkspaceRecoveryState(status="unchanged"),
    )

    assert RecoveryResult(status="ready", bundle=bundle).bundle == bundle
    assert RecoveryResult(
        status="blocked",
        issues=(RecoveryIssue(code="checkpoint.invalid", message="invalid"),),
    ).bundle is None

    with pytest.raises(SessionStateValidationError, match="requires a bundle"):
        RecoveryResult(status="ready")

    with pytest.raises(SessionStateValidationError, match="cannot include a bundle"):
        RecoveryResult(status="blocked", bundle=bundle)
