from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from codepilot.protocols import AgentRunResult, AssistantMessage, TextContent, UserMessage
from codepilot.sessions.contracts import (
    MessageRecord,
    ModelRef,
    RecoveryRequest,
    SessionStateConflictError,
    WaitingState,
)
from codepilot.sessions.filesystem import atomic_write_json, read_json_object, read_jsonl
from codepilot.sessions.repository import FileSessionRepository
from codepilot.sessions.service import (
    BeginRunRequest,
    CommitRunBoundaryRequest,
    CreateSessionRequest,
    ResumeRunRequest,
    SessionStateService,
)
from codepilot.sessions.workspace import capture_workspace_checkpoint


def _service(tmp_path: Path) -> tuple[SessionStateService, object]:
    service = SessionStateService(tmp_path)
    session = service.create_session(
        CreateSessionRequest(
            session_id="session_1",
            workspace_root=str(tmp_path),
            model=ModelRef(provider="test", model="model"),
            current_mode="build",
            system_prompt_hash="sha256:prompt",
        )
    )
    return service, session


def _begin(service: SessionStateService, session_revision: int):
    return service.begin_run(
        BeginRunRequest(
            session_id="session_1",
            request_id="request_1",
            run_id="run_1",
            message_id="message_user",
            user_message=UserMessage(content="inspect repository"),
            initial_core_state={"turn": 0},
            workspace=capture_workspace_checkpoint(service.workspace_dir),
        ),
        expected_session_revision=session_revision,
    )


def test_begin_run_retry_repairs_partial_admission_without_duplicate_message(
    tmp_path: Path,
) -> None:
    service, session = _service(tmp_path)
    request = BeginRunRequest(
        session_id="session_1",
        request_id="request_retry",
        run_id="run_retry",
        message_id="message_retry",
        user_message=UserMessage(content="retry the same prompt"),
    )
    original_update = service.repository.update_session
    failed_once = False

    def fail_after_run_created(state, *, expected_revision):
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise OSError("simulated crash before session update")
        return original_update(state, expected_revision=expected_revision)

    service.repository.update_session = fail_after_run_created  # type: ignore[method-assign]
    with pytest.raises(OSError, match="simulated crash"):
        service.begin_run(request, expected_session_revision=session.revision)

    reopened = SessionStateService(tmp_path)
    repaired = reopened.begin_run(request, expected_session_revision=session.revision)

    assert repaired.reused is True
    assert repaired.session.current_run_id == "run_retry"
    assert repaired.run.request_id == "request_retry"
    assert len(reopened.load_messages("session_1")) == 1


def test_begin_run_rejects_request_id_reuse_with_different_input(tmp_path: Path) -> None:
    service, session = _service(tmp_path)
    first = BeginRunRequest(
        session_id="session_1",
        request_id="request_same",
        user_message=UserMessage(content="first"),
    )
    service.begin_run(first, expected_session_revision=session.revision)

    with pytest.raises(SessionStateConflictError, match="request_id"):
        service.begin_run(
            BeginRunRequest(
                session_id="session_1",
                request_id="request_same",
                user_message=UserMessage(content="different"),
            ),
            expected_session_revision=session.revision + 1,
        )


def test_begin_run_blocks_a_new_request_when_an_orphan_active_run_exists(tmp_path: Path) -> None:
    service, session = _service(tmp_path)
    original_update = service.repository.update_session

    def fail_session_update(_state, *, expected_revision):
        raise OSError(f"simulated crash at revision {expected_revision}")

    service.repository.update_session = fail_session_update  # type: ignore[method-assign]
    with pytest.raises(OSError, match="simulated crash"):
        service.begin_run(
            BeginRunRequest(
                session_id="session_1",
                request_id="request_orphan",
                user_message=UserMessage(content="orphan"),
            ),
            expected_session_revision=session.revision,
        )
    service.repository.update_session = original_update  # type: ignore[method-assign]

    with pytest.raises(SessionStateConflictError, match="orphan active run"):
        service.begin_run(
            BeginRunRequest(
                session_id="session_1",
                request_id="request_new",
                user_message=UserMessage(content="new prompt"),
            ),
            expected_session_revision=session.revision,
        )


def _start(service: SessionStateService, begun):
    return service.commit_run_boundary(
        CommitRunBoundaryRequest(
            commit_id=f"start:{begun.run.revision}",
            kind="progress",
            session_id="session_1",
            run_id="run_1",
            expected_run_revision=begun.run.revision,
            expected_session_revision=begun.session.revision,
            phase="model",
            resume_point="before_model",
            core_state={"turn": 0},
            workspace=begun.run.checkpoint.workspace,
        )
    )


def test_begin_run_creates_authoritative_session_run_and_message_files(tmp_path: Path) -> None:
    service, session = _service(tmp_path)
    begun = _begin(service, session.revision)

    root = tmp_path / ".codepilot"
    assert (root / "sessions" / "session_1" / "session.json").is_file()
    assert (root / "sessions" / "session_1" / "messages.jsonl").is_file()
    assert (root / "runs" / "run_1" / "run.json").is_file()
    assert begun.session.current_run_id == "run_1"
    assert begun.run.status == "created"
    assert begun.run.checkpoint is not None
    assert begun.run.checkpoint.resume_point == "before_model"
    assert service.load_messages("session_1")[-1].message_id == "message_user"


def test_repository_message_append_is_idempotent_and_rejects_conflicts(tmp_path: Path) -> None:
    service, session = _service(tmp_path)
    _begin(service, session.revision)
    repository = FileSessionRepository(tmp_path)
    record = repository.load_message_records("session_1")[0]

    assert repository.append_message(record) == record

    with pytest.raises(SessionStateConflictError, match="Message id already exists"):
        repository.append_message(
            replace(record, message=UserMessage(content="different content"))
        )


def test_commit_run_boundary_updates_run_before_session_navigation(tmp_path: Path) -> None:
    service, session = _service(tmp_path)
    begun = _begin(service, session.revision)
    committed = service.commit_run_boundary(
        CommitRunBoundaryRequest(
            commit_id=f"after-model:{begun.run.revision}",
            kind="progress",
            session_id="session_1",
            run_id="run_1",
            expected_run_revision=begun.run.revision,
            expected_session_revision=begun.session.revision,
            phase="model",
            resume_point="after_model",
            core_state={"turn": 1},
            workspace=begun.run.checkpoint.workspace,
            new_messages=(
                AssistantMessage(content=[TextContent(text="I found the entry point.")]),
            ),
        )
    )

    assert committed.run.revision == 2
    assert committed.session.revision == 3
    assert committed.run.latest_message_id == committed.session.leaf_message_id
    assert committed.run.checkpoint is not None
    assert committed.run.checkpoint.resume_point == "after_model"
    assert len(committed.committed_messages) == 1


def test_commit_run_boundary_retry_is_idempotent(tmp_path: Path) -> None:
    service, session = _service(tmp_path)
    begun = _begin(service, session.revision)
    request = CommitRunBoundaryRequest(
        commit_id="retry-progress",
        kind="progress",
        session_id="session_1",
        run_id="run_1",
        expected_run_revision=begun.run.revision,
        expected_session_revision=begun.session.revision,
        phase="model",
        resume_point="after_model",
        core_state={"turn": 1},
        workspace=begun.run.checkpoint.workspace,
        new_messages=(AssistantMessage(content=[TextContent(text="once")]),),
        durable_events=(
            {
                "event_id": "retry-event",
                "type": "agent_end",
            },
        ),
    )
    first = service.commit_run_boundary(request)
    message_count = len(service.load_messages("session_1"))
    event_count = len(service.load_events("session_1"))

    second = service.commit_run_boundary(request)

    assert second.run == first.run
    assert second.session == first.session
    assert second.committed_messages == first.committed_messages
    assert len(service.load_messages("session_1")) == message_count
    assert len(service.load_events("session_1")) == event_count


def test_waiting_run_can_resume_only_with_current_checkpoint_and_request(tmp_path: Path) -> None:
    service, session = _service(tmp_path)
    begun = _begin(service, session.revision)
    started = _start(service, begun)
    waiting = service.commit_run_boundary(
        CommitRunBoundaryRequest(
            commit_id=f"waiting:{started.run.revision}",
            kind="waiting",
            session_id="session_1",
            run_id="run_1",
            expected_run_revision=started.run.revision,
            expected_session_revision=started.session.revision,
            phase="tools",
            resume_point="before_tools",
            core_state={"turn": 1},
            waiting=WaitingState(
                kind="tool_approval",
                request_id="approval_1",
                payload={"tool_call_id": "call_1"},
            ),
            workspace=started.run.checkpoint.workspace,
        )
    )
    checkpoint = waiting.run.checkpoint
    assert checkpoint is not None

    with pytest.raises(SessionStateConflictError, match="Waiting request changed"):
        service.resume_run(
            ResumeRunRequest(
                session_id="session_1",
                run_id="run_1",
                checkpoint_id=checkpoint.checkpoint_id,
                request_id="approval_old",
            ),
            expected_run_revision=waiting.run.revision,
        )

    resumed = service.resume_run(
        ResumeRunRequest(
            session_id="session_1",
            run_id="run_1",
            checkpoint_id=checkpoint.checkpoint_id,
            request_id="approval_1",
        ),
        expected_run_revision=waiting.run.revision,
    )
    assert resumed.status == "running"
    assert resumed.resume_count == 1
    assert resumed.checkpoint is not None and resumed.checkpoint.waiting is None


def test_terminal_commit_commits_result_artifact_and_clears_current_run(tmp_path: Path) -> None:
    service, session = _service(tmp_path)
    begun = _begin(service, session.revision)
    started = _start(service, begun)
    result = AgentRunResult(
        run_id="run_1",
        session_id="session_1",
        status="completed",
        stop_reason="final_answer",
    )
    request = CommitRunBoundaryRequest(
        commit_id=f"terminal:{started.run.revision}",
        kind="terminal",
        session_id="session_1",
        run_id="run_1",
        expected_run_revision=started.run.revision,
        expected_session_revision=started.session.revision,
        stop_reason="final_answer",
        terminal_status="completed",
        result=result,
    )
    finished = service.commit_run_boundary(request)
    event_count = len(service.load_events("session_1"))
    repeated = service.commit_run_boundary(request)

    assert finished.run.status == "completed"
    assert finished.run.checkpoint is None
    assert finished.session.current_run_id is None
    assert (tmp_path / ".codepilot" / "runs" / "run_1" / "artifacts" / "result.json").is_file()
    assert repeated.run == finished.run
    assert len(service.load_events("session_1")) == event_count


def test_inspect_recovery_uses_run_state_and_message_cursor(tmp_path: Path) -> None:
    service, session = _service(tmp_path)
    begun = _begin(service, session.revision)

    ready = service.inspect_recovery(
        RecoveryRequest(
            session_id="session_1",
            expected_checkpoint_id=begun.run.checkpoint.checkpoint_id,  # type: ignore[union-attr]
        )
    )
    assert ready.status == "ready"
    assert ready.bundle is not None
    assert [item.message_id for item in ready.bundle.messages] == ["message_user"]

    blocked = service.inspect_recovery(
        RecoveryRequest(
            session_id="session_1",
            expected_checkpoint_id="checkpoint_old",
        )
    )
    assert blocked.status == "blocked"


def test_inspect_recovery_detects_workspace_file_changes(tmp_path: Path) -> None:
    from codepilot.sessions.workspace import capture_workspace_checkpoint

    tracked = tmp_path / "app.py"
    tracked.write_text("before\n", encoding="utf-8")
    service, session = _service(tmp_path)
    begun = service.begin_run(
        BeginRunRequest(
            session_id="session_1",
            run_id="run_workspace",
            message_id="message_workspace",
            user_message=UserMessage(content="inspect"),
            workspace=capture_workspace_checkpoint(
                tmp_path,
                tracked_paths=["app.py"],
            ),
        ),
        expected_session_revision=session.revision,
    )
    tracked.write_text("after\n", encoding="utf-8")

    recovery = service.inspect_recovery(
        RecoveryRequest(
            session_id="session_1",
            run_id="run_workspace",
            expected_checkpoint_id=begun.run.checkpoint.checkpoint_id,  # type: ignore[union-attr]
        )
    )

    assert recovery.status == "needs_validation"
    assert recovery.bundle is not None
    assert recovery.bundle.workspace_status.status == "changed"
    assert recovery.bundle.workspace_status.changed_paths == ("app.py",)


def test_appending_event_does_not_modify_authoritative_run_state(tmp_path: Path) -> None:
    service, session = _service(tmp_path)
    begun = _begin(service, session.revision)
    repository = FileSessionRepository(tmp_path)
    before = repository.load_run("run_1")

    repository.append_event(
        {
            "event_id": "event_manual",
            "type": "agent_end",
            "session_id": "session_1",
            "run_id": "run_1",
            "status": "completed",
        }
    )

    assert repository.load_run("run_1") == before == begun.run


def test_atomic_json_failure_preserves_previous_file(tmp_path: Path, monkeypatch) -> None:
    from codepilot.sessions import filesystem

    path = tmp_path / "state.json"
    atomic_write_json(path, {"revision": 1})

    def fail_replace(_source, _target) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(filesystem.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        atomic_write_json(path, {"revision": 2})

    assert read_json_object(path) == {"revision": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_jsonl_allows_only_a_truncated_final_record(tmp_path: Path) -> None:
    path = tmp_path / "messages.jsonl"
    path.write_bytes(b'{"id":"one"}\n{"id":"two"')
    assert read_jsonl(path) == [{"id": "one"}]

    path.write_bytes(b'{"id":\n{"id":"two"}\n')
    with pytest.raises(ValueError, match="line 1"):
        read_jsonl(path)


def test_repository_rejects_unknown_state_fields_instead_of_reading_old_formats(
    tmp_path: Path,
) -> None:
    service, _ = _service(tmp_path)
    path = tmp_path / ".codepilot" / "sessions" / "session_1" / "session.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["runtime_checkpoint"] = {"phase": "legacy"}
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown=.*runtime_checkpoint"):
        service.get_session("session_1")
