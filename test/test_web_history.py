from __future__ import annotations


def test_session_service_discovers_persisted_sessions(tmp_path) -> None:
    from codepilot.sessions import CreateSessionRequest, ModelRef, SessionStateService

    service = SessionStateService(tmp_path)
    service.create_session(CreateSessionRequest(
        workspace_root=str(tmp_path), model=ModelRef(provider="unit", model="unit-model"),
        session_id="session-one", system_prompt_hash="hash", current_mode="build",
    ))

    rows = service.list_sessions()

    assert [row.session_id for row in rows] == ["session-one"]
    assert rows[0].workspace_root == str(tmp_path.resolve()).replace("\\", "/")


def test_delete_session_removes_owned_runs_without_touching_workspace(tmp_path) -> None:
    from codepilot.protocols import UserMessage
    from codepilot.sessions import CreateSessionRequest, ModelRef, SessionStateService
    from codepilot.sessions.service import BeginRunRequest
    from codepilot.sessions.workspace import capture_workspace_checkpoint

    service = SessionStateService(tmp_path)
    session = service.create_session(CreateSessionRequest(
        workspace_root=str(tmp_path), model=ModelRef(provider="unit", model="unit-model"),
        session_id="session-one", system_prompt_hash="hash", current_mode="build",
    ))
    keep = tmp_path / "keep.txt"
    keep.write_text("keep", encoding="utf-8")
    service.begin_run(
        BeginRunRequest(
            session_id="session-one",
            request_id="request-one",
            run_id="run-one",
            user_message=UserMessage(content="run"),
            workspace=capture_workspace_checkpoint(tmp_path),
        ),
        expected_session_revision=session.revision,
    )
    run_dir = tmp_path / ".codepilot" / "runs" / "run-one"
    assert run_dir.is_dir()

    assert service.delete_session("session-one") is True
    assert service.list_sessions() == ()
    assert not run_dir.exists()
    assert keep.read_text(encoding="utf-8") == "keep"
