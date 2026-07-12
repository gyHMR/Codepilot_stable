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


def test_delete_session_record_removes_only_session_directory(tmp_path) -> None:
    from codepilot.sessions import CreateSessionRequest, ModelRef, SessionStateService

    service = SessionStateService(tmp_path)
    service.create_session(CreateSessionRequest(
        workspace_root=str(tmp_path), model=ModelRef(provider="unit", model="unit-model"),
        session_id="session-one", system_prompt_hash="hash", current_mode="build",
    ))
    keep = tmp_path / "keep.txt"
    keep.write_text("keep", encoding="utf-8")

    assert service.delete_session("session-one") is True
    assert service.list_sessions() == ()
    assert keep.read_text(encoding="utf-8") == "keep"
