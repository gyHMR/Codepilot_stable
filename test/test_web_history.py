from __future__ import annotations


def test_list_session_metadata_discovers_persisted_sessions(tmp_path) -> None:
    from codepilot.sessions import list_session_metadata
    from codepilot.sessions.store import SessionStore

    store = SessionStore(tmp_path, "session-one")
    store.ensure_initialized(model_id="unit-model", provider="unit", system_prompt="test")

    rows = list_session_metadata(tmp_path)

    assert [row["session_id"] for row in rows] == ["session-one"]
    assert rows[0]["workspace_root"] == str(tmp_path.resolve()).replace("\\", "/")


def test_delete_session_record_removes_only_session_directory(tmp_path) -> None:
    from codepilot.sessions import delete_session_record, list_session_metadata
    from codepilot.sessions.store import SessionStore

    SessionStore(tmp_path, "session-one").ensure_initialized(
        model_id="unit-model", provider="unit", system_prompt="test"
    )
    keep = tmp_path / "keep.txt"
    keep.write_text("keep", encoding="utf-8")

    assert delete_session_record(tmp_path, "session-one") is True
    assert list_session_metadata(tmp_path) == ()
    assert keep.read_text(encoding="utf-8") == "keep"
