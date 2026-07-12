from __future__ import annotations

import json

import pytest


def test_session_store_rejects_previous_session_schema(tmp_path) -> None:
    from codepilot.sessions.store import SessionLayout, SessionStore

    session_id = "legacy-session"
    layout = SessionLayout.for_workspace(tmp_path, session_id)
    layout.session_dir.mkdir(parents=True)
    layout.session_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "session_id": session_id,
                "runtime_checkpoint": None,
            }
        ),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="Unsupported session schema_version"):
        SessionStore(tmp_path, session_id).ensure_initialized(
            model_id="unit",
            provider="unit",
            system_prompt="",
        )


def test_session_store_rejects_unversioned_message_rows(tmp_path) -> None:
    from codepilot.sessions.store import SessionStore

    store = SessionStore(tmp_path, "session-message-schema")
    store.ensure_initialized(model_id="unit", provider="unit", system_prompt="")
    store.messages_file.write_text(
        json.dumps(
            {
                "id": "old-message",
                "role": "user",
                "content": "old session message",
            }
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="Unsupported session message schema_version"):
        store.load_session_messages()


def test_session_store_rejects_removed_pending_tool_checkpoint_fields(tmp_path) -> None:
    from codepilot.sessions.store import SessionStore

    store = SessionStore(tmp_path, "session-checkpoint-schema")
    store.ensure_initialized(model_id="unit", provider="unit", system_prompt="")

    with pytest.raises(ValueError, match="Removed runtime checkpoint fields"):
        store.set_checkpoint(
            {
                "phase": "tool_approval",
                "run_id": "run-old",
                "pending_tool_calls": [
                    {
                        "id": "call-old",
                        "name": "write",
                        "arguments": {"path": "old.txt"},
                    }
                ],
            }
        )


def test_run_store_rejects_previous_run_artifact_schema(tmp_path) -> None:
    from codepilot.sessions.store import RunStore

    store = RunStore(tmp_path, "session-run-schema")
    path = store.layout.run_file("run-old")
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "run_id": "run-old",
                "session_id": "session-run-schema",
            }
        ),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="Unsupported run artifact schema_version"):
        store.load_run_result("run-old")
