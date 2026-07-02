from __future__ import annotations

from pathlib import Path


def test_context_compaction_module_is_removed() -> None:
    from importlib.util import find_spec

    assert find_spec("codepilot.sessions.context.compaction") is None


def test_agent_session_uses_governor_and_slim_layout(tmp_path: Path) -> None:
    from codepilot.protocols import Model
    from codepilot.sessions import AgentSession
    from codepilot.sessions.types import AgentSessionOptions

    session = AgentSession(
        AgentSessionOptions(
            model=Model(
                id="test",
                name="Test",
                api="unit-test",
                provider="unit-test",
                base_url="",
                reasoning=False,
                input=["text"],
                context_window=1000,
                max_tokens=100,
            ),
            workspace_dir=tmp_path,
            session_id="session_no_legacy_context",
        )
    )
    try:
        session_dir = tmp_path / ".codepilot" / "sessions" / session.session_id
        assert session.context_governor is not None
        assert session.prepare_context == session.context_governor.prepare
        assert (session_dir / "session.json").exists()
        assert (session_dir / "messages.jsonl").exists()
        assert not (session_dir / "context.jsonl").exists()
        assert not (session_dir / "runs.jsonl").exists()
    finally:
        session.close()
