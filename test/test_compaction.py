from __future__ import annotations

from pathlib import Path


def test_context_is_an_independent_source_module() -> None:
    import codepilot.sessions.context as context

    assert hasattr(context, "__path__")


def test_session_runtime_uses_context_service_and_slim_layout(tmp_path: Path) -> None:
    from codepilot.protocols import Model
    from codepilot.sessions.contracts import SessionOptions
    from codepilot.runtime.session_coordinator import RuntimeSessionCoordinator

    session = RuntimeSessionCoordinator(
        SessionOptions(
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
        assert session.context_service is not None
        assert callable(session.context_service.prepare)
        assert not hasattr(session, "prepare_context")
        assert not hasattr(session, "context_governor")
        assert (session_dir / "session.json").exists()
        assert not (session_dir / "context.jsonl").exists()
        assert not (session_dir / "context_ledger.jsonl").exists()
        assert not (session_dir / "runs.jsonl").exists()
    finally:
        session.close()
