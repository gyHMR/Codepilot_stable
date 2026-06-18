from __future__ import annotations

import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _test_model():
    from codepilot.protocols import Model

    return Model(
        id="test-model",
        name="Test Model",
        api="unit-test",
        provider="unit-test",
        base_url="",
        reasoning=False,
        input=["text"],
        context_window=1000,
        max_tokens=100,
    )


def test_runtime_command_router_handles_session_command(tmp_path: Path) -> None:
    asyncio.run(_run_session_command_case(tmp_path))


async def _run_session_command_case(tmp_path: Path) -> None:
    from codepilot.runtime.command_registry import handle_runtime_command
    from codepilot.sessions.session import AgentSession
    from codepilot.sessions.types import AgentSessionOptions

    session = AgentSession(
        AgentSessionOptions(
            model=_test_model(),
            workspace_dir=tmp_path,
            system_prompt="sys",
        )
    )
    try:
        result = await handle_runtime_command(session, "/session")
        assert result.handled
        assert result.switched_session is None
        assert result.output_lines == [f"session_id={session.session_id} leaf_id={session.get_leaf_id()}"]
    finally:
        session.close()


def test_runtime_command_router_clear_switches_session(tmp_path: Path) -> None:
    asyncio.run(_run_clear_command_case(tmp_path))


async def _run_clear_command_case(tmp_path: Path) -> None:
    from codepilot.runtime.command_registry import handle_runtime_command
    from codepilot.sessions.session import AgentSession
    from codepilot.sessions.types import AgentSessionOptions

    session = AgentSession(
        AgentSessionOptions(
            model=_test_model(),
            workspace_dir=tmp_path,
            system_prompt="sys",
        )
    )
    result = None
    try:
        result = await handle_runtime_command(session, "/clear")
        assert result.handled
        assert result.switched_session is not None
        assert result.switched_session.session_id != session.session_id
        assert result.output_lines[0].startswith("context cleared -> new session_id=")
    finally:
        if result is not None and result.switched_session is not None:
            result.switched_session.close()
        session.close()
