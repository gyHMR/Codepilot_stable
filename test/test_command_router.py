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


def _create_runtime_session(tmp_path: Path):
    from codepilot.runtime.contracts import CreateAgentSessionOptions
    from codepilot.runtime.service import RuntimeService

    runtime = RuntimeService()
    handle = runtime.create_session(
        CreateAgentSessionOptions(
            model=_test_model(),
            workspace_dir=tmp_path,
            system_prompt="sys",
        )
    )
    return runtime, handle.session_id


def test_cli_command_router_handles_session_command(tmp_path: Path) -> None:
    asyncio.run(_run_session_command_case(tmp_path))


async def _run_session_command_case(tmp_path: Path) -> None:
    from codepilot.interfaces.cli.commands import handle_cli_command

    runtime, session_id = _create_runtime_session(tmp_path)
    session = runtime.get_session(session_id)
    try:
        result = await handle_cli_command(runtime, session_id, "/session")
        assert result.handled
        assert result.switched_session_id is None
        assert result.output_lines == [f"session_id={session.session_id} leaf_id={session.get_leaf_id()}"]
    finally:
        runtime.close_all()


def test_cli_command_router_clear_switches_session(tmp_path: Path) -> None:
    asyncio.run(_run_clear_command_case(tmp_path))


def test_cli_command_router_shows_context_report(tmp_path: Path) -> None:
    asyncio.run(_run_context_command_case(tmp_path))


def test_cli_command_router_manages_project_memory(tmp_path: Path) -> None:
    asyncio.run(_run_memory_command_case(tmp_path))


async def _run_memory_command_case(tmp_path: Path) -> None:
    from codepilot.interfaces.cli.commands import handle_cli_command

    runtime, session_id = _create_runtime_session(tmp_path)
    session = runtime.get_session(session_id)
    try:
        added = await handle_cli_command(
            runtime,
            session_id,
            "/memory add tests use python -m pytest test -q",
        )
        memory_id = added.output_lines[0].split(": ", 1)[1]
        listed = await handle_cli_command(runtime, session_id, "/memory list project")
        forgotten = await handle_cli_command(runtime, session_id, f"/memory forget {memory_id}")

        assert added.handled
        assert any(memory_id in line for line in listed.output_lines)
        assert forgotten.output_lines == [f"memory forgotten: {memory_id}"]
        assert session.memory_store.get(memory_id).status == "deleted"
    finally:
        runtime.close_all()


async def _run_context_command_case(tmp_path: Path) -> None:
    from codepilot.interfaces.cli.commands import handle_cli_command

    runtime, session_id = _create_runtime_session(tmp_path)
    session = runtime.get_session(session_id)
    session.latest_context_report = {
        "context_id": "ctx_1",
        "repository_fingerprint": "abcdef1234567890",
        "total_budget_tokens": 1000,
        "estimated_tokens_before": 800,
        "estimated_tokens_after": 500,
        "stale_items": [],
        "dropped_items": [{"item_id": "old"}],
        "sections": [],
    }
    try:
        result = await handle_cli_command(runtime, session_id, "/context")
        assert result.handled
        assert any("ctx_1" in line for line in result.output_lines)
        assert any("Dropped items" in line for line in result.output_lines)
    finally:
        runtime.close_all()


async def _run_clear_command_case(tmp_path: Path) -> None:
    from codepilot.interfaces.cli.commands import handle_cli_command

    runtime, session_id = _create_runtime_session(tmp_path)
    try:
        result = await handle_cli_command(runtime, session_id, "/clear")
        assert result.handled
        assert result.switched_session_id is not None
        assert result.switched_session_id != session_id
        assert result.output_lines[0].startswith("context cleared -> new session_id=")
    finally:
        runtime.close_all()
