from __future__ import annotations

import asyncio
import subprocess
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


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _init_repo(root: Path) -> None:
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test User")


def _append_run(session, run_id: str, affected_paths: list[str]) -> None:
    from codepilot.protocols import (
        AgentRunCounters,
        AgentRunResult,
        AssistantMessage,
        TextContent,
    )

    final = AssistantMessage(content=[TextContent(text="done")])
    result = AgentRunResult(
        run_id=run_id,
        session_id=session.session_id,
        status="completed",
        stop_reason="final_answer",
        counters=AgentRunCounters(tool_calls=1),
        messages=[final],
        final_message=final,
        affected_paths=affected_paths,
        workspace_changed=True,
    )
    session.store.append_run_result(result)


def _append_run_with_rollback(session, run_id: str, *, baseline, affected_paths: list[str]) -> None:
    from codepilot.sessions.history.git_rollback import build_rollback_metadata

    _append_run(session, run_id, affected_paths)
    session.store.write_rollback_metadata(
        run_id,
        build_rollback_metadata(
            baseline,
            affected_paths=affected_paths,
            workspace_changed=True,
        ),
    )


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


def test_cli_command_router_does_not_expose_removed_compact_command(tmp_path: Path) -> None:
    asyncio.run(_run_removed_compact_command_case(tmp_path))


def test_cli_command_router_manages_project_memory(tmp_path: Path) -> None:
    asyncio.run(_run_memory_command_case(tmp_path))


def test_cli_command_router_previews_and_applies_rollback(tmp_path: Path) -> None:
    asyncio.run(_run_rollback_command_case(tmp_path))


def test_cli_command_router_reports_no_rollback_run(tmp_path: Path) -> None:
    asyncio.run(_run_rollback_no_run_case(tmp_path))


def test_cli_command_router_reports_blocked_rollback(tmp_path: Path) -> None:
    asyncio.run(_run_rollback_blocked_case(tmp_path))


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


async def _run_rollback_command_case(tmp_path: Path) -> None:
    from codepilot.interfaces.cli.commands import handle_cli_command

    _init_repo(tmp_path)
    tracked = tmp_path / "app.py"
    tracked.write_text("print('before')\n", encoding="utf-8")
    _git(tmp_path, "add", "app.py")
    _git(tmp_path, "commit", "-m", "baseline")

    runtime, session_id = _create_runtime_session(tmp_path)
    session = runtime.get_session(session_id)
    try:
        baseline = session.capture_run_rollback_baseline()
        tracked.write_text("print('after')\n", encoding="utf-8")
        _append_run_with_rollback(
            session,
            "run_cli_rollback",
            baseline=baseline,
            affected_paths=["app.py"],
        )

        preview = await handle_cli_command(runtime, session_id, "/rollback run_cli_rollback")
        applied = await handle_cli_command(runtime, session_id, "/rollback apply run_cli_rollback")

        assert preview.handled
        assert any("Rollback preview" in line for line in preview.output_lines)
        assert any("restore app.py" in line for line in preview.output_lines)
        assert applied.handled
        assert any("Rollback result" in line for line in applied.output_lines)
        assert any("status=reverted" in line for line in applied.output_lines)
        assert tracked.read_text(encoding="utf-8") == "print('before')\n"
    finally:
        runtime.close_all()


async def _run_rollback_no_run_case(tmp_path: Path) -> None:
    from codepilot.interfaces.cli.commands import handle_cli_command

    runtime, session_id = _create_runtime_session(tmp_path)
    try:
        result = await handle_cli_command(runtime, session_id, "/rollback")

        assert result.handled
        assert any("status=not_eligible" in line for line in result.output_lines)
        assert any("no_run_results" in line for line in result.output_lines)
    finally:
        runtime.close_all()


async def _run_rollback_blocked_case(tmp_path: Path) -> None:
    from codepilot.interfaces.cli.commands import handle_cli_command

    _init_repo(tmp_path)
    tracked = tmp_path / "app.py"
    tracked.write_text("print('before')\n", encoding="utf-8")
    _git(tmp_path, "add", "app.py")
    _git(tmp_path, "commit", "-m", "baseline")

    runtime, session_id = _create_runtime_session(tmp_path)
    session = runtime.get_session(session_id)
    try:
        baseline = session.capture_run_rollback_baseline()
        tracked.write_text("print('after')\n", encoding="utf-8")
        _append_run_with_rollback(
            session,
            "run_cli_blocked",
            baseline=baseline,
            affected_paths=["app.py"],
        )
        _git(tmp_path, "add", "app.py")

        result = await handle_cli_command(runtime, session_id, "/rollback apply")

        assert result.handled
        assert any("status=conflict" in line for line in result.output_lines)
        assert any("affected_path_has_staged_changes" in line for line in result.output_lines)
        assert tracked.read_text(encoding="utf-8") == "print('after')\n"
    finally:
        runtime.close_all()


async def _run_removed_compact_command_case(tmp_path: Path) -> None:
    from codepilot.interfaces.cli.commands import (
        builtin_commands,
        handle_cli_command,
        list_runtime_commands,
    )

    runtime, session_id = _create_runtime_session(tmp_path)
    session = runtime.get_session(session_id)
    try:
        assert "compact" not in {command.name for command in builtin_commands()}
        assert "compact" not in {command.name for command in list_runtime_commands(session)}

        result = await handle_cli_command(runtime, session_id, "/compact")

        assert result.handled is False
        assert result.output_lines == []
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
