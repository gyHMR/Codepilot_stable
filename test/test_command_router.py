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
    from codepilot.runtime import RuntimeGateway, SessionOpenIntent

    runtime = RuntimeGateway()
    handle = runtime.open_session(
        SessionOpenIntent(
            model=_test_model(),
            workspace_dir=tmp_path,
            memory_enabled=True,
        )
    )
    return runtime, handle.session_id


def _persistent_session(runtime, session_id: str):
    controller = runtime._require_session(session_id)
    session = getattr(controller, "_session", None)
    assert session is not None
    return session


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
    from codepilot.sessions.rollback import build_rollback_metadata

    _append_run(session, run_id, affected_paths)
    session.store.write_rollback_metadata(
        run_id,
        build_rollback_metadata(
            baseline,
            affected_paths=affected_paths,
            workspace_changed=True,
        ),
    )


def test_cli_command_router_hides_internal_session_tree_commands(tmp_path: Path) -> None:
    asyncio.run(_run_internal_commands_removed_case(tmp_path))


async def _run_internal_commands_removed_case(tmp_path: Path) -> None:
    from codepilot.interfaces.cli.interactive import dispatch_command
    from codepilot.runtime.views import builtin_commands

    runtime, session_id = _create_runtime_session(tmp_path)
    try:
        public_names = {command.name for command in builtin_commands()}
        assert {"session", "tree", "path", "switch", "clear"}.isdisjoint(public_names)

        help_result = await dispatch_command(runtime, session_id, "/help")
        help_text = "\n".join(help_result.output_lines)
        assert "`/resume`" in help_text
        assert "`/session`" not in help_text
        assert "`/tree`" not in help_text

        result = await dispatch_command(runtime, session_id, "/session")
        assert result.handled is False
    finally:
        await runtime.close_all()


def test_cli_command_router_new_switches_to_empty_session(tmp_path: Path) -> None:
    asyncio.run(_run_new_command_case(tmp_path))


def test_cli_command_router_fork_switches_to_copied_session(tmp_path: Path) -> None:
    asyncio.run(_run_fork_command_case(tmp_path))


def test_cli_command_router_resume_lists_and_switches_sessions(tmp_path: Path) -> None:
    asyncio.run(_run_resume_command_case(tmp_path))


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


def test_cli_mode_build_warns_when_plan_is_not_approved(tmp_path: Path) -> None:
    asyncio.run(_run_mode_build_plan_warning_case(tmp_path))


def test_cli_plan_approve_and_repeated_approve_show_plan(tmp_path: Path) -> None:
    asyncio.run(_run_plan_approve_command_case(tmp_path))


async def _run_mode_build_plan_warning_case(tmp_path: Path) -> None:
    from codepilot.interfaces.cli.interactive import dispatch_command

    runtime, session_id = _create_runtime_session(tmp_path)
    session = _persistent_session(runtime, session_id)
    try:
        session.set_current_mode("plan")
        session.plan_state.save(
            {
                "schema_version": 2,
                "plan_id": "plan_cli_warning",
                "owner_run_id": "run_plan",
                "status": "proposed",
                "approval_state": "pending",
                "origin_mode": "plan",
                "objective": "先制定方案",
                "summary": "阅读当前实现后执行聚焦修改。",
                "items": [
                    {"id": "item_1", "step": "阅读实现", "details": "定位相关代码。", "verification": "确认修改点。", "status": "pending"},
                    {"id": "item_2", "step": "执行修改", "details": "实现目标行为。", "verification": "运行相关测试。", "status": "pending"},
                ],
                "revision": 1,
                "explanation": "",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "completed_at": None,
                "completion_source": None,
            }
        )

        result = await dispatch_command(runtime, session_id, "/mode build")

        assert result.handled
        assert result.data["current_mode"] == "plan"
        assert result.data["blocked"] is True
        assert result.data["plan_status"] == "proposed"
        assert any("/plan approve" in line for line in result.output_lines)
        assert session.plan_state.current()["approval_state"] == "pending"
        assert session.current_mode == "plan"
    finally:
        await runtime.close_all()


async def _run_plan_approve_command_case(tmp_path: Path) -> None:
    from codepilot.interfaces.cli.interactive import dispatch_command

    runtime, session_id = _create_runtime_session(tmp_path)
    session = _persistent_session(runtime, session_id)
    try:
        session.set_current_mode("plan")
        session.plan_state.save(
            {
                "schema_version": 2,
                "plan_id": "plan_cli_approve",
                "owner_run_id": "run_plan",
                "status": "proposed",
                "approval_state": "pending",
                "origin_mode": "plan",
                "objective": "优化登录逻辑",
                "summary": "阅读实现并修改登录逻辑。",
                "items": [
                    {"id": "item_1", "step": "阅读实现", "details": "定位登录流程。", "verification": "确认调用路径。", "status": "pending"},
                    {"id": "item_2", "step": "修改登录逻辑", "details": "实现目标行为。", "verification": "运行登录测试。", "status": "pending"},
                ],
                "revision": 1,
                "explanation": "",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "completed_at": None,
                "completion_source": None,
            }
        )

        approved = await dispatch_command(runtime, session_id, "/plan approve")
        repeated = await dispatch_command(runtime, session_id, "/plan approve")

        assert approved.data["current_mode"] == "build"
        assert approved.data["continuation_kind"] == "plan_approved"
        assert approved.data["continuation_run_id"] == "run_plan"
        assert any("Plan approved" in line for line in approved.output_lines)
        assert any("=== Plan ===" in line for line in approved.output_lines)
        assert repeated.data["plan_status"] == "active"
        assert any("Plan already approved" in line for line in repeated.output_lines)
        assert any("=== Plan ===" in line for line in repeated.output_lines)
    finally:
        await runtime.close_all()


async def _run_memory_command_case(tmp_path: Path) -> None:
    from codepilot.interfaces.cli.interactive import dispatch_command

    runtime, session_id = _create_runtime_session(tmp_path)
    try:
        added = await dispatch_command(
            runtime,
            session_id,
            "/memory add tests use python -m pytest test -q",
        )
        memory_id = added.output_lines[0].split(": ", 1)[1]
        listed = await dispatch_command(runtime, session_id, "/memory list project")
        deleted_once = await dispatch_command(runtime, session_id, f"/memory delete {memory_id}")
        deleted = await dispatch_command(runtime, session_id, "/memory list deleted")

        assert added.handled
        assert any(memory_id in line for line in listed.output_lines)
        assert deleted_once.output_lines == (f"memory deleted: {memory_id}",)
        assert any(memory_id in line for line in deleted.output_lines)
    finally:
        await runtime.close_all()


async def _run_context_command_case(tmp_path: Path) -> None:
    from codepilot.interfaces.cli.interactive import dispatch_command

    runtime, session_id = _create_runtime_session(tmp_path)
    session = _persistent_session(runtime, session_id)
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
        result = await dispatch_command(runtime, session_id, "/context")
        assert result.handled
        assert any("ctx_1" in line for line in result.output_lines)
        assert any("Dropped items" in line for line in result.output_lines)
    finally:
        await runtime.close_all()


async def _run_rollback_command_case(tmp_path: Path) -> None:
    from codepilot.interfaces.cli.interactive import dispatch_command
    from codepilot.sessions.commands import capture_run_rollback_baseline

    _init_repo(tmp_path)
    tracked = tmp_path / "app.py"
    tracked.write_text("print('before')\n", encoding="utf-8")
    _git(tmp_path, "add", "app.py")
    _git(tmp_path, "commit", "-m", "baseline")

    runtime, session_id = _create_runtime_session(tmp_path)
    session = _persistent_session(runtime, session_id)
    try:
        baseline = capture_run_rollback_baseline(session)
        tracked.write_text("print('after')\n", encoding="utf-8")
        _append_run_with_rollback(
            session,
            "run_cli_rollback",
            baseline=baseline,
            affected_paths=["app.py"],
        )

        preview = await dispatch_command(runtime, session_id, "/rollback run_cli_rollback")
        applied = await dispatch_command(runtime, session_id, "/rollback apply run_cli_rollback")

        assert preview.handled
        assert any("Rollback preview" in line for line in preview.output_lines)
        assert any("restore app.py" in line for line in preview.output_lines)
        assert applied.handled
        assert any("Rollback result" in line for line in applied.output_lines)
        assert any("status=reverted" in line for line in applied.output_lines)
        assert tracked.read_text(encoding="utf-8") == "print('before')\n"
    finally:
        await runtime.close_all()


async def _run_rollback_no_run_case(tmp_path: Path) -> None:
    from codepilot.interfaces.cli.interactive import dispatch_command

    runtime, session_id = _create_runtime_session(tmp_path)
    try:
        result = await dispatch_command(runtime, session_id, "/rollback")

        assert result.handled
        assert any("status=not_eligible" in line for line in result.output_lines)
        assert any("no_run_results" in line for line in result.output_lines)
    finally:
        await runtime.close_all()


async def _run_rollback_blocked_case(tmp_path: Path) -> None:
    from codepilot.interfaces.cli.interactive import dispatch_command
    from codepilot.sessions.commands import capture_run_rollback_baseline

    _init_repo(tmp_path)
    tracked = tmp_path / "app.py"
    tracked.write_text("print('before')\n", encoding="utf-8")
    _git(tmp_path, "add", "app.py")
    _git(tmp_path, "commit", "-m", "baseline")

    runtime, session_id = _create_runtime_session(tmp_path)
    session = _persistent_session(runtime, session_id)
    try:
        baseline = capture_run_rollback_baseline(session)
        tracked.write_text("print('after')\n", encoding="utf-8")
        _append_run_with_rollback(
            session,
            "run_cli_blocked",
            baseline=baseline,
            affected_paths=["app.py"],
        )
        _git(tmp_path, "add", "app.py")

        result = await dispatch_command(runtime, session_id, "/rollback apply")

        assert result.handled
        assert any("status=conflict" in line for line in result.output_lines)
        assert any("affected_path_has_staged_changes" in line for line in result.output_lines)
        assert tracked.read_text(encoding="utf-8") == "print('after')\n"
    finally:
        await runtime.close_all()


async def _run_removed_compact_command_case(tmp_path: Path) -> None:
    from codepilot.interfaces.cli.interactive import dispatch_command
    from codepilot.runtime.views import builtin_commands

    runtime, session_id = _create_runtime_session(tmp_path)
    try:
        assert "compact" not in {command.name for command in builtin_commands()}
        assert "compact" not in {
            command.name for command in runtime.describe(session_id).commands
        }

        result = await dispatch_command(runtime, session_id, "/compact")

        assert result.handled is False
        assert result.output_lines == ()
    finally:
        await runtime.close_all()


async def _run_new_command_case(tmp_path: Path) -> None:
    from codepilot.interfaces.cli.interactive import dispatch_command

    runtime, session_id = _create_runtime_session(tmp_path)
    try:
        result = await dispatch_command(runtime, session_id, "/new")
        assert result.handled
        assert result.switched_session_id is not None
        assert result.switched_session_id != session_id
        assert result.output_lines[0].startswith("new session -> session_id=")
        switched = _persistent_session(runtime, result.switched_session_id)
        assert switched.store.load_session_messages() == []
    finally:
        await runtime.close_all()


async def _run_fork_command_case(tmp_path: Path) -> None:
    from codepilot.interfaces.cli.interactive import dispatch_command
    from codepilot.protocols import UserMessage

    runtime, session_id = _create_runtime_session(tmp_path)
    session = _persistent_session(runtime, session_id)
    try:
        session.store.append_message(UserMessage(content="seed"))
        result = await dispatch_command(runtime, session_id, "/fork")
        assert result.handled
        assert result.switched_session_id is not None
        assert result.switched_session_id != session_id
        assert result.output_lines[0].startswith("forked session -> session_id=")
        forked = _persistent_session(runtime, result.switched_session_id)
        messages = forked.store.load_session_messages()
        assert len(messages) == 1
        assert messages[0].content == "seed"
    finally:
        await runtime.close_all()


async def _run_resume_command_case(tmp_path: Path) -> None:
    from codepilot.interfaces.cli.interactive import dispatch_command
    from codepilot.protocols import UserMessage
    from codepilot.sessions.commands import create_fresh_session

    runtime, session_id = _create_runtime_session(tmp_path)
    session = _persistent_session(runtime, session_id)
    try:
        other = create_fresh_session(session)
        other.store.append_message(UserMessage(content="older work"))
        session.store.append_message(UserMessage(content="current work"))

        listing = await dispatch_command(runtime, session_id, "/resume")
        assert listing.handled
        assert any("Recent sessions" in line for line in listing.output_lines)
        assert any(other.session_id in line for line in listing.output_lines)

        by_id = await dispatch_command(runtime, session_id, f"/resume {other.session_id}")
        assert by_id.handled
        assert by_id.switched_session_id == other.session_id
        assert runtime.describe(other.session_id).session.session_id == other.session_id
    finally:
        await runtime.close_all()
