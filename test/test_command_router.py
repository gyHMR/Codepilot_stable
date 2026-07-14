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


def test_cli_command_router_hides_internal_session_tree_commands(tmp_path: Path) -> None:
    asyncio.run(_run_internal_commands_removed_case(tmp_path))


async def _run_internal_commands_removed_case(tmp_path: Path) -> None:
    from codepilot.interfaces.cli.interactive import dispatch_command
    from codepilot.runtime.commands import builtin_commands

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


def test_cli_command_router_reports_no_rollback_run(tmp_path: Path) -> None:
    asyncio.run(_run_rollback_no_run_case(tmp_path))


def test_cli_mode_build_warns_when_plan_is_not_approved(tmp_path: Path) -> None:
    asyncio.run(_run_mode_build_plan_warning_case(tmp_path))


def test_cli_plan_approve_and_repeated_approve_show_plan(tmp_path: Path) -> None:
    asyncio.run(_run_plan_approve_command_case(tmp_path))


def test_cli_plan_approve_applies_pending_revision_through_core(tmp_path: Path) -> None:
    asyncio.run(_run_plan_revision_approve_command_case(tmp_path))


async def _run_mode_build_plan_warning_case(tmp_path: Path) -> None:
    from codepilot.interfaces.cli.interactive import dispatch_command

    runtime, session_id = _create_runtime_session(tmp_path)
    session = _persistent_session(runtime, session_id)
    try:
        session.set_current_mode("plan")
        _seed_proposed_plan(
            session,
            run_id="run_plan",
            plan_id="plan_cli_warning",
            request="先制定方案",
            summary="阅读当前实现后执行聚焦修改。",
        )

        result = await dispatch_command(runtime, session_id, "/mode build")

        assert result.handled
        assert result.data["current_mode"] == "plan"
        assert result.data["blocked"] is True
        assert result.data["status"] == "proposed"
        assert any("/plan approve" in line for line in result.output_lines)
        assert session.current_mode == "plan"
    finally:
        await runtime.close_all()


async def _run_plan_approve_command_case(tmp_path: Path) -> None:
    from codepilot.interfaces.cli.interactive import dispatch_command

    runtime, session_id = _create_runtime_session(tmp_path)
    session = _persistent_session(runtime, session_id)
    try:
        session.set_current_mode("plan")
        _seed_proposed_plan(
            session,
            run_id="run_plan",
            plan_id="plan_cli_approve",
            request="优化登录逻辑",
            summary="阅读实现并修改登录逻辑。",
        )

        approved = await dispatch_command(runtime, session_id, "/plan approve")
        repeated = await dispatch_command(runtime, session_id, "/plan approve")

        assert approved.data["current_mode"] == "build"
        assert approved.data["continuation_kind"] == "plan_approved"
        assert approved.data["continuation_run_id"] == "run_plan"
        assert any("Plan approved" in line for line in approved.output_lines)
        assert any("=== Plan ===" in line for line in approved.output_lines)
        assert repeated.data["status"] == "active"
        assert any("Plan already approved" in line for line in repeated.output_lines)
        assert any("=== Plan ===" in line for line in repeated.output_lines)
    finally:
        await runtime.close_all()


async def _run_plan_revision_approve_command_case(tmp_path: Path) -> None:
    from codepilot.interfaces.cli.interactive import dispatch_command

    runtime, session_id = _create_runtime_session(tmp_path)
    session = _persistent_session(runtime, session_id)
    try:
        session.set_current_mode("build")
        _seed_proposed_plan(
            session,
            run_id="run_plan_revision",
            plan_id="plan_cli_revision",
            request="优化登录逻辑",
            summary="执行原计划。",
            pending_revision=True,
        )

        approved = await dispatch_command(runtime, session_id, "/plan approve")
        current = session.current_plan_state()

        assert approved.data["continuation_kind"] == "plan_approved"
        assert approved.data["continuation_run_id"] == "run_plan_revision"
        assert current is not None
        assert current["pending_revision"] is None
        assert current["definition"]["summary"] == "执行修订计划。"
    finally:
        await runtime.close_all()


def _seed_proposed_plan(
    session,
    *,
    run_id: str,
    plan_id: str,
    request: str,
    summary: str,
    pending_revision: bool = False,
) -> None:
    from codepilot.core.plan import (
        PendingPlanRevision,
        PlanDefinition,
        PlanState,
        PlanStep,
    )
    from codepilot.core.state import CoreState, TaskState
    from codepilot.protocols import UserMessage
    from codepilot.sessions.contracts import WaitingState
    from codepilot.sessions.service import BeginRunRequest, CommitRunBoundaryRequest
    from codepilot.sessions.workspace import capture_workspace_checkpoint

    base_steps = (
        PlanStep(
            f"{plan_id}:step:1",
            "执行修改",
            "实现目标行为。",
            "运行相关测试。",
        ),
    )
    pending = (
        PendingPlanRevision(
            reason="user_request",
            definition=PlanDefinition(
                summary="执行修订计划。",
                completion_criteria=("相关测试通过",),
            ),
            steps=(
                PlanStep(
                    f"{plan_id}:revision:2:step:1",
                    "执行修订修改",
                    "实现修订后的目标行为。",
                    "运行相关测试。",
                ),
            ),
            proposed_at_revision=1,
        )
        if pending_revision
        else None
    )
    plan = PlanState(
        plan_id=plan_id,
        origin="plan_mode",
        status="active" if pending_revision else "proposed",
        revision=2 if pending_revision else 1,
        definition=PlanDefinition(
            summary=summary,
            completion_criteria=("相关测试通过",),
            task_understanding="用户希望先审批方案，再执行聚焦修改。",
            current_implementation="已确认相关代码和测试边界。",
            target_design="按现有结构执行聚焦修改。",
            impact_scope="影响当前任务相关模块和验证。",
            risks_and_open_questions=("暂无阻塞待确认项。",),
            verification_plan="运行相关测试。",
        ),
        steps=base_steps,
        pending_revision=pending,
    )
    begun = session.state_service.begin_run(
        BeginRunRequest(
            session_id=session.session_id,
            run_id=run_id,
            user_message=UserMessage(content=request),
            workspace=capture_workspace_checkpoint(session.workspace_dir),
        ),
        expected_session_revision=session.session_state.revision,
    )
    core_state = CoreState(
        task=TaskState(
            original_request=request,
            current_goal=request,
            plan=plan,
        )
    ).to_dict()
    started = session.state_service.commit_run_boundary(
        CommitRunBoundaryRequest(
            commit_id=f"{run_id}:seed_started",
            kind="progress",
            session_id=session.session_id,
            run_id=run_id,
            expected_run_revision=begun.run.revision,
            expected_session_revision=begun.session.revision,
            phase="model",
            resume_point="before_model",
            core_state=core_state,
            workspace=begun.run.checkpoint.workspace,
        )
    )
    committed = session.state_service.commit_run_boundary(
        CommitRunBoundaryRequest(
            commit_id=f"{run_id}:seed_plan",
            kind="waiting",
            session_id=session.session_id,
            run_id=run_id,
            expected_run_revision=started.run.revision,
            expected_session_revision=started.session.revision,
            phase="model",
            resume_point="after_model",
            core_state=core_state,
            waiting=WaitingState(
                kind="plan_confirmation",
                request_id=plan_id,
                payload={"plan_id": plan_id, "revision": plan.revision},
            ),
            workspace=started.run.checkpoint.workspace,
        )
    )
    session.session_state = committed.session


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
    session.context_service.latest_report = {
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
    from codepilot.runtime.commands import capture_run_rollback_baseline

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
    from codepilot.runtime.commands import capture_run_rollback_baseline

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
    from codepilot.runtime.commands import builtin_commands

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
        assert switched.state_service.load_messages(switched.session_id) == ()
    finally:
        await runtime.close_all()


async def _run_fork_command_case(tmp_path: Path) -> None:
    from codepilot.interfaces.cli.interactive import dispatch_command
    from codepilot.protocols import UserMessage

    runtime, session_id = _create_runtime_session(tmp_path)
    session = _persistent_session(runtime, session_id)
    try:
        session.state_service.append_message(session.session_id, UserMessage(content="seed"))
        session.session_state = session.state_service.get_session(session.session_id)
        result = await dispatch_command(runtime, session_id, "/fork")
        assert result.handled
        assert result.switched_session_id is not None
        assert result.switched_session_id != session_id
        assert result.output_lines[0].startswith("forked session -> session_id=")
        forked = _persistent_session(runtime, result.switched_session_id)
        messages = [
            record.message
            for record in forked.state_service.load_messages(forked.session_id)
        ]
        assert len(messages) == 1
        assert messages[0].content == "seed"
    finally:
        await runtime.close_all()


async def _run_resume_command_case(tmp_path: Path) -> None:
    from codepilot.interfaces.cli.interactive import dispatch_command
    from codepilot.protocols import UserMessage
    from codepilot.runtime.commands import create_fresh_session

    runtime, session_id = _create_runtime_session(tmp_path)
    session = _persistent_session(runtime, session_id)
    try:
        other = create_fresh_session(session)
        other.state_service.append_message(other.session_id, UserMessage(content="older work"))
        other.session_state = other.state_service.get_session(other.session_id)
        session.state_service.append_message(session.session_id, UserMessage(content="current work"))
        session.session_state = session.state_service.get_session(session.session_id)

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
