from __future__ import annotations

import asyncio
import json


def test_restricted_tool_port_only_exposes_and_executes_read_allowlist(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.tools.builtins import create_builtin_registrations
        from codepilot.tools.contracts import ToolExecutionRequest
        from codepilot.tools.registry import ToolRegistry
        from codepilot.runtime.tool_adapters.subagents import RestrictedToolPort
        from codepilot.tools.runtime import ToolRuntime

        (tmp_path / "sample.py").write_text("value = 1\n", encoding="utf-8", newline="\n")
        registry = ToolRegistry()
        ids = {
            item.spec.name: registry.register(item)
            for item in create_builtin_registrations(tmp_path, enabled_names=["read", "write"])
        }
        base = ToolRuntime(registry)
        restricted = RestrictedToolPort(base)

        names = {item.spec.name for item in restricted.catalog_snapshot(mode="plan").entries}
        assert names == {"read"}

        allowed = await restricted.execute(
            ToolExecutionRequest(
                "run1", "session1", "read1", "read", {"path": "sample.py"}, "plan", ids["read"]
            )
        )
        denied = await restricted.execute(
            ToolExecutionRequest(
                "run1", "session1", "write1", "write", {"path": "x", "content": "x"}, "plan", ids["write"]
            )
        )
        batch = await restricted.execute_batch(
            [
                ToolExecutionRequest("run1", "session1", "read2", "read", {"path": "sample.py"}, "plan", ids["read"]),
                ToolExecutionRequest("run1", "session1", "read3", "read", {"path": "sample.py"}, "plan", ids["read"]),
            ]
        )

        assert allowed.status == "success"
        assert [item.status for item in batch] == ["success", "success"]
        assert denied.status == "denied"
        assert denied.error.code == "restricted_tool_denied"

    asyncio.run(run_case())


def test_plan_policy_prioritizes_subagents_for_broad_repository_analysis() -> None:
    from codepilot.runtime.session_coordinator import _mode_policy

    policy = _mode_policy("plan")

    assert "固定的宏观工作流" in policy
    assert "探索阶段默认使用 dispatch_exploration" in policy
    assert "主 Agent 不应先用大量 ls/read/grep/find" in policy
    assert "主 Agent 负责" in policy
    assert "框架负责" in policy
    assert "list_exploration_agents" in policy
    assert "dispatch_exploration" in policy
    assert "reuse=auto" in policy


def test_exploration_tool_descriptions_explain_preferred_and_reuse_behavior(tmp_path) -> None:
    from codepilot.runtime.tool_adapters.subagents import create_subagent_registrations

    tools = {
        tool.spec.name: tool
        for tool in create_subagent_registrations(
            workspace=tmp_path,
            session_provider=lambda: None,  # type: ignore[arg-type]
        )
    }

    dispatch = tools["dispatch_exploration"].spec.description
    listed = tools["list_exploration_agents"].spec.description
    assert "Plan mode only" in dispatch
    assert "read-only exploration subagents" in dispatch
    assert "distinct scopes" in dispatch
    assert "integrate the returned evidence" in dispatch
    assert "reuse=auto" in dispatch
    assert "does not create or run subagents" in listed
    assert "reports already produced by dispatch_exploration" in listed


def test_list_exploration_agents_empty_result_points_to_dispatch(tmp_path) -> None:
    async def run_case() -> None:
        from types import SimpleNamespace

        from codepilot.runtime.tool_adapters.subagents import create_subagent_registrations
        from codepilot.tools.contracts import ToolExecutionRequest
        from codepilot.tools.registry import ToolRegistry
        from codepilot.tools.runtime import ToolRuntime

        registrations = create_subagent_registrations(
                workspace=tmp_path,
                session_provider=lambda: SimpleNamespace(session_id="session_a"),
            )
        registry = ToolRegistry()
        ids = {item.spec.name: registry.register(item) for item in registrations}
        result = await ToolRuntime(registry).execute(
            ToolExecutionRequest(
                run_id="run_plan",
                session_id="session_a",
                tool_call_id="list1",
                tool_name="list_exploration_agents",
                mode="plan",
                arguments={},
                registration_id=ids["list_exploration_agents"],
            )
        )

        payload = result.data
        assert payload == {
            "agents": (),
            "has_reports": False,
            "next_action": (
                "Call dispatch_exploration to create read-only exploration subagents."
            ),
        }

    asyncio.run(run_case())


def test_mode_policies_keep_one_agent_identity_and_separate_control_from_task() -> None:
    from codepilot.runtime.session_coordinator import _mode_policy

    plan = _mode_policy("plan")
    build = _mode_policy("build")
    read = _mode_policy("read")

    assert "同一个 Coding Agent" in plan
    assert "对象级" in plan
    assert "控制级" in plan
    assert "派发只读 Subagent 探索仓库" in plan
    assert "禁止修改工作区" in plan
    assert "普通文本方案不是可审批的 Task Plan" in plan
    assert "不要先完整展示文本草案" in plan
    assert "未经运行时确认批准" in plan
    assert "声称已经开始实现" in plan
    assert "执行目标" in build
    assert "不是重新制定方案" in build
    assert "不得创建、推进或完成" in read


def test_plan_approved_continuation_executes_existing_plan_without_replanning() -> None:
    from codepilot.runtime.session_coordinator import _continuation_control

    control = _continuation_control("plan_approved")

    assert control is not None
    assert control["scope"] == "approved_plan_execution_only"
    instruction = str(control["instruction"])
    assert "first unfinished plan item" in instruction
    assert "do not restate it, redesign it, or create another Task Plan" in instruction
    assert "do not call create_build_plan" in instruction
    assert "preserve the canonical item IDs exactly" in instruction
    assert "update_plan_progress" in instruction
    assert "close_plan" in instruction


def test_subagent_registry_keeps_process_local_reports_and_marks_stale(tmp_path) -> None:
    from codepilot.runtime.subagent_registry import SubagentStore

    source = tmp_path / "src" / "app.py"
    source.parent.mkdir()
    source.write_text("print('v1')\n", encoding="utf-8")

    store = SubagentStore(tmp_path, "session_a")
    stored = store.append_report(
        subagent_id="runtime-reader",
        purpose="Read runtime",
        scope_key="scope1",
        focus_paths=["src"],
        report={
            "status": "completed",
            "summary": "found runtime",
            "findings": ["src/app.py exists"],
            "relevant_files": ["src/app.py"],
            "evidence": [{"path": "src/app.py", "note": "entry"}],
            "risks": [],
            "suggested_plan_notes": [],
            "open_questions": [],
            "confidence": 0.8,
        },
        evidence_paths=["src/app.py"],
    )

    agents = store.list_agents()
    assert stored["report_id"]
    assert agents[0]["subagent_id"] == "runtime-reader"
    assert agents[0]["stale"] is False
    assert not (tmp_path / ".codepilot" / "sessions" / "session_a" / "subagents").exists()

    source.write_text("print('v2')\n", encoding="utf-8")

    assert store.list_agents()[0]["stale"] is True


def test_exploration_coordinator_returns_partial_results_and_skips_duplicates(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.llm.ports import ModelDescriptor
        from codepilot.runtime.subagents import ExplorationCoordinator
        from codepilot.runtime.subagent_registry import SubagentStore

        class FakeRunner:
            async def run(self, task, *, peer_assignments, previous_report=None):
                if "risk" in task.purpose.lower():
                    return {
                        "status": "failed",
                        "task_id": task.task_id,
                        "subagent_id": task.subagent_id,
                        "purpose": task.purpose,
                        "scope_key": task.scope_key,
                        "summary": "",
                        "findings": [],
                        "relevant_files": [],
                        "evidence": [],
                        "risks": [],
                        "suggested_plan_notes": [],
                        "open_questions": [],
                        "confidence": 0.0,
                        "error": {"code": "fake.failure"},
                    }
                return {
                    "status": "completed",
                    "task_id": task.task_id,
                    "subagent_id": task.subagent_id,
                    "purpose": task.purpose,
                    "scope_key": task.scope_key,
                    "summary": "ok",
                    "findings": ["fact"],
                    "relevant_files": [],
                    "evidence": [],
                    "risks": [],
                    "suggested_plan_notes": [],
                    "open_questions": [],
                    "confidence": 1.0,
                }

        coordinator = ExplorationCoordinator(
            workspace=tmp_path,
            session_id="session_a",
            model=ModelDescriptor(provider="fake", model_id="unit"),
            model_port=object(),
            tool_port=object(),
            store=SubagentStore(tmp_path, "session_a"),
            runner_factory=FakeRunner,
        )

        result = await coordinator.dispatch(
            {
                "tasks": [
                    {
                        "subagent_id": "reader",
                        "purpose": "Read runtime",
                        "instruction": "read",
                        "focus_paths": ["src/codepilot/runtime"],
                    },
                    {
                        "subagent_id": "risk",
                        "purpose": "Risk review",
                        "instruction": "risk",
                        "focus_paths": ["src/codepilot/tools"],
                    },
                    {
                        "subagent_id": "reader",
                        "purpose": "Read runtime duplicate",
                        "instruction": "read again",
                        "focus_paths": ["src/codepilot/runtime"],
                    },
                ]
            }
        )

        statuses = [report["status"] for report in result["reports"]]
        assert result["batch_status"] == "partial_failed"
        assert "completed" in statuses
        assert "failed" in statuses
        assert "skipped_duplicate" in statuses
        assert result["created_subagent_ids"] == ["reader", "risk"]

    asyncio.run(run_case())


def test_plan_mode_dispatch_exploration_feeds_proposed_plan_and_pauses(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.llm.ports import LLMCompleted
        from codepilot.protocols import AssistantMessage, Model, TextContent, ToolCall
        from codepilot.runtime import RuntimeGateway, SessionOpenIntent
        from codepilot.runtime.actions import PromptSubmitted, RunPausedFrame

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("print('hello')\n", encoding="utf-8")

        class ModelPort:
            def __init__(self) -> None:
                self.parent_calls = 0
                self.subagent_calls = 0

            async def stream(self, request):
                if request.correlation.run_id.startswith("subrun_"):
                    self.subagent_calls += 1
                    yield LLMCompleted(
                        message=AssistantMessage(
                            content=[
                                TextContent(
                                    text=json.dumps(
                                        {
                                            "status": "completed",
                                            "summary": "src/app.py is the only source file.",
                                            "findings": ["src/app.py exists"],
                                            "relevant_files": ["src/app.py"],
                                            "evidence": [{"path": "src/app.py", "note": "source"}],
                                            "risks": [],
                                            "suggested_plan_notes": ["edit src/app.py after approval"],
                                            "open_questions": [],
                                            "confidence": 0.9,
                                        }
                                    )
                                )
                            ]
                        )
                    )
                    return
                self.parent_calls += 1
                if self.parent_calls == 1:
                    yield LLMCompleted(
                        message=AssistantMessage(
                            content=[
                                ToolCall(
                                    id="explore1",
                                    name="dispatch_exploration",
                                    arguments={
                                        "tasks": [
                                            {
                                                "subagent_id": "src-reader",
                                                "purpose": "Read src",
                                                "instruction": "Find source files",
                                                "focus_paths": ["src"],
                                            }
                                        ]
                                    },
                                )
                            ]
                        )
                    )
                    return
                yield LLMCompleted(
                    message=AssistantMessage(
                        content=[
                            ToolCall(
                                id="plan1",
                                name="propose_plan",
                                arguments={
                                    "raw_user_request": "完善 src/app.py，给我一个方案",
                                    "interpreted_goal": "完善 src/app.py 的实现并完成验证。",
                                    "task_understanding": "用户希望先审批完善 src/app.py 的方案。",
                                    "current_implementation": "探索报告已定位 src/app.py 的当前实现。",
                                    "target_design": "按探索结果完善 src/app.py 的实现。",
                                    "impact_scope": "影响 src/app.py 和相关 Python 检查。",
                                    "risks_and_open_questions": ["暂无阻塞待确认项。"],
                                    "verification_plan": "运行相关 Python 检查。",
                                    "summary": "Use exploration evidence to edit src/app.py.",
                                    "explanation": "等待用户审批后执行。",
                                    "completion_criteria": ["Python 检查通过"],
                                    "items": [
                                        {
                                            "step": "修改 src/app.py",
                                            "details": "批准后基于探索结果修改 src/app.py。",
                                            "verification": "运行相关 Python 检查。",
                                        }
                                    ],
                                },
                            )
                        ]
                    )
                )

        model_port = ModelPort()
        gateway = RuntimeGateway(model_port=model_port)
        ref = gateway.open_session(
            SessionOpenIntent(
                workspace_dir=tmp_path,
                current_mode="plan",
                memory_enabled=False,
                model=Model(
                    id="unit",
                    name="Unit",
                    api="unit",
                    provider="unit",
                    base_url="",
                    reasoning=False,
                    input=["text"],
                    context_window=4000,
                    max_tokens=500,
                ),
            )
        )
        assert gateway.describe(ref.session_id).session.current_mode == "plan"

        frames = [
            frame
            async for frame in gateway.dispatch(
                ref.session_id,
                PromptSubmitted(text="完善 src/app.py，给我一个方案"),
            )
        ]
        paused = [frame for frame in frames if isinstance(frame, RunPausedFrame)]
        session = gateway._require_session(ref.session_id)._session  # noqa: SLF001
        messages = [
            record.message
            for record in session.state_service.load_messages(session.session_id)
        ]
        completed_plan_events = [
            frame.event
            for frame in frames
            if getattr(frame, "event", {}).get("type") == "tool_completed"
                and getattr(frame, "event", {}).get("tool_name") == "propose_plan"
        ]
        assert completed_plan_events
        assert completed_plan_events[0]["result"]["data"]["plan_operation"] == "propose_plan"
        assert completed_plan_events[0]["result"]["data"]["plan_state"]["status"] == "proposed"

        assert paused
        assert paused[-1].record.stop_reason == "plan_approval_required"
        assert session.plan_state.current()["interpreted_goal"] == "完善 src/app.py 的实现并完成验证。"
        dispatch_data = next(
            frame.event["result"]["data"]
            for frame in frames
            if getattr(frame, "event", {}).get("type") == "tool_completed"
                and getattr(frame, "event", {}).get("tool_name") == "dispatch_exploration"
        )
        assert model_port.subagent_calls == 1, dispatch_data
        assert any(
            message.tool_name == "dispatch_exploration"
            for message in messages
            if getattr(message, "role", "") == "toolResult"
        )
        assert not any(
            "src/app.py is the only source file" in str(getattr(message, "content", ""))
            for message in messages
            if getattr(message, "role", "") == "assistant"
        )

    asyncio.run(run_case())
