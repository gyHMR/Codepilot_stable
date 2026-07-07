from __future__ import annotations

import asyncio
from typing import Any

import pytest

from codepilot.core.contracts import AgentLoopInput, AgentLoopLimits, AgentLoopPorts, RunCorrelation, TaskStrategy
from codepilot.core.loop import run_agent_loop
from codepilot.llm.ports import LLMCompleted, ModelDescriptor
from codepilot.protocols import AssistantMessage, RunVerification, TextContent, ToolCall, ToolResultMessage, UserMessage
from codepilot.tools.ports import ToolObservation


def _canonical_task_state(**updates: object) -> dict[str, object]:
    state: dict[str, object] = {
        "schema_version": 2,
        "task_id": "task_existing",
        "raw_user_request": "继续任务",
        "current_mode": "build",
        "approval_state": "approved",
        "goal": {"value": "继续任务", "source": "user", "confidence": "explicit"},
        "user_constraints": [],
        "proposed_plan": None,
        "approved_plan": None,
        "current_step_id": "step_2",
        "steps": [
            {
                "id": "step_1",
                "title": "读取上下文",
                "kind": "read",
                "status": "completed",
                "acceptance": None,
                "verification_hint": None,
                "summary": "done",
                "evidence_refs": ["tool:read_1"],
                "failure_count": 0,
            },
            {
                "id": "step_2",
                "title": "运行验证",
                "kind": "verify",
                "status": "pending",
                "acceptance": "验证通过",
                "verification_hint": "pytest",
                "summary": None,
                "evidence_refs": [],
                "failure_count": 0,
            },
        ],
        "verification_status": "unknown",
        "evidence_refs": [],
        "blocked_reason": None,
        "recovery_summary": "",
        "source_run_id": "run_1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    state.update(updates)
    return state


class _ScriptedModel:
    def __init__(self, response) -> None:
        self._response = response
        self.calls = 0
        self.requests: list[Any] = []

    async def stream(self, request):
        self.calls += 1
        self.requests.append(request)
        yield LLMCompleted(message=self._response(request, self.calls))


class _ToolPort:
    def __init__(self, observations: dict[str, ToolObservation]) -> None:
        self.observations = observations
        self.executed: list[str] = []

    def catalog(self):
        return {"tools": list(self.observations)}

    async def execute(self, invocation):
        self.executed.append(invocation.name)
        return self.observations[invocation.name]

    async def resume(self, decision):
        raise AssertionError("resume is not used in these tests")


def _loop_input(
    run_id: str,
    *,
    prompt: str = "do task",
    limits: AgentLoopLimits | None = None,
    task_strategy: TaskStrategy | None = None,
) -> AgentLoopInput:
    return AgentLoopInput(
        run_id=run_id,
        correlation=RunCorrelation(session_id="s1"),
        user_prompt=prompt,
        model=ModelDescriptor(provider="fake", model_id="unit"),
        limits=limits or AgentLoopLimits(),
        task_strategy=task_strategy or TaskStrategy(enabled=True, mode="build"),
    )


def test_task_contracts_keep_modes_and_budget_small() -> None:
    from codepilot.core.task import budget_for_profile, ensure_task_mode, policy_for_mode

    assert ensure_task_mode("build") == "build"
    assert policy_for_mode("read").read_only is True
    assert policy_for_mode("plan").planner_required is True
    assert budget_for_profile("wide").max_tool_calls > budget_for_profile("balanced").max_tool_calls
    with pytest.raises(ValueError):
        ensure_task_mode("auto")


def test_task_state_tracks_current_step_and_status_lists() -> None:
    from codepilot.core.task import TaskState, TaskStep

    task = TaskState(
        task_id="task_1",
        goal="修复问题",
        steps=[
            TaskStep(id="step_1", title="读取代码", kind="read"),
            TaskStep(id="step_2", title="运行验证", kind="verify"),
        ],
    )

    first = task.advance()
    assert first is not None
    assert first.status == "in_progress"
    first.complete(evidence_refs=["tool:read"])
    task.advance()

    assert task.completed_step_titles() == ["读取代码"]
    assert task.pending_step_titles() == ["运行验证"]
    assert task.current_step_id == "step_2"


def test_task_planner_parses_json_and_falls_back() -> None:
    from codepilot.core.task import TaskPlanner

    planner = TaskPlanner()
    draft = planner.parse_plan_message(
        AssistantMessage(
            content=[
                TextContent(
                    text='{"goal":"修复 bug","steps":[{"title":"读取文件","kind":"read"}]}'
                )
            ]
        ),
        fallback_goal="完成当前请求",
    )

    assert draft.source == "llm"
    assert draft.goal == "修复 bug"
    assert draft.steps[0].title == "读取文件"
    assert planner.parse_plan_message(
        AssistantMessage(content=[TextContent(text="not json")]),
        fallback_goal="兜底",
    ).source == "fallback"


def test_task_controller_updates_steps_from_tool_results() -> None:
    from codepilot.core.state import RunState
    from codepilot.core.task import TaskController

    controller = TaskController()
    task = controller.initialize(
        [UserMessage(content="修复并测试")],
        proposed_steps=[
            {"title": "读取代码", "kind": "read"},
            {"title": "运行验证", "kind": "verify"},
        ],
    )
    run = RunState("run_1", "s1")

    read_result = ToolResultMessage(tool_call_id="read_1", tool_name="read_file")
    run.collect_tool_results([read_result])
    decision = controller.after_tool_results(task, run, [read_result])
    assert decision.action == "continue"
    assert task.completed_step_titles() == ["读取代码"]
    assert task.current_step_id == "step_2"

    failed = ToolResultMessage(
        tool_call_id="test_1",
        tool_name="pytest",
        status="error",
        verification={"status": "failed", "command": "pytest", "summary": "failed"},
    )
    run.collect_tool_results([failed])
    decision = controller.after_tool_results(task, run, [failed])
    assert decision.reason == "verification_failed"
    assert task.current_step().failure_count == 1  # type: ignore[union-attr]

    passed = ToolResultMessage(
        tool_call_id="test_2",
        tool_name="pytest",
        verification={"status": "passed", "command": "pytest", "summary": "passed"},
    )
    run.collect_tool_results([passed])
    decision = controller.after_tool_results(task, run, [passed])
    assert decision.action == "continue"
    assert controller.check_completion(task, run).satisfied is True


def test_build_task_does_not_finish_after_generic_read_only_tool_result() -> None:
    from codepilot.core.state import RunState
    from codepilot.core.task import TaskController

    controller = TaskController()
    task = controller.initialize([UserMessage(content="修复登录注册功能")])
    run = RunState("run_1", "s1")

    result = ToolResultMessage(tool_call_id="ls_1", tool_name="ls")
    run.collect_tool_results([result])
    decision = controller.after_tool_results(task, run, [result])

    assert decision.action == "continue"
    assert task.open_steps()
    assert task.completed_step_titles() == []
    assert task.current_step_id == "step_1"


def test_completion_gate_requires_fresh_verification_after_workspace_change() -> None:
    from codepilot.core.state import RunState
    from codepilot.core.task import TaskController

    controller = TaskController()
    task = controller.initialize([UserMessage(content="修改文件")])
    run = RunState("run_1", "s1")
    changed = ToolResultMessage(
        tool_call_id="edit_1",
        tool_name="edit_file",
        affected_paths=["src/app.py"],
        workspace_changed=True,
    )
    run.collect_tool_results([changed])
    controller.after_tool_results(task, run, [changed])

    check = controller.check_completion(task, run)
    assert check.satisfied is False
    assert check.reason == "modified_without_fresh_verification"
    assert check.can_continue is True
    assert controller.check_completion(task, run).can_continue is False


def test_task_state_payload_mapping_builds_task_state() -> None:
    from codepilot.core.task import build_task_state_from_payload

    task = build_task_state_from_payload(
        [UserMessage(content="继续任务")],
        _canonical_task_state(),
    )

    assert task is not None
    assert task.task_id == "task_existing"
    assert task.goal == "继续任务"
    assert task.current_step_id == "step_2"
    assert task.steps[1].status == "in_progress"


def test_agent_loop_completes_after_passed_verification() -> None:
    asyncio.run(_agent_loop_completes_after_passed_verification())


async def _agent_loop_completes_after_passed_verification() -> None:
    model = _ScriptedModel(
        lambda _request, calls: AssistantMessage(
            content=[ToolCall(id="test_1", name="pytest", arguments={})],
            stop_reason="toolUse",
        )
        if calls == 1
        else AssistantMessage(content=[TextContent(text="验证已通过，任务完成。")])
    )
    tools = _ToolPort(
        {
            "pytest": ToolObservation(
                tool_call_id="test_1",
                name="pytest",
                status="success",
                verification=(
                    RunVerification(
                        tool_call_id="test_1",
                        tool_name="pytest",
                        status="passed",
                        command="pytest",
                        exit_code=0,
                        summary="passed",
                    ),
                ),
            )
        }
    )

    outcome = await run_agent_loop(
        _loop_input("run_verify", limits=AgentLoopLimits(max_tool_iterations=1)),
        AgentLoopPorts(model=model, tools=tools),
    )

    assert outcome.status == "completed"
    assert model.calls == 2
    assert outcome.final_message is not None
    assert outcome.final_message.content[0].text == "验证已通过，任务完成。"
    assert outcome.task is not None
    assert outcome.task.completion_satisfied is True
    assert any(event["type"] == "completion_checked" for event in outcome.events)


def test_agent_loop_continues_after_task_control_complete_step() -> None:
    asyncio.run(_agent_loop_continues_after_task_control_complete_step())


async def _agent_loop_continues_after_task_control_complete_step() -> None:
    model = _ScriptedModel(
        lambda _request, calls: AssistantMessage(
            content=[
                ToolCall(
                    id="complete_1",
                    name="complete_task_step",
                    arguments={
                        "summary": "已读完上下文",
                        "evidence_refs": ["tool:read_1"],
                    },
                )
            ],
            stop_reason="toolUse",
        )
        if calls == 1
        else AssistantMessage(content=[TextContent(text="上下文已整理完成。")])
    )
    tools = _ToolPort(
        {
            "complete_task_step": ToolObservation(
                tool_call_id="complete_1",
                name="complete_task_step",
                status="success",
                content=(TextContent(text="Current task step completed: 已读完上下文"),),
                metadata={
                    "task_control": {
                        "action": "complete_step",
                        "summary": "已读完上下文",
                        "evidence_refs": ["tool:read_1"],
                        "tool_call_id": "complete_1",
                    }
                },
            )
        }
    )

    outcome = await run_agent_loop(
        _loop_input("run_task_control_complete", prompt="整理当前上下文"),
        AgentLoopPorts(model=model, tools=tools),
    )

    assert outcome.status == "completed"
    assert model.calls == 2
    assert tools.executed == ["complete_task_step"]
    assert outcome.final_message is not None
    assert outcome.final_message.content[0].text == "上下文已整理完成。"


def test_agent_loop_continues_after_read_tool_before_final_answer() -> None:
    asyncio.run(_agent_loop_continues_after_read_tool_before_final_answer())


async def _agent_loop_continues_after_read_tool_before_final_answer() -> None:
    model = _ScriptedModel(
        lambda _request, calls: AssistantMessage(
            content=[ToolCall(id="read_1", name="read", arguments={"path": "app.py"})],
            stop_reason="toolUse",
        )
        if calls == 1
        else AssistantMessage(content=[TextContent(text="已完成分析，准备修改。")])
    )
    tools = _ToolPort(
        {
            "read": ToolObservation(
                tool_call_id="read_1",
                name="read",
                status="success",
                content=(TextContent(text="source"),),
            )
        }
    )

    outcome = await run_agent_loop(
        _loop_input("run_read_continue", prompt="修复登录注册功能"),
        AgentLoopPorts(model=model, tools=tools),
    )

    assert outcome.status == "completed"
    assert model.calls == 2
    assert tools.executed == ["read"]
    assert outcome.final_message is not None
    assert outcome.final_message.content[0].text == "已完成分析，准备修改。"


def test_agent_loop_waits_for_verification_after_workspace_change() -> None:
    asyncio.run(_agent_loop_waits_for_verification_after_workspace_change())


async def _agent_loop_waits_for_verification_after_workspace_change() -> None:
    model = _ScriptedModel(
        lambda _request, calls: AssistantMessage(
            content=[ToolCall(id="edit_1", name="edit_file", arguments={})]
            if calls == 1
            else [TextContent(text="done")],
            stop_reason="toolUse" if calls == 1 else "stop",
        )
    )
    tools = _ToolPort(
        {
            "edit_file": ToolObservation(
                tool_call_id="edit_1",
                name="edit_file",
                status="success",
                affected_paths=("src/app.py",),
                workspace_changed=True,
            )
        }
    )

    outcome = await run_agent_loop(
        _loop_input("run_unverified", limits=AgentLoopLimits(max_model_turns=2)),
        AgentLoopPorts(model=model, tools=tools),
    )

    assert outcome.status == "waiting_user"
    assert outcome.stop_reason == "task_incomplete"
    assert outcome.task is not None
    assert outcome.task.completion_reason == "modified_without_fresh_verification"


def test_agent_loop_pauses_at_tool_iteration_limit() -> None:
    asyncio.run(_agent_loop_pauses_at_tool_iteration_limit())


async def _agent_loop_pauses_at_tool_iteration_limit() -> None:
    model = _ScriptedModel(
        lambda _request, calls: AssistantMessage(
            content=[ToolCall(id="edit_1", name="edit_file", arguments={})]
            if calls == 1
            else [ToolCall(id="test_1", name="pytest", arguments={})],
            stop_reason="toolUse",
        )
    )
    tools = _ToolPort(
        {
            "edit_file": ToolObservation(
                tool_call_id="edit_1",
                name="edit_file",
                status="success",
                affected_paths=("src/app.py",),
                workspace_changed=True,
            ),
            "pytest": ToolObservation(
                tool_call_id="test_1",
                name="pytest",
                status="success",
            ),
        }
    )

    outcome = await run_agent_loop(
        _loop_input(
            "run_limit",
            limits=AgentLoopLimits(max_model_turns=3, max_tool_iterations=1),
        ),
        AgentLoopPorts(model=model, tools=tools),
    )

    assert outcome.status == "waiting_user"
    assert outcome.stop_reason == "max_iterations"
    assert tools.executed == ["edit_file"]
