from __future__ import annotations

import asyncio
from typing import Any

import pytest

from codepilot.core.contracts import AgentLoopInput, AgentLoopLimits, AgentLoopPorts, RunCorrelation
from codepilot.core.plan import PlanState, PlanUpdate, PlanUpdateItem, PlanValidationError
from codepilot.core.run_guard import RunGuard
from codepilot.core.runner import run_agent_loop
from codepilot.llm.ports import LLMCompleted, ModelDescriptor
from codepilot.protocols import (
    AssistantMessage,
    RunSignalsSummary,
    RunVerification,
    TextContent,
    ToolCall,
    ToolResultMessage,
)
from codepilot.tools.contracts import ToolObservation


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

    def catalog(self, current_mode: str = "build"):
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
    mode: str = "build",
    plan_state: dict[str, object] | None = None,
) -> AgentLoopInput:
    return AgentLoopInput(
        run_id=run_id,
        correlation=RunCorrelation(session_id="s1"),
        user_prompt=prompt,
        model=ModelDescriptor(provider="fake", model_id="unit"),
        limits=limits or AgentLoopLimits(),
        mode=mode,  # type: ignore[arg-type]
        plan_state=plan_state,
    )


def test_plan_update_validates_shape() -> None:
    update = PlanUpdate(
        explanation="推进实现",
        items=(
            PlanUpdateItem(step="阅读相关实现", status="completed"),
            PlanUpdateItem(step="重构 runner", status="in_progress"),
        ),
    )
    assert update.items[1].status == "in_progress"

    with pytest.raises(PlanValidationError, match="at most one"):
        PlanUpdate(
            items=(
                PlanUpdateItem(step="a", status="in_progress"),
                PlanUpdateItem(step="b", status="in_progress"),
            )
        )


def test_plan_mode_update_stays_proposed_and_build_mode_update_becomes_active() -> None:
    plan_mode = PlanState.new(objective="制定方案", origin_mode="plan", run_id="run_1")
    proposed = plan_mode.apply_update(
        PlanUpdate(items=(PlanUpdateItem(step="给出方案", status="completed"),)),
        mode="plan",
        run_id="run_1",
    )
    assert proposed.status == "proposed"

    build_mode = PlanState.new(objective="实现方案", origin_mode="build", run_id="run_2")
    active = build_mode.apply_update(
        PlanUpdate(
            items=(
                PlanUpdateItem(step="阅读实现", status="completed"),
                PlanUpdateItem(step="修改代码", status="in_progress"),
            )
        ),
        mode="build",
        run_id="run_2",
    )
    assert active.status == "active"


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
    assert isinstance(model.requests[-1].messages[-1], ToolResultMessage)
    assert outcome.final_text == "已完成分析，准备修改。"


def test_update_plan_tool_result_updates_soft_plan_and_continues_to_model() -> None:
    asyncio.run(_update_plan_tool_result_updates_soft_plan_and_continues_to_model())


async def _update_plan_tool_result_updates_soft_plan_and_continues_to_model() -> None:
    model = _ScriptedModel(
        lambda _request, calls: AssistantMessage(
            content=[
                ToolCall(
                    id="plan_1",
                    name="update_plan",
                    arguments={
                        "explanation": "开始重构",
                        "plan": [
                            {"step": "阅读实现", "status": "completed"},
                            {"step": "重写 runner", "status": "in_progress"},
                        ],
                    },
                )
            ],
            stop_reason="toolUse",
        )
        if calls == 1
        else AssistantMessage(content=[TextContent(text="计划已更新，继续实现。")])
    )
    tools = _ToolPort(
        {
            "update_plan": ToolObservation(
                tool_call_id="plan_1",
                name="update_plan",
                status="success",
                content=(TextContent(text="Plan updated."),),
                metadata={
                    "plan_update": {
                        "explanation": "开始重构",
                        "plan": [
                            {"step": "阅读实现", "status": "completed"},
                            {"step": "重写 runner", "status": "in_progress"},
                        ],
                    }
                },
            )
        }
    )

    outcome = await run_agent_loop(
        _loop_input("run_update_plan", prompt="重构 runtime"),
        AgentLoopPorts(model=model, tools=tools),
    )

    assert outcome.status == "completed"
    assert model.calls == 2
    assert outcome.plan is not None
    assert outcome.plan.status == "active"
    assert outcome.plan.items[1]["status"] == "in_progress"
    assert any(event["type"] == "plan_updated" for event in outcome.events)


def test_plan_completed_does_not_complete_run_before_model_final_answer() -> None:
    asyncio.run(_plan_completed_does_not_complete_run_before_model_final_answer())


async def _plan_completed_does_not_complete_run_before_model_final_answer() -> None:
    model = _ScriptedModel(
        lambda _request, calls: AssistantMessage(
            content=[
                ToolCall(
                    id="plan_1",
                    name="update_plan",
                    arguments={"plan": [{"step": "总结", "status": "completed"}]},
                )
            ],
            stop_reason="toolUse",
        )
        if calls == 1
        else AssistantMessage(content=[TextContent(text="完成总结。")])
    )
    tools = _ToolPort(
        {
            "update_plan": ToolObservation(
                tool_call_id="plan_1",
                name="update_plan",
                status="success",
                metadata={
                    "plan_update": {
                        "plan": [{"step": "总结", "status": "completed"}],
                    }
                },
            )
        }
    )

    outcome = await run_agent_loop(
        _loop_input("run_plan_completed", prompt="总结"),
        AgentLoopPorts(model=model, tools=tools),
    )

    assert outcome.status == "completed"
    assert model.calls == 2
    assert outcome.plan is not None
    assert outcome.plan.status == "completed"
    assert any(event["type"] == "plan_completed" for event in outcome.events)


def test_run_guard_rejects_empty_final_answer() -> None:
    decision = RunGuard().check(
        assistant=AssistantMessage(content=[TextContent(text=" ")]),
        signals=RunSignalsSummary(),
        mode="build",
    )

    assert decision.action == "continue_with_instruction"
    assert decision.reason == "empty_final_answer"


def test_run_guard_stops_read_mode_workspace_change() -> None:
    decision = RunGuard().check(
        assistant=AssistantMessage(content=[TextContent(text="done")]),
        signals=RunSignalsSummary(
            workspace_changed=True,
            verification_status="passed",
        ),
        mode="read",
    )

    assert decision.action == "stopped"
    assert decision.reason == "read_mode_workspace_changed"


def test_run_guard_requires_verification_after_workspace_change() -> None:
    asyncio.run(_run_guard_requires_verification_after_workspace_change())


async def _run_guard_requires_verification_after_workspace_change() -> None:
    model = _ScriptedModel(
        lambda _request, calls: AssistantMessage(
            content=[ToolCall(id="edit_1", name="edit", arguments={})]
            if calls == 1
            else [TextContent(text="done")],
            stop_reason="toolUse" if calls == 1 else "stop",
        )
    )
    tools = _ToolPort(
        {
            "edit": ToolObservation(
                tool_call_id="edit_1",
                name="edit",
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
    assert outcome.stop_reason == "run_guard"
    assert outcome.signals.workspace_changed is True
    assert outcome.signals.verification_status == "stale"
    assert any(event["type"] == "run_guard_checked" for event in outcome.events)


def test_pytest_passed_still_returns_to_model_for_summary() -> None:
    asyncio.run(_pytest_passed_still_returns_to_model_for_summary())


async def _pytest_passed_still_returns_to_model_for_summary() -> None:
    model = _ScriptedModel(
        lambda _request, calls: AssistantMessage(
            content=[ToolCall(id="test_1", name="pytest", arguments={})],
            stop_reason="toolUse",
        )
        if calls == 1
        else AssistantMessage(content=[TextContent(text="测试已通过。")])
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
        _loop_input("run_pytest_passed"),
        AgentLoopPorts(model=model, tools=tools),
    )

    assert outcome.status == "completed"
    assert model.calls == 2
    assert outcome.final_text == "测试已通过。"
