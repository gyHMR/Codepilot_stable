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


def _raise(message: str) -> Any:
    raise AssertionError(message)


def _plan_item(step: str, status: str) -> PlanUpdateItem:
    return PlanUpdateItem(
        step=step,
        details=f"执行：{step}",
        verification=f"验证：{step}",
        status=status,  # type: ignore[arg-type]
    )


def _plan_payload(*items: tuple[str, str], summary: str = "按可验证步骤推进。") -> dict[str, Any]:
    return {
        "summary": summary,
        "plan": [
            {
                "step": step,
                "details": f"执行：{step}",
                "verification": f"验证：{step}",
                "status": status,
            }
            for step, status in items
        ],
    }


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
        summary="阅读后重构 runner。",
        explanation="推进实现",
        items=(
            _plan_item("阅读相关实现", "completed"),
            _plan_item("重构 runner", "in_progress"),
        ),
    )
    assert update.items[1].status == "in_progress"

    with pytest.raises(PlanValidationError, match="at most one"):
        PlanUpdate(
            summary="并行执行两个步骤。",
            items=(
                _plan_item("a", "in_progress"),
                _plan_item("b", "in_progress"),
            )
        )


def test_plan_mode_update_stays_proposed_and_build_mode_update_becomes_active() -> None:
    plan_mode = PlanState.new(objective="制定方案", origin_mode="plan", run_id="run_1")
    proposed = plan_mode.apply_update(
        PlanUpdate(
            summary="阅读后给出方案。",
            items=(
                _plan_item("阅读实现", "completed"),
                _plan_item("给出方案", "in_progress"),
            )
        ),
        mode="plan",
        run_id="run_1",
    )
    assert proposed.status == "proposed"
    assert proposed.approval_state == "pending"
    assert [item.status for item in proposed.items] == ["pending", "pending"]

    from_mapping = PlanUpdate.from_mapping(
        {
            "summary": "测试 proposal 归一化。",
            "plan": [
                {
                    "step": "a",
                    "details": "执行 a",
                    "verification": "验证 a",
                    "status": "in_progress",
                },
                {
                    "step": "b",
                    "details": "执行 b",
                    "verification": "验证 b",
                    "status": "in_progress",
                },
            ]
        },
        proposal=True,
    )
    assert [item.status for item in from_mapping.items] == ["pending", "pending"]

    build_mode = PlanState.new(objective="实现方案", origin_mode="build", run_id="run_2")
    active = build_mode.apply_update(
        PlanUpdate(
            summary="阅读后修改代码。",
            items=(
                _plan_item("阅读实现", "completed"),
                _plan_item("修改代码", "in_progress"),
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
                        **_plan_payload(
                            ("阅读实现", "completed"),
                            ("重写 runner", "completed"),
                        ),
                        "explanation": "开始重构",
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
                            **_plan_payload(
                                ("阅读实现", "completed"),
                                ("重写 runner", "completed"),
                            ),
                            "explanation": "开始重构",
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
    assert outcome.plan.approval_state == "not_required"
    assert outcome.plan.items[1]["status"] == "completed"
    assert any(event["type"] == "plan_updated" for event in outcome.events)


def test_active_plan_items_do_not_block_final_answer() -> None:
    asyncio.run(_active_plan_items_do_not_block_final_answer())


async def _active_plan_items_do_not_block_final_answer() -> None:
    active_plan = PlanState.new(
        objective="重构 runtime",
        origin_mode="build",
        run_id="run_plan_completion_guard",
    ).apply_update(
        PlanUpdate(
            summary="阅读后重写 runner。",
            items=(
                _plan_item("阅读实现", "completed"),
                _plan_item("重写 runner", "in_progress"),
            )
        ),
        mode="build",
        run_id="run_plan_completion_guard",
    )
    model = _ScriptedModel(
        lambda _request, _calls: AssistantMessage(
            content=[TextContent(text="已经完成。")]
        )
    )
    tools = _ToolPort({})

    outcome = await run_agent_loop(
        _loop_input(
            "run_plan_completion_guard",
            prompt="重构 runtime",
            plan_state=active_plan.to_dict(),
        ),
        AgentLoopPorts(model=model, tools=tools),
    )

    assert outcome.status == "completed"
    assert model.calls == 1
    assert outcome.plan is not None
    assert outcome.plan.status == "active"
    assert not any(
        event["type"] == "run_guard_checked"
        and event["decision"]["reason"] == "plan_completion_missing"
        for event in outcome.events
    )


def test_plan_mode_update_plan_pauses_for_user_approval() -> None:
    asyncio.run(_plan_mode_update_plan_pauses_for_user_approval())


async def _plan_mode_update_plan_pauses_for_user_approval() -> None:
    model = _ScriptedModel(
        lambda _request, calls: AssistantMessage(
            content=[
                ToolCall(
                        id="plan_1",
                        name="update_plan",
                        arguments=_plan_payload(
                            ("阅读注册逻辑", "completed"),
                            ("提出重构方案", "in_progress"),
                            summary="重构注册逻辑并补充验证。",
                        ),
                )
            ],
            stop_reason="toolUse",
        )
        if calls == 1
        else AssistantMessage(
            content=[TextContent(text="方案已经整理完毕，请审批。")]
        )
    )
    tools = _ToolPort(
        {
            "update_plan": ToolObservation(
                tool_call_id="plan_1",
                name="update_plan",
                status="success",
                content=(TextContent(text="Plan updated."),),
                    metadata={
                    "plan_update": _plan_payload(
                        ("阅读注册逻辑", "completed"),
                        ("提出重构方案", "in_progress"),
                        summary="重构注册逻辑并补充验证。",
                    )
                },
            )
        }
    )

    outcome = await run_agent_loop(
        _loop_input("run_plan_pause", prompt="先给我方案", mode="plan"),
        AgentLoopPorts(model=model, tools=tools),
    )

    assert outcome.status == "waiting_user"
    assert outcome.stop_reason == "plan_approval_required"
    assert model.calls == 2
    assert tools.executed == ["update_plan"]
    assert outcome.final_text == "方案已经整理完毕，请审批。"
    assert outcome.plan is not None
    assert outcome.plan.status == "proposed"
    assert outcome.plan.approval_state == "pending"
    assert [item["status"] for item in outcome.plan.items] == ["pending", "pending"]
    event_types = [event["type"] for event in outcome.events]
    assert "plan_proposed" in event_types
    assert "plan_approval_required" in event_types


def test_plan_mode_question_pauses_same_run_for_clarification() -> None:
    async def run_case() -> None:
        model = _ScriptedModel(
            lambda _request, _calls: AssistantMessage(
                content=[TextContent(text="你希望保留旧命令兼容吗？")]
            )
        )

        outcome = await run_agent_loop(
            _loop_input(
                "run_plan_clarification",
                prompt="先设计重构方案",
                mode="plan",
            ),
            AgentLoopPorts(model=model, tools=_ToolPort({})),
        )

        assert outcome.status == "waiting_user"
        assert outcome.stop_reason == "plan_clarification_required"
        assert outcome.run_id == "run_plan_clarification"
        assert outcome.final_text == "你希望保留旧命令兼容吗？"

    asyncio.run(run_case())


def test_plan_summary_model_failure_uses_canonical_fallback() -> None:
    async def run_case() -> None:
        class SummaryFailureModel:
            def __init__(self) -> None:
                self.calls = 0

            async def stream(self, _request):
                self.calls += 1
                if self.calls == 1:
                    yield LLMCompleted(
                        message=AssistantMessage(
                            content=[
                                ToolCall(
                                    id="plan_1",
                                    name="update_plan",
                                    arguments=_plan_payload(
                                        ("阅读实现", "pending"),
                                        ("执行修改", "pending"),
                                        summary="阅读实现后完成聚焦修改。",
                                    ),
                                )
                            ],
                            stop_reason="toolUse",
                        )
                    )
                    return
                raise RuntimeError("summary model unavailable")

        tools = _ToolPort(
            {
                "update_plan": ToolObservation(
                    tool_call_id="plan_1",
                    name="update_plan",
                    status="success",
                    metadata={
                        "plan_update": _plan_payload(
                            ("阅读实现", "pending"),
                            ("执行修改", "pending"),
                            summary="阅读实现后完成聚焦修改。",
                        )
                    },
                )
            }
        )
        outcome = await run_agent_loop(
            _loop_input("run_plan_fallback", prompt="先给计划", mode="plan"),
            AgentLoopPorts(model=SummaryFailureModel(), tools=tools),
        )

        assert outcome.status == "waiting_user"
        assert outcome.stop_reason == "plan_approval_required"
        assert "阅读实现后完成聚焦修改" in outcome.final_text
        assert outcome.final_message is not None
        assert outcome.final_message.metadata["summary_fallback"] is True
        assert outcome.final_message.metadata["message_kind"] == "plan_summary"

    asyncio.run(run_case())


def test_plan_completed_does_not_complete_run_before_model_final_answer() -> None:
    asyncio.run(_plan_completed_does_not_complete_run_before_model_final_answer())


async def _plan_completed_does_not_complete_run_before_model_final_answer() -> None:
    model = _ScriptedModel(
        lambda _request, calls: AssistantMessage(
            content=[
                ToolCall(
                    id="plan_1",
                    name="update_plan",
                    arguments=_plan_payload(("总结", "completed")),
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
                    "plan_update": _plan_payload(("总结", "completed"))
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
    assert outcome.plan.status == "active"
    assert any(event["type"] == "plan_updated" for event in outcome.events)


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
