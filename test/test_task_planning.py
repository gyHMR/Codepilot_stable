from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any


def test_task_controller_normalizes_steps_and_updates_from_tool_results() -> None:
    from codepilot.core.run_state import RunState
    from codepilot.core.task_controller import TaskController
    from codepilot.protocols import TextContent, ToolResultMessage, UserMessage

    controller = TaskController()
    task = controller.initialize(
        [UserMessage(content="修复配置加载失败并验证")],
        proposed_steps=[
            "定位配置加载调用链",
            "",
            "定位配置加载调用链",
            "修改实现",
            "运行相关测试",
            "多余步骤 1",
            "多余步骤 2",
            "多余步骤 3",
        ],
    )
    assert [step.title for step in task.steps] == [
        "定位配置加载调用链",
        "修改实现",
        "运行相关测试",
        "多余步骤 1",
        "多余步骤 2",
        "多余步骤 3",
    ]
    assert task.current_step_id == "step_1"

    read = ToolResultMessage(
        tool_call_id="read_1",
        tool_name="read",
        content=[TextContent(text="config.py lines")],
        status="success",
    )
    run = RunState(run_id="run_1", session_id="session_1")
    run.collect_tool_results([read])
    decision = controller.after_tool_results(task, run, [read])

    assert decision.action == "continue"
    assert task.steps[0].status == "completed"
    assert task.steps[0].evidence_refs == ["tool:read_1"]
    assert task.steps[1].status == "in_progress"

    failed_verification = ToolResultMessage(
        tool_call_id="test_1",
        tool_name="bash",
        status="error",
        is_error=True,
        verification={
            "status": "failed",
            "command": "python -m pytest test/test_config.py -q",
            "exit_code": 1,
            "summary": "failed",
        },
    )
    run.collect_tool_results([failed_verification])
    decision = controller.after_tool_results(task, run, [failed_verification])

    assert decision.action == "repair"
    assert task.steps[1].failure_count == 1
    assert task.steps[1].status == "in_progress"


def test_completion_gate_requires_fresh_verification_after_workspace_change() -> None:
    from codepilot.core.run_state import RunState
    from codepilot.core.task_controller import TaskController
    from codepilot.protocols import TextContent, ToolResultMessage, UserMessage

    controller = TaskController()
    task = controller.initialize(
        [UserMessage(content="修改实现并运行测试")],
        proposed_steps=["修改实现", "运行相关测试"],
    )
    run = RunState(run_id="run_1", session_id="session_1")

    edit = ToolResultMessage(
        tool_call_id="edit_1",
        tool_name="edit",
        content=[TextContent(text="Edited file")],
        affected_paths=["src/app.py"],
        workspace_changed=True,
        status="success",
    )
    run.collect_tool_results([edit])
    controller.after_tool_results(task, run, [edit])

    missing = controller.check_completion(task, run)
    assert missing.satisfied is False
    assert missing.reason == "modified_without_fresh_verification"
    assert missing.can_continue is True
    assert "fresh_verification" in missing.missing

    passed = ToolResultMessage(
        tool_call_id="test_1",
        tool_name="bash",
        status="success",
        verification={
            "status": "passed",
            "command": "python -m pytest test -q",
            "exit_code": 0,
            "summary": "passed",
        },
    )
    run.collect_tool_results([passed])
    controller.after_tool_results(task, run, [passed])

    ok = controller.check_completion(task, run)
    assert ok.satisfied is True
    assert ok.reason == "all_steps_completed"


def test_replan_preserves_completed_steps_and_stops_after_limit() -> None:
    from codepilot.core.run_state import RunState
    from codepilot.core.task_controller import TaskController
    from codepilot.protocols import ToolResultMessage, UserMessage

    controller = TaskController()
    task = controller.initialize(
        [UserMessage(content="修复失败测试")],
        proposed_steps=["定位失败", "修改实现", "运行验证"],
    )
    run = RunState(run_id="run_1", session_id="session_1")

    read = ToolResultMessage(
        tool_call_id="read_1",
        tool_name="read",
        status="success",
    )
    run.collect_tool_results([read])
    controller.after_tool_results(task, run, [read])
    assert task.steps[0].status == "completed"

    first = _failed_verification("test_1")
    second = _failed_verification("test_2")
    run.collect_tool_results([first])
    assert controller.after_tool_results(task, run, [first]).action == "repair"
    run.collect_tool_results([second])
    decision = controller.after_tool_results(task, run, [second])

    assert decision.action == "replan"
    assert decision.reason == "repeated_step_failure"
    assert task.replan_count == 1
    assert task.steps[0].title == "定位失败"
    assert task.steps[0].status == "completed"
    assert task.steps[1].title == "根据最新失败证据调整方案"
    assert task.steps[1].status == "in_progress"
    assert task.steps[2].title == "重新运行相关验证"

    for call_id in ["test_3", "test_4", "test_5", "test_6"]:
        failed = _failed_verification(call_id)
        run.collect_tool_results([failed])
        decision = controller.after_tool_results(task, run, [failed])

    assert decision.action == "stop"
    assert decision.reason == "replan_limit_exceeded"
    assert task.steps[1].status == "blocked"
    assert task.next_action == "报告连续失败并等待用户指示"


def test_task_controller_rebuilds_task_state_from_memory_projection() -> None:
    from codepilot.core.task_controller import TaskController
    from codepilot.protocols import UserMessage

    controller = TaskController()
    task = controller.initialize(
        [UserMessage(content="继续修复失败测试")],
        recovered_task={
            "goal": "修复失败测试",
            "task_progress": {
                "completed_steps": ["定位失败"],
                "pending_steps": ["重新运行相关验证"],
                "blocked_steps": ["根据最新失败证据调整方案"],
                "completion_satisfied": False,
                "completion_reason": "replan_limit_exceeded",
            },
            "next_action": "报告连续失败并等待用户指示",
        },
    )

    assert task.goal == "修复失败测试"
    assert [step.status for step in task.steps] == [
        "completed",
        "blocked",
        "in_progress",
    ]
    assert [step.title for step in task.steps] == [
        "定位失败",
        "根据最新失败证据调整方案",
        "重新运行相关验证",
    ]
    assert task.current_step_id == "step_3"
    assert task.next_action == "报告连续失败并等待用户指示"
    assert task.completion_reason == "replan_limit_exceeded"


def test_agent_loop_emits_task_events_and_result_summary() -> None:
    asyncio.run(_agent_loop_task_summary_case())


def test_agent_loop_uses_recovered_task_projection_in_context() -> None:
    asyncio.run(_agent_loop_recovered_task_context_case())


async def _agent_loop_recovered_task_context_case() -> None:
    from codepilot.core import AgentContext, AgentLoopConfig, run_agent_loop
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import AssistantMessage, Model, TextContent, UserMessage

    async def fake_stream(_model, context, _options):
        stream = AssistantMessageEventStream()
        system_prompt = context.system_prompt or ""
        assert "定位失败" in system_prompt
        assert "根据最新失败证据调整方案" in system_prompt
        assert "Next action: 报告连续失败并等待用户指示" in system_prompt
        stream.end(AssistantMessage(content=[TextContent(text="继续")]))
        return stream

    result = await run_agent_loop(
        prompts=[UserMessage(content="继续修复失败测试")],
        context=AgentContext(
            system_prompt="rules",
            messages=[],
            recovered_task={
                "goal": "修复失败测试",
                "task_progress": {
                    "completed_steps": ["定位失败"],
                    "pending_steps": ["重新运行相关验证"],
                    "blocked_steps": ["根据最新失败证据调整方案"],
                    "completion_satisfied": False,
                    "completion_reason": "replan_limit_exceeded",
                },
                "next_action": "报告连续失败并等待用户指示",
            },
        ),
        config=AgentLoopConfig(
            model=Model(
                id="task-test",
                name="Task Test",
                api="unit-test",
                provider="unit-test",
                base_url="",
                reasoning=False,
                input=["text"],
                context_window=4000,
                max_tokens=500,
            ),
            convert_to_llm=lambda items: items,
        ),
        emit=lambda _event: None,
        stream_fn=fake_stream,
    )

    assert result.task is not None
    assert result.task.completed_steps == ["定位失败"]
    assert result.task.pending_steps == ["重新运行相关验证"]
    assert result.task.blocked_steps == ["根据最新失败证据调整方案"]


def test_agent_loop_stops_when_replan_limit_is_exceeded() -> None:
    asyncio.run(_agent_loop_replan_limit_case())


async def _agent_loop_replan_limit_case() -> None:
    from codepilot.core import AgentContext, AgentLoopConfig, run_agent_loop
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import AssistantMessage, Model, ToolCall, UserMessage
    from codepilot.tools import AgentTool, AgentToolResult

    attempts = 0

    async def fake_stream(_model, _context, _options):
        nonlocal attempts
        attempts += 1
        stream = AssistantMessageEventStream()
        stream.end(
            AssistantMessage(
                content=[
                    ToolCall(
                        id=f"test_{attempts}",
                        name="bash_test",
                        arguments={"attempt": attempts},
                    )
                ],
                stop_reason="toolUse",
            )
        )
        return stream

    async def test_tool(*_args):
        return AgentToolResult(
            status="error",
            is_error=True,
            verification={
                "status": "failed",
                "command": "python -m pytest test/test_task.py -q",
                "exit_code": 1,
                "summary": "failed",
            },
        )

    events: list[dict[str, Any]] = []
    result = await run_agent_loop(
        prompts=[UserMessage(content="修复失败测试")],
        context=AgentContext(
            system_prompt="rules",
            messages=[],
            tools=[
                AgentTool(
                    name="bash_test",
                    label="bash",
                    description="bash",
                    parameters={},
                    execute=test_tool,
                )
            ],
        ),
        config=AgentLoopConfig(
            model=Model(
                id="task-test",
                name="Task Test",
                api="unit-test",
                provider="unit-test",
                base_url="",
                reasoning=False,
                input=["text"],
                context_window=4000,
                max_tokens=500,
            ),
            convert_to_llm=lambda items: items,
            allow_unmanaged_tools=True,
            repeated_tool_call_limit=20,
            max_tool_iterations=20,
        ),
        emit=events.append,
        stream_fn=fake_stream,
    )

    assert result.status == "failed"
    assert result.stop_reason == "replan_limit"
    assert result.task is not None
    assert result.task.blocked_steps == ["根据最新失败证据调整方案"]
    assert result.task.next_action == "报告连续失败并等待用户指示"
    assert any(
        event.get("type") == "task_decision"
        and event.get("decision", {}).get("reason") == "replan_limit_exceeded"
        for event in events
    )


async def _agent_loop_task_summary_case() -> None:
    from codepilot.core import AgentContext, AgentLoopConfig, run_agent_loop
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import (
        AssistantMessage,
        Model,
        TextContent,
        ToolCall,
        ToolResultMessage,
        UserMessage,
    )
    from codepilot.tools import AgentTool, AgentToolResult

    async def fake_stream(_model, context, _options):
        stream = AssistantMessageEventStream()
        assert "## Current Task" in (context.system_prompt or "")
        if any(isinstance(message, ToolResultMessage) for message in context.messages):
            stream.end(AssistantMessage(content=[TextContent(text="done")]))
        else:
            stream.end(
                AssistantMessage(
                    content=[ToolCall(id="read_1", name="read_test", arguments={})],
                    stop_reason="toolUse",
                )
            )
        return stream

    async def read_tool(*_args):
        return AgentToolResult(content=[TextContent(text="read result")])

    events: list[dict[str, Any]] = []
    result = await run_agent_loop(
        prompts=[UserMessage(content="解释这个文件")],
        context=AgentContext(
            system_prompt="rules",
            messages=[],
            tools=[
                AgentTool(
                    name="read_test",
                    label="read",
                    description="read",
                    parameters={},
                    execute=read_tool,
                )
            ],
        ),
        config=AgentLoopConfig(
            model=Model(
                id="task-test",
                name="Task Test",
                api="unit-test",
                provider="unit-test",
                base_url="",
                reasoning=False,
                input=["text"],
                context_window=4000,
                max_tokens=500,
            ),
            convert_to_llm=lambda items: items,
            allow_unmanaged_tools=True,
        ),
        emit=events.append,
        stream_fn=fake_stream,
    )

    assert result.task is not None
    assert result.task.goal == "解释这个文件"
    assert result.task.completed_steps == ["完成当前请求"]
    assert result.task.completion_satisfied is True
    assert any(event["type"] == "task_plan_created" for event in events)
    assert any(event["type"] == "task_step_updated" for event in events)
    assert events[-1]["result"].task is result.task


def _failed_verification(tool_call_id: str):
    from codepilot.protocols import ToolResultMessage

    return ToolResultMessage(
        tool_call_id=tool_call_id,
        tool_name="bash",
        status="error",
        is_error=True,
        verification={
            "status": "failed",
            "command": "python -m pytest test/test_task.py -q",
            "exit_code": 1,
            "summary": "failed",
        },
    )
