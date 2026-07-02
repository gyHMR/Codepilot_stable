from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _model():
    from codepilot.protocols import Model

    return Model(
        id="run-test",
        name="Run Test",
        api="unit-test",
        provider="unit-test",
        base_url="",
        reasoning=False,
        input=["text"],
        context_window=1000,
        max_tokens=100,
    )


def _managed_tool(name: str, execute):
    from codepilot.tools import AgentTool, ToolMetadata, ToolRegistry, ToolRuntime

    registry = ToolRegistry()
    registry.register(
        AgentTool(
            name=name,
            label=name,
            description=name,
            parameters={},
            execute=execute,
        ),
        metadata=ToolMetadata(
            name=name,
            category="test",
            read_only=False,
            concurrency_safe=False,
            exclusive=True,
            requires_approval=False,
            risk_level="low",
            resource_scope=("workspace",),
        ),
    )
    return ToolRuntime(registry).as_agent_tools()[0]


def test_run_result_collects_counters_changes_and_verification() -> None:
    asyncio.run(_run_result_case())


async def _run_result_case() -> None:
    from codepilot.core import AgentContext, AgentLoopConfig, run_agent_loop
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import (
        AssistantMessage,
        TextContent,
        ToolCall,
        ToolResultMessage,
        UserMessage,
    )
    from codepilot.tools import AgentToolResult

    async def fake_stream(_model, context, _options):
        stream = AssistantMessageEventStream()
        if any(isinstance(message, ToolResultMessage) for message in context.messages):
            stream.end(AssistantMessage(content=[TextContent(text="finished")]))
        else:
            stream.end(
                AssistantMessage(
                    content=[ToolCall(id="call_edit", name="edit", arguments={})],
                    stop_reason="toolUse",
                )
            )
        return stream

    async def edit_tool(*_args):
        return AgentToolResult(
            content=[TextContent(text="edited")],
            affected_paths=["src/example.py"],
            workspace_changed=True,
            diff_summary="one replacement",
            verification={
                "status": "passed",
                "command": "python -m compileall src",
                "exit_code": 0,
                "summary": "compiled",
            },
        )

    events: list[dict[str, Any]] = []
    result = await run_agent_loop(
        prompts=[UserMessage(content="edit and verify")],
        context=AgentContext(
            system_prompt="",
            messages=[],
            tools=[_managed_tool("edit", edit_tool)],
        ),
        config=AgentLoopConfig(model=_model(), convert_to_llm=lambda items: items),
        emit=events.append,
        stream_fn=fake_stream,
    )

    assert result.status == "completed"
    assert result.stop_reason == "final_answer"
    assert result.counters.model_attempts == 1
    assert result.counters.tool_iterations == 1
    assert result.counters.tool_calls == 1
    assert result.affected_paths == ["src/example.py"]
    assert result.workspace_changed
    assert result.verification[0].status == "passed"
    agent_end = events[-1]
    assert agent_end["runId"] == result.run_id
    assert agent_end["result"] is result


def test_run_stops_on_waiting_approval_and_repeated_calls() -> None:
    asyncio.run(_run_stop_cases())


async def _run_stop_cases() -> None:
    from codepilot.core import AgentContext, AgentLoopConfig, run_agent_loop
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import AssistantMessage, TextContent, ToolCall, UserMessage
    from codepilot.tools import AgentToolResult

    async def approval_stream(_model, _context, _options):
        stream = AssistantMessageEventStream()
        stream.end(
            AssistantMessage(
                content=[ToolCall(id="call_approval", name="deploy", arguments={})],
                stop_reason="toolUse",
            )
        )
        return stream

    async def approval_tool(*_args):
        return AgentToolResult(
            content=[TextContent(text="approval needed")],
            status="approval_required",
            approved=False,
            error_code="approval_required",
        )

    approval = await run_agent_loop(
        prompts=[UserMessage(content="deploy")],
        context=AgentContext(
            system_prompt="",
            messages=[],
            tools=[_managed_tool("deploy", approval_tool)],
        ),
        config=AgentLoopConfig(model=_model(), convert_to_llm=lambda items: items),
        emit=lambda _event: None,
        stream_fn=approval_stream,
    )
    assert approval.status == "waiting_approval"
    assert approval.stop_reason == "approval_required"

    executions = 0

    async def repeated_tool(*_args):
        nonlocal executions
        executions += 1
        return AgentToolResult(content=[TextContent(text="same")])

    async def repeated_stream(_model, _context, _options):
        stream = AssistantMessageEventStream()
        stream.end(
            AssistantMessage(
                content=[ToolCall(id="call_repeat", name="repeat", arguments={})],
                stop_reason="toolUse",
            )
        )
        return stream

    repeated = await run_agent_loop(
        prompts=[UserMessage(content="repeat")],
        context=AgentContext(
            system_prompt="",
            messages=[],
            tools=[_managed_tool("repeat", repeated_tool)],
        ),
        config=AgentLoopConfig(
            model=_model(),
            convert_to_llm=lambda items: items,
            repeated_tool_call_limit=1,
        ),
        emit=lambda _event: None,
        stream_fn=repeated_stream,
    )
    assert repeated.status == "failed"
    assert repeated.stop_reason == "repeated_tool_call"
    assert executions == 1


def test_retryable_model_error_remains_inside_one_run() -> None:
    asyncio.run(_run_retry_case())


def test_model_retry_decision_explains_retry_and_stop_cases() -> None:
    from codepilot.core import AgentLoopConfig
    from codepilot.core.run_decisions import decide_model_retry
    from codepilot.protocols import ErrorInfo

    retryable = ErrorInfo(
        code="llm.rate_limit",
        message="rate limited",
        retryable=True,
        source="llm",
    )
    permanent = ErrorInfo(
        code="llm.bad_request",
        message="bad request",
        retryable=False,
        source="llm",
    )
    config = AgentLoopConfig(
        model=_model(),
        convert_to_llm=lambda items: items,
        retry_base_delay_ms=100,
        max_model_retries=2,
    )

    first = decide_model_retry(retryable, retries_so_far=0, config=config)
    second = decide_model_retry(retryable, retries_so_far=1, config=config)
    exhausted = decide_model_retry(retryable, retries_so_far=2, config=config)
    disabled = decide_model_retry(
        retryable,
        retries_so_far=0,
        config=AgentLoopConfig(
            model=_model(),
            convert_to_llm=lambda items: items,
            retry_enabled=False,
        ),
    )

    assert first.should_retry is True
    assert first.next_retry_count == 1
    assert first.delay_ms == 100
    assert second.should_retry is True
    assert second.next_retry_count == 2
    assert second.delay_ms == 200
    assert exhausted.should_retry is False
    assert exhausted.reason == "retry_limit_exhausted"
    assert disabled.reason == "retry_disabled"
    assert decide_model_retry(permanent, retries_so_far=0, config=config).reason == (
        "error_not_retryable"
    )


def test_tool_execution_gate_explains_loop_stop_cases() -> None:
    from codepilot.core import AgentLoopConfig, RunState
    from codepilot.core.run_decisions import decide_tool_execution_gate
    from codepilot.protocols import ToolCall

    config = AgentLoopConfig(
        model=_model(),
        convert_to_llm=lambda items: items,
        repeated_tool_call_limit=1,
        max_tool_iterations=2,
    )
    first_call = [ToolCall(id="read_1", name="read", arguments={"path": "a.py"})]
    repeated_call = [ToolCall(id="read_2", name="read", arguments={"path": "a.py"})]

    repeated_state = RunState(run_id="run_1", session_id="session_1")
    assert decide_tool_execution_gate(first_call, repeated_state, config).should_execute
    repeated = decide_tool_execution_gate(repeated_call, repeated_state, config)

    maxed_state = RunState(run_id="run_2", session_id="session_1")
    maxed_state.counters.tool_iterations = 2
    maxed = decide_tool_execution_gate(first_call, maxed_state, config)

    assert repeated.should_execute is False
    assert repeated.stop_reason == "repeated_tool_call"
    assert repeated.error_code == "run.repeated_tool_call"
    assert maxed.should_execute is False
    assert maxed.stop_reason == "max_iterations"
    assert maxed.assistant_stop_reason == "max_iterations"
    assert maxed.error_code == "run.max_iterations"


def test_post_tool_run_decision_explains_pause_and_stop_cases() -> None:
    from codepilot.core.run_decisions import decide_post_tool_run
    from codepilot.core import ExecutionDecision
    from codepilot.protocols import ToolResultMessage

    approval = decide_post_tool_run(
        [
            ToolResultMessage(
                tool_call_id="tool_1",
                tool_name="write",
                status="approval_required",
            )
        ],
        task_decision=ExecutionDecision("wait_approval", "approval_required"),
    )
    cancelled = decide_post_tool_run(
        [
            ToolResultMessage(
                tool_call_id="tool_2",
                tool_name="bash",
                status="cancelled",
            )
        ],
        task_decision=ExecutionDecision("stop", "cancelled"),
    )
    revert = decide_post_tool_run(
        [],
        task_decision=ExecutionDecision("propose_revert", "repeated_failure_after_change"),
    )
    replan_limit = decide_post_tool_run(
        [],
        task_decision=ExecutionDecision("stop", "replan_limit_exceeded"),
    )
    task_blocked = decide_post_tool_run(
        [],
        task_decision=ExecutionDecision("stop", "tool_unavailable"),
    )
    finished = decide_post_tool_run(
        [],
        task_decision=ExecutionDecision("finish", "all_steps_completed"),
    )
    keep_going = decide_post_tool_run(
        [],
        task_decision=ExecutionDecision("continue", "next_step"),
    )

    assert approval.should_stop is True
    assert approval.status == "waiting_approval"
    assert approval.stop_reason == "approval_required"
    assert cancelled.status == "aborted"
    assert cancelled.stop_reason == "aborted"
    assert revert.status == "waiting_user"
    assert revert.stop_reason == "task_blocked"
    assert replan_limit.status == "failed"
    assert replan_limit.error_code == "run.replan_limit"
    assert replan_limit.stop_reason == "replan_limit"
    assert task_blocked.status == "waiting_user"
    assert task_blocked.stop_reason == "task_blocked"
    assert finished.should_stop is False
    assert finished.force_completion_check is True
    assert finished.reason == "finish"
    assert keep_going.should_stop is False
    assert keep_going.reason == "continue"


def test_task_finish_decision_exits_tool_loop_for_completion_check() -> None:
    asyncio.run(_task_finish_exits_tool_loop_case())


async def _task_finish_exits_tool_loop_case() -> None:
    from codepilot.core import AgentContext, AgentLoopConfig, run_agent_loop
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import AssistantMessage, TextContent, ToolCall, UserMessage
    from codepilot.tools import AgentTool, AgentToolResult

    model_calls = 0

    async def fake_stream(_model, _context, _options):
        nonlocal model_calls
        model_calls += 1
        stream = AssistantMessageEventStream()
        stream.end(
            AssistantMessage(
                content=[ToolCall(id=f"test_{model_calls}", name="run_tests", arguments={})],
                stop_reason="toolUse",
            )
        )
        return stream

    async def run_tests(*_args):
        return AgentToolResult(
            content=[TextContent(text="tests passed")],
            verification={
                "status": "passed",
                "command": "python -m pytest test -q",
                "exit_code": 0,
                "summary": "passed",
            },
        )

    events: list[dict[str, Any]] = []
    result = await run_agent_loop(
        prompts=[UserMessage(content="运行验证")],
        context=AgentContext(
            system_prompt="",
            messages=[],
            tools=[
                AgentTool(
                    name="run_tests",
                    label="Run tests",
                    description="Run tests",
                    parameters={},
                    execute=run_tests,
                )
            ],
        ),
        config=AgentLoopConfig(
            model=_model(),
            convert_to_llm=lambda items: items,
            allow_unmanaged_tools=True,
            max_tool_iterations=1,
            repeated_tool_call_limit=20,
        ),
        emit=events.append,
        stream_fn=fake_stream,
    )

    assert result.status == "completed"
    assert result.stop_reason == "final_answer"
    assert model_calls == 1
    assert any(event.get("type") == "completion_checked" for event in events)


def test_completion_run_decision_explains_continue_wait_and_done_cases() -> None:
    from codepilot.core.run_decisions import decide_completion_run
    from codepilot.core import CompletionCheck

    needs_more_work = decide_completion_run(
        CompletionCheck(
            satisfied=False,
            reason="modified_without_fresh_verification",
            missing=["fresh_verification"],
            can_continue=True,
        )
    )
    blocked = decide_completion_run(
        CompletionCheck(
            satisfied=False,
            reason="blocked_steps",
            missing=["unblocked_steps"],
        )
    )
    incomplete = decide_completion_run(
        CompletionCheck(
            satisfied=False,
            reason="incomplete_steps",
            missing=["运行测试"],
        )
    )
    done = decide_completion_run(
        CompletionCheck(satisfied=True, reason="all_steps_completed")
    )

    assert needs_more_work.action == "continue_with_steering"
    assert needs_more_work.should_stop is False
    assert blocked.action == "stop"
    assert blocked.status == "waiting_user"
    assert blocked.stop_reason == "task_blocked"
    assert incomplete.action == "stop"
    assert incomplete.stop_reason == "task_incomplete"
    assert done.action == "satisfied"
    assert done.should_stop is False


async def _run_retry_case() -> None:
    from codepilot.core import AgentContext, AgentLoopConfig, run_agent_loop
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import (
        AssistantMessage,
        LLMErrorInfo,
        TextContent,
        UserMessage,
    )

    attempts = 0

    async def retry_stream(_model, _context, _options):
        nonlocal attempts
        attempts += 1
        stream = AssistantMessageEventStream()
        if attempts == 1:
            info = LLMErrorInfo(
                code="llm.rate_limit",
                message="rate limited",
                retryable=True,
                kind="rate_limit",
            )
            stream.end(
                AssistantMessage(
                    stop_reason="error",
                    error_message=info.message,
                    error_info=info,
                )
            )
        else:
            stream.end(AssistantMessage(content=[TextContent(text="recovered")]))
        return stream

    events: list[dict[str, Any]] = []
    result = await run_agent_loop(
        prompts=[UserMessage(content="retry")],
        context=AgentContext(system_prompt="", messages=[]),
        config=AgentLoopConfig(
            model=_model(),
            convert_to_llm=lambda items: items,
            retry_base_delay_ms=1,
        ),
        emit=events.append,
        stream_fn=retry_stream,
    )

    assert result.status == "completed"
    assert result.counters.model_attempts == 2
    assert attempts == 2
    run_ids = {event["runId"] for event in events}
    assert run_ids == {result.run_id}
    assert any(event["type"] == "model_retry_start" for event in events)


def test_builtin_file_and_shell_results_are_structured(tmp_path: Path, monkeypatch) -> None:
    asyncio.run(_run_builtin_result_case(tmp_path, monkeypatch))


async def _run_builtin_result_case(tmp_path: Path, monkeypatch) -> None:
    from codepilot.tools.builtin import create_builtin_tools, get_builtin_tool_metadata
    from codepilot.tools.registry import ToolRegistry
    from codepilot.tools.runtime import ToolRuntime
    from codepilot.tools.permissions import PermissionPolicy

    return_codes = iter([3, 0])

    class FakeProcess:
        def __init__(self, return_code: int) -> None:
            self.returncode = return_code

        async def communicate(self):
            return b"output", b""

        def kill(self) -> None:
            self.returncode = -1

    async def fake_subprocess(*_args, **_kwargs):
        return FakeProcess(next(return_codes))

    monkeypatch.setattr(asyncio, "create_subprocess_shell", fake_subprocess)

    registry = ToolRegistry()
    for tool in create_builtin_tools(tmp_path):
        registry.register(tool, metadata=get_builtin_tool_metadata(tool.name))
    tools = {
        tool.name: tool
        for tool in ToolRuntime(
            registry,
            permission_policy=PermissionPolicy(
                bash_allow_patterns=[r"^python -c"],
            ),
        ).as_agent_tools()
    }

    write = await tools["write"].execute(
        "write_1",
        {"path": "hello.txt", "content": "hello"},
    )
    assert write.workspace_changed is True
    assert write.affected_paths == ["hello.txt"]
    assert write.diff_summary
    assert write.metadata["file_state"]["path"] == "hello.txt"
    assert isinstance(write.metadata["file_state"]["sha256"], str)

    unchanged = await tools["write"].execute(
        "write_2",
        {"path": "hello.txt", "content": "hello"},
    )
    assert unchanged.workspace_changed is False
    assert unchanged.metadata["file_state"]["path"] == "hello.txt"

    read = await tools["read"].execute(
        "read_1",
        {"path": "hello.txt"},
    )
    assert read.metadata["file_state"]["path"] == "hello.txt"

    shell = await tools["bash"].execute(
        "bash_1",
        {"command": 'python -c "import sys; sys.exit(3)"'},
    )
    assert shell.status == "error"
    assert shell.error_code == "shell_exit_nonzero"
    assert shell.exit_code == 3

    verification = await tools["bash"].execute(
        "bash_2",
        {"command": "python -m pytest -q"},
    )
    assert verification.status == "success"
    assert verification.verification is not None
    assert verification.verification["status"] == "passed"
