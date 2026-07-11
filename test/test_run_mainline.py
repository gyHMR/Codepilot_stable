from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_run_result_collects_counters_changes_and_verification() -> None:
    asyncio.run(_run_result_case())


async def _run_result_case() -> None:
    from codepilot.core.contracts import AgentLoopPorts
    from codepilot.core.runner import run_agent_loop
    from codepilot.protocols import AssistantMessage, RunVerification, TextContent, ToolCall, ToolResultMessage
    from codepilot.tools.contracts import ToolObservation

    model = _ScriptedModel(
        lambda request, calls: AssistantMessage(
            content=[TextContent(text="finished")]
            if any(isinstance(message, ToolResultMessage) for message in request.messages)
            else [ToolCall(id="call_edit", name="edit", arguments={})],
            stop_reason="stop" if calls > 1 else "toolUse",
        )
    )
    tools = _StaticToolPort(
        ToolObservation(
            tool_call_id="call_edit",
            name="edit",
            status="success",
            content=(TextContent(text="edited"),),
            affected_paths=("src/example.py",),
            workspace_changed=True,
            verification=(
                RunVerification(
                    tool_call_id="call_edit",
                    tool_name="edit",
                    status="passed",
                    command="python -m compileall src",
                    exit_code=0,
                    summary="compiled",
                ),
            ),
        )
    )

    outcome = await run_agent_loop(
        _loop_input("run_result", prompt="edit and verify"),
        AgentLoopPorts(model=model, tools=tools),
    )

    assert outcome.status == "completed"
    assert outcome.stop_reason == "final_answer"
    assert outcome.counters.model_attempts == 2
    assert outcome.counters.tool_iterations == 1
    assert outcome.counters.tool_calls == 1
    assert outcome.workspace_effects.affected_paths == ("src/example.py",)
    assert outcome.workspace_effects.changed
    assert outcome.verification[0].status == "passed"
    assert outcome.events[-1]["runId"] == outcome.run_id


def test_run_stops_on_waiting_approval_and_repeated_calls() -> None:
    asyncio.run(_run_stop_cases())


async def _run_stop_cases() -> None:
    from codepilot.core.contracts import AgentLoopLimits, AgentLoopPorts
    from codepilot.core.runner import run_agent_loop
    from codepilot.protocols import AssistantMessage, TextContent, ToolCall
    from codepilot.tools.contracts import ToolInterruption, ToolObservation, ToolRiskView

    approval_model = _ScriptedModel(
        lambda _request, _calls: AssistantMessage(
            content=[ToolCall(id="call_approval", name="deploy", arguments={})],
            stop_reason="toolUse",
        )
    )
    approval_tools = _StaticToolPort(
        ToolObservation(
            tool_call_id="call_approval",
            name="deploy",
            status="approval_required",
            interruption=ToolInterruption(
                approval_id="approval_1",
                run_id="run_approval",
                tool_call_id="call_approval",
                tool_name="deploy",
                reason="approval needed",
                risk=ToolRiskView(level="high", summary="deploy"),
            ),
            metadata={"approval_id": "approval_1"},
        )
    )

    approval = await run_agent_loop(
        _loop_input("run_approval", prompt="deploy"),
        AgentLoopPorts(model=approval_model, tools=approval_tools),
    )
    assert approval.status == "waiting_approval"
    assert approval.stop_reason == "approval_required"

    repeated_tools = _CountingToolPort(
        ToolObservation(
            tool_call_id="call_repeat",
            name="repeat",
            status="success",
            content=(TextContent(text="same"),),
        )
    )
    repeated = await run_agent_loop(
        _loop_input(
            "run_repeat",
            prompt="repeat",
            limits=AgentLoopLimits(
                max_model_turns=3,
                repeated_tool_call_limit=1,
            ),
        ),
        AgentLoopPorts(
            model=_ScriptedModel(
                lambda _request, _calls: AssistantMessage(
                    content=[ToolCall(id="call_repeat", name="repeat", arguments={})],
                    stop_reason="toolUse",
                )
            ),
            tools=repeated_tools,
        ),
    )
    assert repeated.status == "failed"
    assert repeated.stop_reason == "repeated_tool_call"
    assert repeated_tools.executions == 1


def test_retryable_model_error_remains_inside_one_run() -> None:
    asyncio.run(_run_retry_case())


async def _run_retry_case() -> None:
    from codepilot.core.contracts import AgentLoopLimits, AgentLoopPorts, RetryPolicy
    from codepilot.core.runner import run_agent_loop
    from codepilot.llm.ports import LLMCompleted, LLMFailed
    from codepilot.protocols import AssistantMessage, ErrorInfo, TextContent

    class RetryModel:
        def __init__(self) -> None:
            self.attempts = 0

        async def stream(self, _request):
            self.attempts += 1
            if self.attempts == 1:
                yield LLMFailed(
                    error=ErrorInfo(
                        code="llm.rate_limit",
                        message="rate limited",
                        retryable=True,
                        source="llm",
                    )
                )
                return
            yield LLMCompleted(
                message=AssistantMessage(content=[TextContent(text="recovered")])
            )

    model = RetryModel()
    outcome = await run_agent_loop(
        _loop_input(
            "run_retry",
            prompt="retry",
            limits=AgentLoopLimits(max_model_turns=1),
            retry_policy=RetryPolicy(enabled=True, max_retries=1, base_delay_ms=0),
        ),
        AgentLoopPorts(model=model, tools=None),
    )

    assert outcome.status == "completed"
    assert outcome.counters.model_attempts == 2
    assert model.attempts == 2
    run_ids = {event["runId"] for event in outcome.events}
    assert run_ids == {outcome.run_id}
    assert any(event["type"] == "model_retry_start" for event in outcome.events)


def test_passed_verification_returns_to_model_before_completion_check() -> None:
    asyncio.run(_passed_verification_returns_to_model_case())


async def _passed_verification_returns_to_model_case() -> None:
    from codepilot.core.contracts import AgentLoopLimits, AgentLoopPorts
    from codepilot.core.runner import run_agent_loop
    from codepilot.protocols import AssistantMessage, RunVerification, TextContent, ToolCall
    from codepilot.tools.contracts import ToolObservation

    model = _ScriptedModel(
        lambda _request, calls: AssistantMessage(
            content=[ToolCall(id=f"test_{calls}", name="run_tests", arguments={})],
            stop_reason="toolUse",
        )
        if calls == 1
        else AssistantMessage(content=[TextContent(text="verified done")])
    )
    tools = _StaticToolPort(
        ToolObservation(
            tool_call_id="test_1",
            name="run_tests",
            status="success",
            verification=(
                RunVerification(
                    tool_call_id="test_1",
                    tool_name="run_tests",
                    status="passed",
                    command="python -m pytest test -q",
                    exit_code=0,
                    summary="passed",
                ),
            ),
        )
    )

    outcome = await run_agent_loop(
        _loop_input(
            "run_finish",
            prompt="运行验证",
            limits=AgentLoopLimits(max_tool_iterations=1, repeated_tool_call_limit=20),
        ),
        AgentLoopPorts(model=model, tools=tools),
    )

    assert outcome.status == "completed"
    assert outcome.stop_reason == "final_answer"
    assert model.calls == 2
    assert outcome.final_text == "verified done"
    assert any(event.get("type") == "run_guard_checked" for event in outcome.events)


def test_builtin_file_and_shell_results_are_structured(tmp_path: Path, monkeypatch) -> None:
    asyncio.run(_run_builtin_result_case(tmp_path, monkeypatch))


async def _run_builtin_result_case(tmp_path: Path, monkeypatch) -> None:
    from codepilot.tools.builtins import create_builtin_tools
    from codepilot.tools.contracts import ToolInvocation
    from codepilot.tools.permissions import PermissionPolicy
    from codepilot.tools.registry import ToolRegistry
    from codepilot.tools.runtime import ToolRuntime

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
    registry.extend(create_builtin_tools(tmp_path))
    runtime = ToolRuntime(
        registry,
        permission_policy=PermissionPolicy(
            bash_allow_patterns=[r"^python -c"],
        ),
    )

    write = await runtime.execute(
        ToolInvocation(
            run_id="run_builtin",
            tool_call_id="write_1",
            name="write",
            arguments={"path": "hello.txt", "content": "hello"},
        )
    )
    assert write.workspace_changed is True
    assert write.affected_paths == ("hello.txt",)
    assert write.metadata["file_state"]["path"] == "hello.txt"
    assert isinstance(write.metadata["file_state"]["sha256"], str)

    unchanged = await runtime.execute(
        ToolInvocation(
            run_id="run_builtin",
            tool_call_id="write_2",
            name="write",
            arguments={"path": "hello.txt", "content": "hello"},
        )
    )
    assert unchanged.workspace_changed is False
    assert unchanged.metadata["file_state"]["path"] == "hello.txt"

    read = await runtime.execute(
        ToolInvocation(
            run_id="run_builtin",
            tool_call_id="read_1",
            name="read",
            arguments={"path": "hello.txt"},
        )
    )
    assert read.metadata["file_state"]["path"] == "hello.txt"

    shell = await runtime.execute(
        ToolInvocation(
            run_id="run_builtin",
            tool_call_id="bash_1",
            name="bash",
            arguments={"command": 'python -c "import sys; sys.exit(3)"'},
        )
    )
    assert shell.status == "error"
    assert shell.metadata["error_code"] == "shell_exit_nonzero"
    assert shell.metadata["details"]["exit_code"] == 3

    verification = await runtime.execute(
        ToolInvocation(
            run_id="run_builtin",
            tool_call_id="bash_2",
            name="bash",
            arguments={"command": "python -m pytest -q"},
        )
    )
    assert verification.status == "success"
    assert verification.verification
    assert verification.verification[0].status == "passed"


def _loop_input(
    run_id: str,
    *,
    prompt: str,
    limits: Any | None = None,
    retry_policy: "RetryPolicy | None" = None,
):
    from codepilot.core.contracts import (
        AgentLoopInput,
        AgentLoopLimits,
        RetryPolicy,
        RunCorrelation,
    )
    from codepilot.llm.ports import ModelDescriptor

    return AgentLoopInput(
        run_id=run_id,
        correlation=RunCorrelation(session_id="s1"),
        user_prompt=prompt,
        model=ModelDescriptor(provider="unit-test", model_id="run-test"),
        limits=limits or AgentLoopLimits(max_model_turns=4),
        retry_policy=retry_policy or RetryPolicy(),
    )


class _ScriptedModel:
    def __init__(self, factory: Callable[[Any, int], Any]) -> None:
        self._factory = factory
        self.calls = 0

    async def stream(self, request):
        from codepilot.llm.ports import LLMCompleted

        self.calls += 1
        yield LLMCompleted(message=self._factory(request, self.calls))


class _StaticToolPort:
    def __init__(self, observation):
        self._observation = observation

    def catalog(self, current_mode: str = "build"):
        return {"tools": [self._observation.name]}

    async def execute(self, invocation):
        from dataclasses import replace

        return replace(
            self._observation,
            tool_call_id=invocation.tool_call_id,
            name=invocation.name,
        )


class _CountingToolPort(_StaticToolPort):
    def __init__(self, observation) -> None:
        super().__init__(observation)
        self.executions = 0

    async def execute(self, invocation):
        self.executions += 1
        return await super().execute(invocation)
