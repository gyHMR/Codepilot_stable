from __future__ import annotations

import asyncio

from codepilot.core.contracts import TaskStrategy


class _TaskStatePromptPort:
    def prepare(self, request):
        context = request.get("context")
        context = context if isinstance(context, dict) else {}
        state = context.get("task_state")
        lines = []
        if isinstance(state, dict):
            steps = state.get("steps")
            if isinstance(steps, list):
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    status = step.get("status")
                    title = step.get("title")
                    if isinstance(status, str) and isinstance(title, str):
                        lines.append(f"- [{status}] {title}")
        system_prompt = request.get("system_prompt", "")
        if lines:
            system_prompt = f"{system_prompt}\n\nTask State:\n" + "\n".join(lines)
        return {
            "system_prompt": system_prompt,
            "messages": request["messages"],
            "tools": request["tools"],
        }


def test_core_loop_uses_model_port_and_returns_outcome() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentLoopInput,
            AgentLoopLimits,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.loop import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, TextContent, Usage

        class FakeModel:
            async def stream(self, request):
                yield LLMCompleted(
                    message=AssistantMessage(
                        content=[TextContent(text=f"answer:{request.messages[-1].content}")]
                    ),
                    usage=Usage(input=1, output=1),
                )

        events: list[dict] = []
        loop_input = AgentLoopInput(
            run_id="run1",
            correlation=RunCorrelation(session_id="s1"),
            messages=[],
            user_prompt="hello",
            context={"system_prompt": "sys"},
            model=ModelDescriptor(provider="fake", model_id="unit"),
            limits=AgentLoopLimits(max_model_turns=1),
        )

        outcome = await run_agent_loop(
            loop_input,
            AgentLoopPorts(model=FakeModel(), tools=None, events=events.append),
        )

        assert outcome.status == "completed"
        assert outcome.stop_reason == "final_answer"
        assert outcome.final_text == "answer:hello"
        assert [event["type"] for event in outcome.events] == [
            "agent_start",
            "turn_start",
            "message_start",
            "message_end",
            "message_start",
            "message_end",
            "turn_end",
            "agent_end",
        ]
        assert events == outcome.events
        assert outcome.events[0]["runId"] == "run1"
        assert outcome.events[1]["turnId"] == 1
        assert all("eventId" in event for event in outcome.events)

    asyncio.run(run_case())


def test_core_loop_emits_model_text_deltas_as_message_updates() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentLoopInput,
            AgentLoopLimits,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.loop import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, LLMTextDelta, ModelDescriptor
        from codepilot.protocols import AssistantMessage, TextContent

        class FakeModel:
            async def stream(self, request):
                yield LLMTextDelta(text="hel")
                yield LLMTextDelta(text="lo")
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text="hello")])
                )

        events: list[dict] = []
        loop_input = AgentLoopInput(
            run_id="run_stream",
            correlation=RunCorrelation(session_id="s1"),
            user_prompt="hello",
            model=ModelDescriptor(provider="fake", model_id="unit"),
            limits=AgentLoopLimits(max_model_turns=1),
        )

        outcome = await run_agent_loop(
            loop_input,
            AgentLoopPorts(model=FakeModel(), tools=None, events=events.append),
        )

        deltas = [
            event["assistantMessageEvent"]["delta"]
            for event in outcome.events
            if event["type"] == "message_update"
        ]
        assert outcome.status == "completed"
        assert deltas == ["hel", "lo"]
        assert events == outcome.events

    asyncio.run(run_case())


def test_core_loop_turns_tool_approval_observation_into_waiting_outcome() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentLoopInput,
            AgentLoopLimits,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.loop import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, ToolCall
        from codepilot.tools.ports import ToolInterruption, ToolObservation, ToolRiskView

        class FakeModel:
            async def stream(self, request):
                yield LLMCompleted(
                    message=AssistantMessage(
                        content=[ToolCall(id="call1", name="shell", arguments={"cmd": "touch x"})]
                    )
                )

        class FakeTools:
            def catalog(self):
                return {"tools": ["shell"]}

            async def execute(self, invocation):
                return ToolObservation(
                    tool_call_id=invocation.tool_call_id,
                    name=invocation.name,
                    status="approval_required",
                    interruption=ToolInterruption(
                        approval_id="approval1",
                        run_id=invocation.run_id,
                        tool_call_id=invocation.tool_call_id,
                        tool_name=invocation.name,
                        arguments=invocation.arguments,
                        reason="dangerous",
                        risk=ToolRiskView(level="high", summary="writes workspace"),
                    ),
                )

        outcome = await run_agent_loop(
            AgentLoopInput(
                run_id="run1",
                correlation=RunCorrelation(session_id="s1"),
                messages=[],
                user_prompt="change files",
                context={},
                model=ModelDescriptor(provider="fake", model_id="unit"),
                limits=AgentLoopLimits(max_model_turns=1),
            ),
            AgentLoopPorts(model=FakeModel(), tools=FakeTools()),
        )

        assert outcome.status == "waiting_approval"
        assert outcome.stop_reason == "approval_required"
        assert outcome.interruptions[0].approval_id == "approval1"
        event_types = [event["type"] for event in outcome.events]
        assert "tool_execution_start" in event_types
        assert "tool_execution_end" in event_types
        tool_end = next(event for event in outcome.events if event["type"] == "tool_execution_end")
        assert tool_end["toolCallId"] == "call1"
        assert tool_end["toolName"] == "shell"
        assert tool_end["status"] == "approval_required"
        assert tool_end["isError"] is True

    asyncio.run(run_case())


def test_core_loop_preserves_completed_tool_results_before_approval_pause() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentLoopInput,
            AgentLoopLimits,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.loop import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, TextContent, ToolCall, ToolResultMessage
        from codepilot.tools.ports import ToolInterruption, ToolObservation, ToolRiskView

        class FakeModel:
            async def stream(self, request):
                _ = request
                yield LLMCompleted(
                    message=AssistantMessage(
                        content=[
                            ToolCall(id="read_1", name="read", arguments={"path": "a.py"}),
                            ToolCall(id="write_1", name="write", arguments={"path": "a.py"}),
                            ToolCall(id="bash_1", name="bash", arguments={"command": "pytest"}),
                        ]
                    )
                )

        class FakeTools:
            def __init__(self) -> None:
                self.executed: list[str] = []

            def catalog(self):
                return {"tools": ["read", "write", "bash"]}

            async def execute(self, invocation):
                self.executed.append(invocation.tool_call_id)
                if invocation.name == "write":
                    return ToolObservation(
                        tool_call_id=invocation.tool_call_id,
                        name=invocation.name,
                        status="approval_required",
                        interruption=ToolInterruption(
                            approval_id="approval1",
                            run_id=invocation.run_id,
                            tool_call_id=invocation.tool_call_id,
                            tool_name=invocation.name,
                            arguments=invocation.arguments,
                            risk=ToolRiskView(level="medium"),
                        ),
                    )
                return ToolObservation(
                    tool_call_id=invocation.tool_call_id,
                    name=invocation.name,
                    status="success",
                    content=(TextContent(text="read result"),),
                )

        tools = FakeTools()
        outcome = await run_agent_loop(
            AgentLoopInput(
                run_id="run1",
                correlation=RunCorrelation(session_id="s1"),
                messages=[],
                user_prompt="change files",
                context={},
                model=ModelDescriptor(provider="fake", model_id="unit"),
                limits=AgentLoopLimits(max_model_turns=1),
            ),
            AgentLoopPorts(model=FakeModel(), tools=tools),
        )

        assert outcome.status == "waiting_approval"
        assert tools.executed == ["read_1", "write_1"]
        tool_results = [
            message
            for message in outcome.new_messages
            if isinstance(message, ToolResultMessage)
        ]
        assert [message.tool_call_id for message in tool_results] == ["read_1"]
        assert tool_results[0].status == "success"

    asyncio.run(run_case())


def test_core_loop_feeds_tool_observation_back_into_model_until_final_answer() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentLoopInput,
            AgentLoopLimits,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.loop import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, TextContent, ToolCall, ToolResultMessage
        from codepilot.tools.ports import ToolObservation

        class FakeModel:
            def __init__(self) -> None:
                self.requests = []

            async def stream(self, request):
                self.requests.append(request)
                if len(self.requests) == 1:
                    yield LLMCompleted(
                        message=AssistantMessage(
                            content=[
                                ToolCall(
                                    id="call1",
                                    name="read_file",
                                    arguments={"path": "README.md"},
                                )
                            ]
                        )
                    )
                    return
                last_message = request.messages[-1]
                assert isinstance(last_message, ToolResultMessage)
                assert last_message.tool_call_id == "call1"
                yield LLMCompleted(
                    message=AssistantMessage(
                        content=[TextContent(text=f"saw:{last_message.content[0].text}")]
                    )
                )

        class FakeTools:
            def catalog(self):
                return {"tools": ["read_file"]}

            async def execute(self, invocation):
                return ToolObservation(
                    tool_call_id=invocation.tool_call_id,
                    name=invocation.name,
                    status="success",
                    content=(TextContent(text="file body"),),
                    affected_paths=("README.md",),
                    workspace_changed=False,
                )

        model = FakeModel()
        outcome = await run_agent_loop(
            AgentLoopInput(
                run_id="run1",
                correlation=RunCorrelation(session_id="s1"),
                messages=[],
                user_prompt="read it",
                context={},
                model=ModelDescriptor(provider="fake", model_id="unit"),
                limits=AgentLoopLimits(max_model_turns=3),
            ),
            AgentLoopPorts(model=model, tools=FakeTools()),
        )

        assert outcome.status == "completed"
        assert outcome.final_text == "saw:file body"
        assert len(model.requests) == 2
        assert any(isinstance(message, ToolResultMessage) for message in outcome.new_messages)
        assert outcome.counters.model_attempts == 2
        assert outcome.counters.tool_iterations == 1
        assert outcome.counters.tool_calls == 1
        assert outcome.workspace_effects.affected_paths == ("README.md",)
        event_types = [event["type"] for event in outcome.events]
        assert event_types.count("turn_start") == 2
        assert event_types.count("turn_end") == 2
        assert "tool_execution_start" in event_types
        assert "tool_execution_end" in event_types

    asyncio.run(run_case())


def test_convert_to_llm_drops_orphan_tool_results_before_provider_call() -> None:
    from codepilot.core.model_step import convert_to_llm
    from codepilot.protocols import (
        AssistantMessage,
        TextContent,
        ToolCall,
        ToolResultMessage,
        UserMessage,
    )

    messages = [
        UserMessage(content="continue"),
        ToolResultMessage(
            tool_call_id="missing_call",
            tool_name="read_file",
            content=[TextContent(text="orphan output")],
        ),
        AssistantMessage(
            content=[
                TextContent(text="I can continue."),
                ToolCall(id="kept_call", name="read_file", arguments={"path": "README.md"}),
                ToolCall(id="dropped_call", name="read_file", arguments={"path": "old.md"}),
            ],
            stop_reason="toolUse",
        ),
        ToolResultMessage(
            tool_call_id="kept_call",
            tool_name="read_file",
            content=[TextContent(text="paired output")],
        ),
    ]

    converted = convert_to_llm(messages)

    assert not any(
        isinstance(message, ToolResultMessage)
        and message.tool_call_id == "missing_call"
        for message in converted
    )
    assistant = next(
        message for message in converted if isinstance(message, AssistantMessage)
    )
    tool_calls = [block for block in assistant.content if isinstance(block, ToolCall)]
    assert [call.id for call in tool_calls] == ["kept_call"]
    assert isinstance(converted[-1], ToolResultMessage)
    assert converted[-1].tool_call_id == "kept_call"


def test_core_loop_stops_before_repeated_tool_call_execution() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentLoopInput,
            AgentLoopLimits,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.loop import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, ToolCall
        from codepilot.tools.ports import ToolObservation

        class FakeModel:
            async def stream(self, _request):
                yield LLMCompleted(
                    message=AssistantMessage(
                        content=[
                            ToolCall(
                                id="call_repeat",
                                name="read_file",
                                arguments={"path": "README.md"},
                            )
                        ]
                    )
                )

        class FakeTools:
            def __init__(self) -> None:
                self.executions = 0

            def catalog(self):
                return {"tools": ["read_file"]}

            async def execute(self, invocation):
                self.executions += 1
                return ToolObservation(
                    tool_call_id=invocation.tool_call_id,
                    name=invocation.name,
                    status="success",
                )

        tools = FakeTools()
        outcome = await run_agent_loop(
            AgentLoopInput(
                run_id="run_repeat",
                correlation=RunCorrelation(session_id="s1"),
                user_prompt="repeat",
                model=ModelDescriptor(provider="fake", model_id="unit"),
                limits=AgentLoopLimits(
                    max_model_turns=3,
                    repeated_tool_call_limit=1,
                ),
            ),
            AgentLoopPorts(model=FakeModel(), tools=tools),
        )

        assert outcome.status == "failed"
        assert outcome.stop_reason == "repeated_tool_call"
        assert tools.executions == 1
        assert outcome.counters.tool_iterations == 1
        assert outcome.error["code"] == "run.repeated_tool_call"

    asyncio.run(run_case())


def test_core_loop_pauses_for_user_when_tool_iteration_limit_is_reached() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentLoopInput,
            AgentLoopLimits,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.loop import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, ToolCall
        from codepilot.tools.ports import ToolObservation

        class FakeModel:
            def __init__(self) -> None:
                self.calls = 0

            async def stream(self, _request):
                self.calls += 1
                yield LLMCompleted(
                    message=AssistantMessage(
                        content=[
                            ToolCall(
                                id=f"call_{self.calls}",
                                name="read_file",
                                arguments={"path": f"{self.calls}.md"},
                            )
                        ]
                    )
                )

        class FakeTools:
            def __init__(self) -> None:
                self.executions = 0

            def catalog(self):
                return {"tools": ["read_file"]}

            async def execute(self, invocation):
                self.executions += 1
                return ToolObservation(
                    tool_call_id=invocation.tool_call_id,
                    name=invocation.name,
                    status="success",
                )

        model = FakeModel()
        tools = FakeTools()
        outcome = await run_agent_loop(
            AgentLoopInput(
                run_id="run_tool_limit",
                correlation=RunCorrelation(session_id="s1"),
                user_prompt="keep reading",
                model=ModelDescriptor(provider="fake", model_id="unit"),
                limits=AgentLoopLimits(
                    max_model_turns=3,
                    max_tool_iterations=1,
                ),
            ),
            AgentLoopPorts(model=model, tools=tools),
        )

        assert outcome.status == "waiting_user"
        assert outcome.stop_reason == "max_iterations"
        assert model.calls == 2
        assert tools.executions == 1
        assert outcome.error is None
        assert outcome.final_text
        assert "工具调用" in outcome.final_text
        assert "继续" in outcome.final_text

    asyncio.run(run_case())


def test_core_loop_retries_retryable_model_turn_from_retry_policy() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentLoopInput,
            AgentLoopLimits,
            AgentLoopPorts,
            RetryPolicy,
            RunCorrelation,
        )
        from codepilot.core.loop import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, LLMFailed, ModelDescriptor
        from codepilot.protocols import AssistantMessage, TextContent

        class FakeModel:
            def __init__(self) -> None:
                self.calls = 0

            async def stream(self, _request):
                self.calls += 1
                if self.calls == 1:
                    yield LLMFailed(error=RuntimeError("temporary outage"))
                    return
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text="recovered")])
                )

        model = FakeModel()
        outcome = await run_agent_loop(
            AgentLoopInput(
                run_id="run_retry",
                correlation=RunCorrelation(session_id="s1"),
                user_prompt="hello",
                model=ModelDescriptor(provider="fake", model_id="unit"),
                limits=AgentLoopLimits(max_model_turns=1),
                retry_policy=RetryPolicy(
                    enabled=True,
                    max_retries=1,
                    base_delay_ms=0,
                ),
            ),
            AgentLoopPorts(model=model, tools=None, context=_TaskStatePromptPort()),
        )

        assert model.calls == 2
        assert outcome.status == "completed"
        assert outcome.final_text == "recovered"
        assert outcome.counters.model_attempts == 2
        retry_event = next(
            event for event in outcome.events if event["type"] == "model_retry_start"
        )
        assert retry_event["attempt"] == 1
        assert retry_event["maxAttempts"] == 2
        assert retry_event["delayMs"] == 0

    asyncio.run(run_case())


def test_core_loop_injects_task_context_and_returns_task_summary() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentLoopInput,
            AgentLoopLimits,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.loop import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, TextContent

        class FakeModel:
            def __init__(self) -> None:
                self.system_prompts: list[str] = []

            async def stream(self, request):
                self.system_prompts.append(request.system_prompt)
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text="done")])
                )

        model = FakeModel()
        outcome = await run_agent_loop(
            AgentLoopInput(
                run_id="run_task",
                correlation=RunCorrelation(session_id="s1"),
                user_prompt="ship the task-control migration",
                context={"system_prompt": "base rules"},
                model=ModelDescriptor(provider="fake", model_id="unit"),
                task_strategy=TaskStrategy(enabled=True, mode="build"),
                limits=AgentLoopLimits(max_model_turns=1),
            ),
            AgentLoopPorts(model=model, tools=None, context=_TaskStatePromptPort()),
        )

        assert outcome.status == "completed"
        assert outcome.task is not None
        assert outcome.task.goal == "ship the task-control migration"
        assert "base rules" in model.system_prompts[0]
        assert "Task State:" in model.system_prompts[0]
        event_types = [event["type"] for event in outcome.events]
        assert "task_plan_created" in event_types
        assert "completion_checked" in event_types

    asyncio.run(run_case())


def test_core_loop_plan_mode_uses_planning_budget_and_proposed_steps() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentLoopInput,
            AgentLoopLimits,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.loop import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, TextContent

        class FakeModel:
            async def stream(self, request):
                assert "Planning verification hints" not in request.system_prompt
                assert "- [in_progress] Inspect target files" in request.system_prompt
                assert "- [pending] Apply focused refactor" in request.system_prompt
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text="done")])
                )

        outcome = await run_agent_loop(
            AgentLoopInput(
                run_id="run_plan_task",
                correlation=RunCorrelation(session_id="s1"),
                user_prompt="refactor by plan",
                model=ModelDescriptor(provider="fake", model_id="unit"),
                task_strategy=TaskStrategy(
                    enabled=True,
                    mode="plan",
                    planning_budget_profile="wide",
                    steps=[
                        {
                            "title": "Inspect target files",
                            "kind": "read",
                            "acceptance": "Relevant files are known",
                        },
                        {
                            "title": "Apply focused refactor",
                            "kind": "edit",
                            "verification_hint": "python -m pytest -q",
                        },
                    ],
                ),
                limits=AgentLoopLimits(max_model_turns=1),
            ),
                AgentLoopPorts(model=FakeModel(), tools=None, context=_TaskStatePromptPort()),
        )

        assert outcome.task is not None
        assert outcome.task.control_signal["mode"] == "plan"
        planning = outcome.task.control_signal["planning"]
        assert planning["phase"] == "execution"
        assert planning["budget"]["profile"] == "wide"
        assert outcome.task.completed_steps == ["Inspect target files"]
        assert outcome.task.pending_steps == ["Apply focused refactor"]

    asyncio.run(run_case())


def test_core_loop_preserves_task_summary_when_waiting_for_approval() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentLoopInput,
            AgentLoopLimits,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.loop import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, ToolCall
        from codepilot.tools.ports import ToolInterruption, ToolObservation, ToolRiskView

        class FakeModel:
            async def stream(self, request):
                assert "Task State:" in request.system_prompt
                yield LLMCompleted(
                    message=AssistantMessage(
                        content=[
                            ToolCall(
                                id="call1",
                                name="shell",
                                arguments={"cmd": "touch x"},
                            )
                        ]
                    )
                )

        class FakeTools:
            def catalog(self):
                return {"tools": ["shell"]}

            async def execute(self, invocation):
                return ToolObservation(
                    tool_call_id=invocation.tool_call_id,
                    name=invocation.name,
                    status="approval_required",
                    interruption=ToolInterruption(
                        approval_id="approval1",
                        run_id=invocation.run_id,
                        tool_call_id=invocation.tool_call_id,
                        tool_name=invocation.name,
                        arguments=invocation.arguments,
                        reason="dangerous",
                        risk=ToolRiskView(level="high", summary="writes workspace"),
                    ),
                    metadata={"approval_id": "approval1"},
                )

        outcome = await run_agent_loop(
            AgentLoopInput(
                run_id="run_approval_task",
                correlation=RunCorrelation(session_id="s1"),
                user_prompt="edit the file",
                model=ModelDescriptor(provider="fake", model_id="unit"),
                task_strategy=TaskStrategy(enabled=True, mode="build"),
                limits=AgentLoopLimits(max_model_turns=1),
            ),
                AgentLoopPorts(model=FakeModel(), tools=FakeTools(), context=_TaskStatePromptPort()),
        )

        assert outcome.status == "waiting_approval"
        assert outcome.task is not None
        assert outcome.task.goal == "edit the file"
        assert outcome.task.control_signal["recent_error_code"] == "approval_required"

    asyncio.run(run_case())


def test_resume_agent_loop_uses_tool_port_resume_before_continuing_model() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentResumeInput,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.loop import resume_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, TextContent, ToolResultMessage
        from codepilot.tools.ports import ToolObservation

        class FakeModel:
            async def stream(self, request):
                last_message = request.messages[-1]
                assert isinstance(last_message, ToolResultMessage)
                assert last_message.approval_id == "approval1"
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text="continued")])
                )

        class FakeTools:
            def __init__(self) -> None:
                self.decisions = []

            def catalog(self):
                return {"tools": ["shell"]}

            async def execute(self, invocation):
                raise AssertionError("approval resume must not call execute()")

            async def resume(self, decision):
                self.decisions.append(decision)
                return ToolObservation(
                    tool_call_id="call1",
                    name="shell",
                    status="success",
                    content=(TextContent(text="approved output"),),
                    affected_paths=("created.txt",),
                    workspace_changed=True,
                    metadata={"approval_id": decision.approval_id},
                )

        tools = FakeTools()
        outcome = await resume_agent_loop(
            AgentResumeInput(
                run_id="run1",
                correlation=RunCorrelation(session_id="s1"),
                messages=[],
                context={},
                model=ModelDescriptor(provider="fake", model_id="unit"),
                approval_id="approval1",
                decision="approve",
                reason="ok",
            ),
            AgentLoopPorts(model=FakeModel(), tools=tools),
        )

        assert [decision.approval_id for decision in tools.decisions] == ["approval1"]
        assert outcome.status == "completed"
        assert outcome.final_text == "continued"
        assert outcome.workspace_effects.changed is True
        assert outcome.workspace_effects.affected_paths == ("created.txt",)
        event_types = [event["type"] for event in outcome.events]
        assert event_types[:2] == ["agent_start", "turn_start"]
        assert "tool_execution_start" in event_types
        assert "tool_execution_end" in event_types
        tool_end = next(event for event in outcome.events if event["type"] == "tool_execution_end")
        assert tool_end["approvalId"] == "approval1"
        assert tool_end["toolCallId"] == "call1"
        assert tool_end["status"] == "success"

    asyncio.run(run_case())


def test_core_loop_waits_for_user_when_final_answer_lacks_required_verification() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentLoopInput,
            AgentLoopLimits,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.loop import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, TextContent, ToolCall, ToolResultMessage
        from codepilot.tools.ports import ToolObservation

        class FakeModel:
            def __init__(self) -> None:
                self.calls = 0

            async def stream(self, request):
                self.calls += 1
                if self.calls == 1:
                    yield LLMCompleted(
                        message=AssistantMessage(
                            content=[ToolCall(id="edit_1", name="edit_file", arguments={})]
                        )
                    )
                    return
                assert isinstance(request.messages[-1], ToolResultMessage)
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text="done")])
                )

        class FakeTools:
            def catalog(self):
                return {"tools": ["edit_file"]}

            async def execute(self, invocation):
                return ToolObservation(
                    tool_call_id=invocation.tool_call_id,
                    name=invocation.name,
                    status="success",
                    affected_paths=("src/app.py",),
                    workspace_changed=True,
                )

        outcome = await run_agent_loop(
            AgentLoopInput(
                run_id="run_unverified",
                correlation=RunCorrelation(session_id="s1"),
                user_prompt="edit code",
                model=ModelDescriptor(provider="fake", model_id="unit"),
                task_strategy=TaskStrategy(enabled=True, mode="build"),
                limits=AgentLoopLimits(max_model_turns=2),
            ),
            AgentLoopPorts(model=FakeModel(), tools=FakeTools()),
        )

        assert outcome.status == "waiting_user"
        assert outcome.stop_reason == "task_incomplete"
        assert outcome.task is not None
        assert outcome.task.completion_reason == "modified_without_fresh_verification"

    asyncio.run(run_case())


def test_core_loop_pauses_at_tool_iteration_limit_before_more_tools() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentLoopInput,
            AgentLoopLimits,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.loop import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, RunVerification, TextContent, ToolCall
        from codepilot.tools.ports import ToolObservation

        class FakeModel:
            def __init__(self) -> None:
                self.calls = 0

            async def stream(self, _request):
                self.calls += 1
                if self.calls == 1:
                    yield LLMCompleted(
                        message=AssistantMessage(
                            content=[ToolCall(id="edit_1", name="edit_file", arguments={})]
                        )
                    )
                    return
                yield LLMCompleted(
                    message=AssistantMessage(
                        content=[
                            ToolCall(
                                id="test_1",
                                name="bash",
                                arguments={"command": "python -m pytest -q"},
                            )
                        ]
                    )
                )

        class FakeTools:
            def __init__(self) -> None:
                self.executed: list[str] = []

            def catalog(self):
                return {"tools": ["edit_file", "bash"]}

            async def execute(self, invocation):
                self.executed.append(invocation.name)
                if invocation.name == "edit_file":
                    return ToolObservation(
                        tool_call_id=invocation.tool_call_id,
                        name=invocation.name,
                        status="success",
                        affected_paths=("src/app.py",),
                        workspace_changed=True,
                    )
                return ToolObservation(
                    tool_call_id=invocation.tool_call_id,
                    name=invocation.name,
                    status="success",
                    verification=(
                        RunVerification(
                            tool_call_id=invocation.tool_call_id,
                            tool_name=invocation.name,
                            status="passed",
                            command=invocation.arguments["command"],
                            exit_code=0,
                            summary="passed",
                        ),
                    ),
                )

        tools = FakeTools()
        outcome = await run_agent_loop(
            AgentLoopInput(
                run_id="run_final_verification",
                correlation=RunCorrelation(session_id="s1"),
                user_prompt="edit and verify",
                model=ModelDescriptor(provider="fake", model_id="unit"),
                task_strategy=TaskStrategy(enabled=True, mode="build"),
                limits=AgentLoopLimits(
                    max_model_turns=3,
                    max_tool_iterations=1,
                    repeated_tool_call_limit=10,
                ),
            ),
            AgentLoopPorts(model=FakeModel(), tools=tools),
        )

        assert outcome.status == "waiting_user"
        assert outcome.stop_reason == "max_iterations"
        assert tools.executed == ["edit_file"]
        assert not any(event["type"] == "tool_execution_grace" for event in outcome.events)
        assert outcome.task is not None
        assert outcome.task.completion_satisfied is False

    asyncio.run(run_case())


def test_core_loop_denied_tool_blocks_completion_through_task_control() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentLoopInput,
            AgentLoopLimits,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.loop import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, TextContent, ToolCall, ToolResultMessage
        from codepilot.tools.ports import ToolObservation

        class FakeModel:
            def __init__(self) -> None:
                self.calls = 0

            async def stream(self, request):
                self.calls += 1
                if self.calls == 1:
                    yield LLMCompleted(
                        message=AssistantMessage(
                            content=[ToolCall(id="write_1", name="write_file", arguments={})]
                        )
                    )
                    return
                assert isinstance(request.messages[-1], ToolResultMessage)
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text="final")])
                )

        class FakeTools:
            def catalog(self):
                return {"tools": ["write_file"]}

            async def execute(self, invocation):
                return ToolObservation(
                    tool_call_id=invocation.tool_call_id,
                    name=invocation.name,
                    status="denied",
                    metadata={"error_code": "read_only_mode"},
                )

        outcome = await run_agent_loop(
            AgentLoopInput(
                run_id="run_denied",
                correlation=RunCorrelation(session_id="s1"),
                user_prompt="write file",
                model=ModelDescriptor(provider="fake", model_id="unit"),
                task_strategy=TaskStrategy(enabled=True, mode="build"),
                limits=AgentLoopLimits(max_model_turns=2),
            ),
            AgentLoopPorts(model=FakeModel(), tools=FakeTools()),
        )

        assert outcome.status == "waiting_user"
        assert outcome.stop_reason == "task_blocked"
        assert outcome.task is not None
        assert outcome.task.blocked_steps == ["完成当前请求"]
        assert outcome.task.control_signal["recent_error_code"] == "permission_denied"

    asyncio.run(run_case())
