from __future__ import annotations

import asyncio


class _PlanPromptPort:
    def prepare(self, request):
        context = request.get("context")
        context = context if isinstance(context, dict) else {}
        state = context.get("plan_state")
        lines = []
        if isinstance(state, dict):
            items = state.get("items")
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    status = item.get("status")
                    title = item.get("step")
                    if isinstance(status, str) and isinstance(title, str):
                        lines.append(f"- [{status}] {title}")
        system_prompt = request.get("system_prompt", "")
        if lines:
            system_prompt = f"{system_prompt}\n\nPlan Brief:\n" + "\n".join(lines)
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
        from codepilot.core.runner import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, TextContent, Usage, UserMessage

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
            messages=[UserMessage(content="hello")],
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
            "run_guard_checked",
            "turn_end",
            "agent_end",
        ]
        assert events == outcome.events
        assert outcome.events[0]["runId"] == "run1"
        assert outcome.events[1]["turnId"] == 1
        assert all("eventId" in event for event in outcome.events)

    asyncio.run(run_case())


def test_run_guard_steering_is_an_ephemeral_runtime_directive() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentLoopInput,
            AgentLoopLimits,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.runner import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, TextContent, UserMessage

        class FakeModel:
            def __init__(self) -> None:
                self.requests = []

            async def stream(self, request):
                self.requests.append(request)
                text = "" if len(self.requests) == 1 else "现在给出完整答复。"
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text=text)])
                )

        model = FakeModel()
        outcome = await run_agent_loop(
            AgentLoopInput(
                run_id="run_guard_directive",
                correlation=RunCorrelation(session_id="s1"),
                user_prompt="回答问题",
                model=ModelDescriptor(provider="fake", model_id="unit"),
                limits=AgentLoopLimits(max_model_turns=2),
            ),
            AgentLoopPorts(model=model, tools=None),
        )

        assert outcome.status == "completed"
        assert "没有给出用户可见的最终答复" in model.requests[1].system_prompt
        assert not any(isinstance(message, UserMessage) for message in outcome.new_messages)

    asyncio.run(run_case())


def test_core_loop_emits_model_text_deltas_as_message_updates() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentLoopInput,
            AgentLoopLimits,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.runner import run_agent_loop
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
        from codepilot.core.runner import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, ToolCall
        from codepilot.tools.contracts import ToolInterruption, ToolObservation, ToolRiskView

        class FakeModel:
            async def stream(self, request):
                yield LLMCompleted(
                    message=AssistantMessage(
                        content=[ToolCall(id="call1", name="shell", arguments={"cmd": "touch x"})]
                    )
                )

        class FakeTools:
            def catalog(self, current_mode: str = "build"):
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
        assert "tool_started" in event_types
        assert "tool_interrupted" in event_types
        tool_end = next(event for event in outcome.events if event["type"] == "tool_interrupted")
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
        from codepilot.core.runner import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, TextContent, ToolCall, ToolResultMessage
        from codepilot.tools.contracts import ToolInterruption, ToolObservation, ToolRiskView

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

            def catalog(self, current_mode: str = "build"):
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
        from codepilot.core.runner import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, TextContent, ToolCall, ToolResultMessage
        from codepilot.tools.contracts import ToolObservation

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
            def catalog(self, current_mode: str = "build"):
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
        assert "tool_started" in event_types
        assert "tool_completed" in event_types

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
        from codepilot.core.runner import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, ToolCall
        from codepilot.tools.contracts import ToolObservation

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

            def catalog(self, current_mode: str = "build"):
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
        from codepilot.core.runner import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, ToolCall
        from codepilot.tools.contracts import ToolObservation

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

            def catalog(self, current_mode: str = "build"):
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


def test_core_loop_pauses_after_tools_when_no_model_turn_remains() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentLoopInput,
            AgentLoopLimits,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.runner import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, ToolCall
        from codepilot.tools.contracts import ToolObservation

        class FakeModel:
            async def stream(self, _request):
                yield LLMCompleted(
                    message=AssistantMessage(
                        content=[
                            ToolCall(
                                id="call_last",
                                name="edit",
                                arguments={"path": "app.py"},
                            )
                        ]
                    )
                )

        class FakeTools:
            def __init__(self) -> None:
                self.executions = 0

            def catalog(self, current_mode: str = "build"):
                return {"tools": ["edit"]}

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
                run_id="run_model_turn_limit",
                correlation=RunCorrelation(session_id="s1"),
                user_prompt="edit once",
                model=ModelDescriptor(provider="fake", model_id="unit"),
                limits=AgentLoopLimits(max_model_turns=1),
            ),
            AgentLoopPorts(model=FakeModel(), tools=tools),
        )

        assert outcome.status == "waiting_user"
        assert outcome.stop_reason == "max_iterations"
        assert tools.executions == 1
        assert outcome.counters.tool_iterations == 1
        assert outcome.counters.tool_calls == 1
        assert outcome.error is None
        assert "模型轮次" in outcome.final_text
        assert "工具已执行" in outcome.final_text
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
        from codepilot.core.runner import run_agent_loop
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
            AgentLoopPorts(model=model, tools=None, context=_PlanPromptPort()),
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


def test_core_loop_injects_plan_context_and_returns_plan_summary() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentLoopInput,
            AgentLoopLimits,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.runner import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, TextContent, ToolCall
        from codepilot.tools.contracts import ToolObservation
        from codepilot.core.plan import PlanState, PlanUpdate, PlanUpdateItem

        class FakeModel:
            def __init__(self) -> None:
                self.system_prompts: list[str] = []
                self.calls = 0

            async def stream(self, request):
                self.calls += 1
                self.system_prompts.append(request.system_prompt)
                if self.calls == 1:
                    yield LLMCompleted(
                        message=AssistantMessage(
                            content=[
                                ToolCall(
                                    id="plan_closeout",
                                    name="update_plan",
                                    arguments={
                                        "summary": "Read and migrate the plan code.",
                                        "plan_status": "completed",
                                        "plan": [
                                            {
                                                "step": "Read code",
                                                "details": "Inspect the current implementation.",
                                                "verification": "Confirm the relevant symbols.",
                                                "status": "completed",
                                            }
                                        ],
                                    },
                                )
                            ],
                            stop_reason="toolUse",
                        )
                    )
                    return
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text="done")])
                )

        class FakeTools:
            def catalog(self, current_mode: str = "build"):
                return {"tools": ["update_plan"]}

            async def execute(self, invocation):
                return ToolObservation(
                    tool_call_id=invocation.tool_call_id,
                    name=invocation.name,
                    status="success",
                    content=(TextContent(text="Plan updated and marked completed."),),
                    metadata={
                        "plan_update": {
                            "summary": "Read and migrate the plan code.",
                            "plan": [
                                {
                                    "step": "Read code",
                                    "details": "Inspect the current implementation.",
                                    "verification": "Confirm the relevant symbols.",
                                    "status": "completed",
                                }
                            ],
                        },
                        "plan_status": "completed",
                    },
                )

        plan = PlanState.new(
            objective="ship the plan migration",
            origin_mode="build",
            run_id="run_plan",
        )
        plan = plan.apply_update(
            PlanUpdate(
                summary="Read and migrate the plan code.",
                items=(
                    PlanUpdateItem(
                        step="Read code",
                        details="Inspect the current implementation.",
                        verification="Confirm the relevant symbols.",
                        status="completed",
                    ),
                ),
            ),
            mode="build",
            run_id="run_plan",
        )
        model = FakeModel()
        outcome = await run_agent_loop(
            AgentLoopInput(
                run_id="run_plan",
                correlation=RunCorrelation(session_id="s1"),
                user_prompt="ship the plan migration",
                context={"system_prompt": "base rules"},
                model=ModelDescriptor(provider="fake", model_id="unit"),
                plan_state=plan.to_dict(),
                limits=AgentLoopLimits(max_model_turns=2),
            ),
            AgentLoopPorts(model=model, tools=FakeTools(), context=_PlanPromptPort()),
        )

        assert outcome.status == "completed"
        assert outcome.plan is not None
        assert outcome.plan.objective == "ship the plan migration"
        assert outcome.plan.status == "completed"
        assert "base rules" in model.system_prompts[0]
        assert "Plan Brief:" in model.system_prompts[0]
        event_types = [event["type"] for event in outcome.events]
        assert "run_guard_checked" in event_types

    asyncio.run(run_case())


def test_core_loop_plan_mode_keeps_soft_plan_proposed() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentLoopInput,
            AgentLoopLimits,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.runner import run_agent_loop
        from codepilot.core.plan import PlanState, PlanUpdate, PlanUpdateItem
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, TextContent

        plan = PlanState.new(
            objective="refactor by plan",
            origin_mode="plan",
            run_id="run_plan_task",
        )
        plan = plan.apply_update(
            PlanUpdate(
                summary="Inspect the target and apply a focused refactor.",
                items=(
                    PlanUpdateItem(
                        step="Inspect target files",
                        details="Read the files that own the behavior.",
                        verification="Identify the exact edit points.",
                        status="pending",
                    ),
                    PlanUpdateItem(
                        step="Apply focused refactor",
                        details="Implement the agreed behavior.",
                        verification="Run focused tests.",
                        status="pending",
                    ),
                )
            ),
            mode="plan",
            run_id="run_plan_task",
        )

        class FakeModel:
            async def stream(self, request):
                assert "- [pending] Inspect target files" in request.system_prompt
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
                mode="plan",
                plan_state=plan.to_dict(),
                limits=AgentLoopLimits(max_model_turns=1),
            ),
                AgentLoopPorts(model=FakeModel(), tools=None, context=_PlanPromptPort()),
        )

        assert outcome.plan is not None
        assert outcome.plan.status == "proposed"
        assert outcome.plan.items[0]["status"] == "pending"

    asyncio.run(run_case())


def test_core_loop_preserves_plan_summary_when_waiting_for_approval() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentLoopInput,
            AgentLoopLimits,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.runner import run_agent_loop
        from codepilot.core.plan import PlanState, PlanUpdate, PlanUpdateItem
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, ToolCall
        from codepilot.tools.contracts import ToolInterruption, ToolObservation, ToolRiskView

        plan = PlanState.new(
            objective="edit the file",
            origin_mode="build",
            run_id="run_approval_plan",
        )
        plan = plan.apply_update(
            PlanUpdate(
                summary="Edit and verify the target file.",
                items=(
                    PlanUpdateItem(
                        step="Edit file",
                        details="Apply the required source change.",
                        verification="Inspect the diff.",
                        status="in_progress",
                    ),
                ),
            ),
            mode="build",
            run_id="run_approval_plan",
        )

        class FakeModel:
            async def stream(self, request):
                assert "Plan Brief:" in request.system_prompt
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
            def catalog(self, current_mode: str = "build"):
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
                plan_state=plan.to_dict(),
                limits=AgentLoopLimits(max_model_turns=1),
            ),
                AgentLoopPorts(model=FakeModel(), tools=FakeTools(), context=_PlanPromptPort()),
        )

        assert outcome.status == "waiting_approval"
        assert outcome.plan is not None
        assert outcome.plan.objective == "edit the file"
        assert outcome.signals.approval_required is True

    asyncio.run(run_case())


def test_core_loop_steers_repeated_truncated_reads_before_retrying() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentLoopInput,
            AgentLoopLimits,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.runner import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, TextContent, ToolCall, UserMessage
        from codepilot.tools.contracts import ToolObservation

        class FakeModel:
            def __init__(self) -> None:
                self.system_prompts: list[str] = []

            async def stream(self, request):
                self.system_prompts.append(request.system_prompt)
                if len(self.system_prompts) <= 2:
                    yield LLMCompleted(
                        message=AssistantMessage(
                            content=[
                                ToolCall(
                                    id=f"read_{len(self.system_prompts)}",
                                    name="read",
                                    arguments={
                                        "path": "big.py",
                                        "offset": 1,
                                        "limit": 200,
                                        "max_chars": 4000 + len(self.system_prompts),
                                    },
                                )
                            ]
                        )
                    )
                    return
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text="used pagination guidance")])
                )

        class FakeTools:
            def catalog(self, current_mode: str = "build"):
                return {"tools": ["read"]}

            async def execute(self, invocation):
                return ToolObservation(
                    tool_call_id=invocation.tool_call_id,
                    name=invocation.name,
                    status="success",
                    content=(TextContent(text="lines 1-20 of 500\n...<truncated>..."),),
                    metadata={
                        "read_paths": ["big.py"],
                        "actual_start_line": 1,
                        "actual_end_line": 20,
                        "next_offset": 21,
                        "has_more": True,
                        "truncated": True,
                        "output_quality": {"truncated": True},
                    },
                )

        model = FakeModel()
        outcome = await run_agent_loop(
            AgentLoopInput(
                run_id="run_truncated_reads",
                correlation=RunCorrelation(session_id="s1"),
                user_prompt="read big file",
                model=ModelDescriptor(provider="fake", model_id="unit"),
                limits=AgentLoopLimits(max_model_turns=4, repeated_tool_call_limit=10),
            ),
            AgentLoopPorts(model=model, tools=FakeTools()),
        )

        assert outcome.status == "completed"
        assert outcome.final_text == "used pagination guidance"
        assert any("offset/limit" in prompt for prompt in model.system_prompts)
        assert any("next_offset=21" in prompt for prompt in model.system_prompts)
        assert not any(
            isinstance(message, UserMessage)
            and "Do not repeat truncated reads" in str(message.content)
            for message in outcome.new_messages
        )

    asyncio.run(run_case())


def test_resume_agent_loop_uses_tool_port_resume_before_continuing_model() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentResumeInput,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.runner import resume_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, RunVerification, TextContent, ToolCall, ToolResultMessage
        from codepilot.tools.contracts import ToolObservation

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

            def catalog(self, current_mode: str = "build"):
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
                    verification=(
                        RunVerification(
                            tool_call_id="call1",
                            tool_name="shell",
                            status="passed",
                            command="pytest",
                            exit_code=0,
                            summary="passed",
                        ),
                    ),
                    metadata={"approval_id": decision.approval_id},
                )

        tools = FakeTools()
        outcome = await resume_agent_loop(
            AgentResumeInput(
                run_id="run1",
                correlation=RunCorrelation(session_id="s1"),
                messages=[
                    AssistantMessage(
                        content=[
                            ToolCall(
                                id="call1",
                                name="shell",
                                arguments={"command": "pytest"},
                            )
                        ]
                    )
                ],
                context={},
                model=ModelDescriptor(provider="fake", model_id="unit"),
                approval_id="approval1",
                decision="approve",
                reason="ok",
                event_start_seq=10,
                turn_start_seq=7,
                run_state={
                    "counters": {
                        "model_attempts": 2,
                        "tool_iterations": 1,
                        "tool_calls": 1,
                    },
                    "workspace_changed": True,
                    "affected_paths": ["existing.txt"],
                    "verification": [],
                    "verification_status": "stale",
                    "approval_required": True,
                    "seen_tool_call_ids": ["call1"],
                    "pending_approval_tool_call_ids": ["call1"],
                },
            ),
            AgentLoopPorts(model=FakeModel(), tools=tools),
        )

        assert [decision.approval_id for decision in tools.decisions] == ["approval1"]
        assert outcome.status == "completed"
        assert outcome.final_text == "continued"
        assert outcome.workspace_effects.changed is True
        assert outcome.workspace_effects.affected_paths == ("created.txt", "existing.txt")
        assert outcome.counters.model_attempts == 3
        assert outcome.counters.tool_iterations == 2
        assert outcome.counters.tool_calls == 1
        assert outcome.signals.approval_required is False
        assert outcome.run_state["verification_status"] == "passed"
        assert outcome.run_state["seen_tool_call_ids"] == ["call1"]
        assert outcome.run_state["pending_approval_tool_call_ids"] == []
        event_types = [event["type"] for event in outcome.events]
        assert event_types[:2] == ["agent_start", "turn_start"]
        assert "tool_started" in event_types
        assert "tool_completed" in event_types
        assert outcome.events[0]["eventId"] == "run1:11"
        assert outcome.events[0]["turnId"] == 7
        assert outcome.events[1]["turnId"] == 8
        tool_end = next(event for event in outcome.events if event["type"] == "tool_completed")
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
        from codepilot.core.runner import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, TextContent, ToolCall, ToolResultMessage
        from codepilot.tools.contracts import ToolObservation

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
            def catalog(self, current_mode: str = "build"):
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
                limits=AgentLoopLimits(max_model_turns=2),
            ),
            AgentLoopPorts(model=FakeModel(), tools=FakeTools()),
        )

        assert outcome.status == "waiting_user"
        assert outcome.stop_reason == "run_guard"
        assert outcome.signals.workspace_changed is True
        assert outcome.signals.verification_status == "stale"

    asyncio.run(run_case())


def test_core_loop_pauses_at_tool_iteration_limit_before_more_tools() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentLoopInput,
            AgentLoopLimits,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.runner import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, RunVerification, TextContent, ToolCall
        from codepilot.tools.contracts import ToolObservation

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

            def catalog(self, current_mode: str = "build"):
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
        assert not any(event["type"] == "tool_interrupted" for event in outcome.events)
        assert outcome.signals.workspace_changed is True

    asyncio.run(run_case())


def test_core_loop_feeds_denied_tool_result_back_to_model() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentLoopInput,
            AgentLoopLimits,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.runner import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, TextContent, ToolCall, ToolResultMessage
        from codepilot.tools.contracts import ToolObservation

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
                last = request.messages[-1]
                assert isinstance(last, ToolResultMessage)
                assert last.status == "denied"
                assert last.error_code == "read_only_permission_mode"
                yield LLMCompleted(
                    message=AssistantMessage(
                        content=[TextContent(text="我无法执行写入操作，请切换 build 模式或确认替代方案。")]
                    )
                )

        class FakeTools:
            def catalog(self, current_mode: str = "build"):
                return {"tools": ["write_file"]}

            async def execute(self, invocation):
                return ToolObservation(
                    tool_call_id=invocation.tool_call_id,
                    name=invocation.name,
                    status="denied",
                    metadata={"error_code": "read_only_permission_mode"},
                )

        model = FakeModel()
        outcome = await run_agent_loop(
            AgentLoopInput(
                run_id="run_denied",
                correlation=RunCorrelation(session_id="s1"),
                user_prompt="write file",
                model=ModelDescriptor(provider="fake", model_id="unit"),
                limits=AgentLoopLimits(max_model_turns=2),
            ),
            AgentLoopPorts(model=model, tools=FakeTools()),
        )

        assert model.calls == 2
        assert outcome.status == "completed"
        assert outcome.stop_reason == "final_answer"
        assert "无法执行写入操作" in outcome.final_text
        assert outcome.signals.last_error is not None
        assert outcome.signals.last_error["error_code"] == "read_only_permission_mode"

    asyncio.run(run_case())
