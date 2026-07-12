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
        prompt = model.requests[1].system_prompt
        assert "## Synthetic Control" in prompt
        assert "Source: runner" in prompt
        assert "Scope: final_answer_only" in prompt
        assert "This is not a user request" in prompt
        assert "没有给出用户可见的最终答复" in prompt
        assert "Raw request: 没有给出用户可见的最终答复" not in prompt
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
def test_core_loop_plan_mode_keeps_soft_plan_proposed() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentLoopInput,
            AgentLoopLimits,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.runner import run_agent_loop
        from codepilot.core.plan import PlanSnapshot, apply_plan_snapshot
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, TextContent

        plan = apply_plan_snapshot(
            None,
            PlanSnapshot.from_mapping(
                {
                    "raw_user_request": "refactor by plan",
                    "interpreted_goal": "refactor by plan",
                    "task_understanding": "User wants a refactor plan that Build can execute.",
                    "current_implementation": "The target files and behavior have been inspected.",
                    "target_design": "Apply a focused refactor while preserving behavior.",
                    "impact_scope": "Target implementation files and focused tests.",
                    "risks_and_open_questions": ["No blocker."],
                    "verification_plan": "Run focused tests.",
                    "summary": "Inspect the target and apply a focused refactor.",
                    "completion_criteria": ["Focused tests pass"],
                    "items": [
                        {
                            "step": "Update the target implementation",
                            "details": "Apply the agreed focused refactor in the owning files.",
                            "verification": "Review the resulting diff for the expected behavior.",
                            "status": "pending",
                        },
                        {
                            "step": "Apply focused refactor",
                            "details": "Implement the agreed behavior.",
                            "verification": "Run focused tests.",
                            "status": "pending",
                        },
                    ],
                }
            ),
            mode="plan",
            run_id="run_plan_task",
        )

        class FakeModel:
            async def stream(self, request):
                assert "- [pending] Update the target implementation" in request.system_prompt
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



def test_active_plan_closeout_stops_after_model_reports_incomplete() -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentLoopInput,
            AgentLoopLimits,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.plan import PlanSnapshot, apply_plan_snapshot
        from codepilot.core.runner import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.protocols import AssistantMessage, TextContent

        plan = apply_plan_snapshot(
            None,
            PlanSnapshot.from_mapping(
                {
                    "raw_user_request": "完成目标修改",
                    "interpreted_goal": "完成目标修改",
                    "summary": "完成并验证目标修改。",
                    "completion_criteria": ["相关测试通过"],
                    "items": [
                        {
                            "step": "完成目标修改",
                            "details": "实现用户要求。",
                            "verification": "运行相关测试。",
                            "status": "in_progress",
                        }
                    ],
                }
            ),
            mode="build",
            run_id="run_incomplete_plan",
        )

        class Model:
            calls = 0

            async def stream(self, _request):
                self.calls += 1
                yield LLMCompleted(
                    message=AssistantMessage(
                        content=[TextContent(text="当前仍未满足完成标准，存在阻塞。")]
                    )
                )

        model = Model()
        outcome = await run_agent_loop(
            AgentLoopInput(
                run_id="run_incomplete_plan",
                correlation=RunCorrelation(session_id="s1"),
                user_prompt="完成目标修改",
                model=ModelDescriptor(provider="fake", model_id="unit"),
                plan_state=plan.to_dict(),
                limits=AgentLoopLimits(max_model_turns=3),
            ),
            AgentLoopPorts(model=model, tools=None, context=_PlanPromptPort()),
        )

        assert outcome.status == "waiting_user"
        assert outcome.stop_reason == "plan_incomplete"
        assert model.calls == 2
        assert outcome.plan is not None
        assert outcome.plan.status == "active"

    asyncio.run(run_case())




def test_plan_mode_allows_clarification_after_one_protocol_steering_turn() -> None:
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

        class Context:
            def __init__(self) -> None:
                self.controls = []

            def prepare(self, request):
                control = request.get("context", {}).get("synthetic_control")
                self.controls.append(control)
                return {
                    "system_prompt": request.get("system_prompt", ""),
                    "messages": request["messages"],
                    "tools": request["tools"],
                }

        class Model:
            def __init__(self) -> None:
                self.calls = 0

            async def stream(self, _request):
                self.calls += 1
                text = (
                    "需要确认界面是否必须兼容移动端。"
                    if self.calls == 1
                    else "请确认：这个界面是否必须兼容移动端？"
                )
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text=text)])
                )

        context = Context()
        model = Model()
        outcome = await run_agent_loop(
            AgentLoopInput(
                run_id="run_plan_clarification_after_steering",
                correlation=RunCorrelation(session_id="s1"),
                messages=[UserMessage(content="为应用设计一个界面方案")],
                user_prompt="为应用设计一个界面方案",
                mode="plan",
                model=ModelDescriptor(provider="fake", model_id="unit"),
                limits=AgentLoopLimits(max_model_turns=2),
            ),
            AgentLoopPorts(model=model, tools=None, context=context),
        )

        assert outcome.status == "waiting_user"
        assert outcome.stop_reason == "plan_clarification_required"
        assert outcome.final_text == "请确认：这个界面是否必须兼容移动端？"
        assert model.calls == 2
        assert context.controls[0] is None
        assert context.controls[1]["kind"] == "plan_publish_required"
        assert not any(isinstance(message, UserMessage) for message in outcome.new_messages)

    asyncio.run(run_case())
