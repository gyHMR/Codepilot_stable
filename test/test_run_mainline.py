from __future__ import annotations

import asyncio


def test_retryable_model_error_remains_inside_one_run() -> None:
    asyncio.run(_run_retry_case())


async def _run_retry_case() -> None:
    from codepilot.core.contracts import (
        AgentLoopInput,
        AgentLoopLimits,
        AgentLoopPorts,
        RetryPolicy,
        RunCorrelation,
    )
    from codepilot.core.runner import run_agent_loop
    from codepilot.llm.ports import LLMCompleted, LLMFailed, ModelDescriptor
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
        AgentLoopInput(
            run_id="run_retry",
            correlation=RunCorrelation(session_id="s1"),
            user_prompt="retry",
            model=ModelDescriptor(provider="unit-test", model_id="run-test"),
            limits=AgentLoopLimits(max_model_turns=1),
            retry_policy=RetryPolicy(enabled=True, max_retries=1, base_delay_ms=0),
        ),
        AgentLoopPorts(model=model, tools=None),
    )

    assert outcome.status == "completed"
    assert outcome.counters.model_attempts == 2
    assert model.attempts == 2
    assert {event["runId"] for event in outcome.events} == {outcome.run_id}
    assert any(event["type"] == "model_retry_start" for event in outcome.events)
