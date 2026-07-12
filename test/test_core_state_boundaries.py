from __future__ import annotations

import asyncio

import pytest

from codepilot.core.contracts import (
    AgentLoopInput,
    AgentLoopLimits,
    AgentLoopPorts,
    CoreRunBoundary,
    RunCorrelation,
)
from codepilot.core.runner import run_agent_loop
from codepilot.llm.ports import LLMCompleted, ModelDescriptor
from codepilot.protocols import AssistantMessage, TextContent, ToolCall, UserMessage
from codepilot.tools.registry import ToolCatalogSnapshot


class RecordingStatePort:
    def __init__(self, *, fail_on: str | None = None) -> None:
        self.boundaries: list[CoreRunBoundary] = []
        self.fail_on = fail_on

    async def commit(self, boundary: CoreRunBoundary) -> None:
        self.boundaries.append(boundary)
        if boundary.kind == self.fail_on:
            raise RuntimeError(f"state commit failed: {boundary.kind}")


def _input() -> AgentLoopInput:
    return AgentLoopInput(
        run_id="run_boundaries",
        correlation=RunCorrelation(session_id="session_1"),
        messages=[UserMessage(content="inspect")],
        user_prompt="inspect",
        model=ModelDescriptor(provider="fake", model_id="unit"),
        limits=AgentLoopLimits(max_model_turns=1),
    )


def test_core_commits_model_and_finalization_boundaries() -> None:
    async def run_case() -> None:
        class Model:
            async def stream(self, _request):
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text="done")])
                )

        state = RecordingStatePort()
        outcome = await run_agent_loop(
            _input(),
            AgentLoopPorts(model=Model(), tools=None, state=state),
        )

        assert outcome.status == "completed"
        assert [boundary.kind for boundary in state.boundaries] == [
            "before_model",
            "after_model",
            "before_finalization",
        ]
        assert state.boundaries[1].new_messages == (outcome.final_message,)
        assert state.boundaries[-1].new_messages == ()

    asyncio.run(run_case())


def test_before_model_commit_failure_prevents_model_call() -> None:
    async def run_case() -> None:
        class Model:
            calls = 0

            async def stream(self, _request):
                self.calls += 1
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text="unexpected")])
                )

        model = Model()
        state = RecordingStatePort(fail_on="before_model")

        with pytest.raises(RuntimeError, match="before_model"):
            await run_agent_loop(
                _input(),
                AgentLoopPorts(model=model, tools=None, state=state),
            )

        assert model.calls == 0
        assert [boundary.kind for boundary in state.boundaries] == ["before_model"]

    asyncio.run(run_case())


def test_before_tools_commit_failure_prevents_tool_execution() -> None:
    async def run_case() -> None:
        class Model:
            async def stream(self, _request):
                yield LLMCompleted(
                    message=AssistantMessage(
                        content=[
                            ToolCall(
                                id="call_1",
                                name="shell",
                                arguments={"command": "pytest"},
                            )
                        ]
                    )
                )

        class Tools:
            execute_calls = 0

            def catalog_snapshot(self, *, mode=None):
                return ToolCatalogSnapshot("catalog_1", (), 0)

            async def execute_batch(self, _requests):
                self.execute_calls += 1
                return []

        tools = Tools()
        state = RecordingStatePort(fail_on="before_tools")

        with pytest.raises(RuntimeError, match="before_tools"):
            await run_agent_loop(
                _input(),
                AgentLoopPorts(model=Model(), tools=tools, state=state),  # type: ignore[arg-type]
            )

        assert tools.execute_calls == 0
        assert [boundary.kind for boundary in state.boundaries] == [
            "before_model",
            "after_model",
            "before_tools",
        ]

    asyncio.run(run_case())
