from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _model(**capability_overrides):
    from codepilot.protocols import Model, ModelCapabilities

    capabilities = ModelCapabilities(**capability_overrides)
    return Model(
        id="contract-model",
        name="Contract Model",
        api="unit-test",
        provider="unit-test",
        base_url="",
        reasoning=capabilities.reasoning,
        input=["text", "image"] if capabilities.vision else ["text"],
        context_window=1000,
        max_tokens=100,
        capabilities=capabilities,
    )


def test_event_emitter_normalizes_envelope_and_rejects_unknown_event_type() -> None:
    asyncio.run(_run_event_emitter_contract_case())


async def _run_event_emitter_contract_case() -> None:
    from codepilot.core.runner import AgentEventEmitter

    events: list[dict[str, Any]] = []
    emitter = AgentEventEmitter(events.append, run_id=" run_events ", session_id=" session_1 ")

    await emitter.emit({"type": "turn_start", "run_id": "caller_override"})
    await emitter.emit({"type": "message_end", "message": "payload"})

    assert events[0]["run_id"] == "run_events"
    assert events[0]["session_id"] == "session_1"
    assert events[0]["turn_id"] == 1
    assert events[0]["event_id"] == "run_events:1"
    assert events[1]["turn_id"] == 1
    assert events[1]["event_id"] == "run_events:2"

    with pytest.raises(ValueError, match="runtime event type"):
        await emitter.emit({"type": "unknown_event"})

    with pytest.raises(ValueError, match="event type"):
        await emitter.emit({})


def test_core_context_validates_session_context_boundaries() -> None:
    from codepilot.core.contracts import AgentContext
    from codepilot.protocols import UserMessage

    messages = [UserMessage(content="hello")]
    plan_state = {"plan_id": "plan_1", "nested": {"step": "s1"}}
    context = AgentContext(
        system_prompt="rules",
        messages=messages,
        plan_state=plan_state,
        run_signals={"verification_status": "unknown"},
    )
    messages.append(UserMessage(content="mutated"))
    plan_state["nested"] = {"step": "mutated"}

    assert context.messages == [UserMessage(content="hello")]
    assert context.plan_state == {"plan_id": "plan_1", "nested": {"step": "s1"}}
    assert context.run_signals == {"verification_status": "unknown"}

    with pytest.raises(TypeError, match="messages"):
        AgentContext(system_prompt="rules", messages="not a list")  # type: ignore[arg-type]


def test_legacy_agent_state_is_not_a_core_contract() -> None:
    import codepilot.core as core
    import codepilot.core.contracts as core_types
    from codepilot.core import __all__ as core_exports

    assert not hasattr(core_types, "AgentState")
    assert not hasattr(core, "AgentState")
    assert "AgentState" not in core_exports


def test_model_turn_prepares_context_after_plan_state_is_injected() -> None:
    asyncio.run(_run_prepare_context_sees_plan_state_case())


async def _run_prepare_context_sees_plan_state_case() -> None:
    from codepilot.core.contracts import AgentLoopInput, AgentLoopPorts, RunCorrelation
    from codepilot.core.model_step import build_model_request
    from codepilot.llm.ports import ModelDescriptor
    from codepilot.protocols import UserMessage

    captured: dict[str, Any] = {}

    class ContextPort:
        def prepare(self, request):
            captured["system_prompt"] = request["system_prompt"]
            captured["plan_state"] = request["context"]["plan_state"]
            return {
                "system_prompt": request["system_prompt"],
                "messages": request["messages"],
                "tools": request["tools"],
            }

    request = await build_model_request(
        AgentLoopInput(
            run_id="run_prepare_context",
            correlation=RunCorrelation(session_id="s1"),
            messages=[UserMessage(content="continue")],
                context={
                    "system_prompt": "Base rules",
                    "plan_state": {"plan_id": "plan_1", "origin_mode": "build"},
                },
                model=ModelDescriptor(provider="unit-test", model_id="contract-model"),
        ),
        AgentLoopPorts(model=None, tools=None, context=ContextPort()),
        [UserMessage(content="continue")],
    )

    assert captured["plan_state"] == {"plan_id": "plan_1", "origin_mode": "build"}
    assert "Base rules" in request.system_prompt
    assert "plan_1" not in request.system_prompt
