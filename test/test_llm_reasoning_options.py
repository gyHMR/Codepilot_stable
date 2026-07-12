from __future__ import annotations


def _model(provider: str):
    from codepilot.protocols import Model

    return Model(
        id="reasoner", name="Reasoner", api="openai-compatible",
        provider=provider, base_url="https://example.invalid", reasoning=True,
        input=["text"], context_window=4000, max_tokens=2000,
    )


def test_openai_reasoning_level_maps_to_reasoning_effort() -> None:
    from codepilot.llm.providers.openai import apply_openai_reasoning_options
    from codepilot.llm.stream import SimpleStreamOptions

    payload = {}
    apply_openai_reasoning_options(payload, _model("openai"), SimpleStreamOptions(reasoning="xhigh"))

    assert payload == {"reasoning_effort": "high"}


def test_deepseek_reasoning_enables_thinking_without_openai_parameter() -> None:
    from codepilot.llm.providers.openai import apply_openai_reasoning_options
    from codepilot.llm.stream import SimpleStreamOptions

    payload = {}
    apply_openai_reasoning_options(payload, _model("deepseek"), SimpleStreamOptions(reasoning="medium"))

    assert payload == {"thinking": {"type": "enabled"}}


def test_anthropic_reasoning_level_maps_to_bounded_thinking_budget() -> None:
    from codepilot.llm.providers.anthropic import apply_anthropic_reasoning_options
    from codepilot.llm.stream import SimpleStreamOptions

    payload = {"max_tokens": 2000, "temperature": 0.2}
    apply_anthropic_reasoning_options(payload, SimpleStreamOptions(reasoning="high"))

    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 1600}
    assert payload["temperature"] == 1
