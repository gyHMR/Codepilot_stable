from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_api_registry_registers_and_overwrites_provider() -> None:
    from codepilot.llm import ApiProvider, AssistantMessage, AssistantMessageEventStream, get_api_provider
    from codepilot.llm.api_registry import register_api_provider

    def stream(*_args):
        event_stream = AssistantMessageEventStream()
        event_stream.end(AssistantMessage())
        return event_stream

    first = ApiProvider(api="unit-test-api", stream=stream, stream_simple=stream)
    second = ApiProvider(api="unit-test-api", stream=stream, stream_simple=stream)

    register_api_provider(first)
    assert get_api_provider("unit-test-api") is first

    register_api_provider(second)
    assert get_api_provider("unit-test-api") is second


def test_llm_package_reexports_protocol_types() -> None:
    from codepilot.llm import AssistantMessage as LlmAssistantMessage
    from codepilot.llm import Model as LlmModel
    from codepilot.protocols import AssistantMessage as ProtocolAssistantMessage
    from codepilot.protocols import Model as ProtocolModel

    assert LlmAssistantMessage is ProtocolAssistantMessage
    assert LlmModel is ProtocolModel


def test_model_provider_identity_and_capabilities() -> None:
    from codepilot.llm.models import get_model, get_models, get_providers

    deepseek = get_model("deepseek", "deepseek-v4-pro")
    assert deepseek.api == "openai-compatible"
    assert deepseek.provider == "deepseek"
    assert deepseek.capabilities is not None
    assert deepseek.capabilities.tools
    assert deepseek.capabilities.reasoning

    assert {model.provider for model in get_models("openai")} == {"openai"}
    assert "deepseek" in get_providers()


def test_deepseek_api_key_uses_deepseek_env(monkeypatch) -> None:
    from codepilot.llm.env_api_keys import get_env_api_key

    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    assert get_env_api_key("deepseek") == "deepseek-key"
    assert get_env_api_key("openai") == "openai-key"


def test_old_openai_standard_alias_is_removed() -> None:
    from codepilot.llm import get_api_provider, reset_api_providers
    from codepilot.llm.models import get_model, get_models

    reset_api_providers()

    assert get_api_provider("openai-compatible") is not None
    assert get_api_provider("openai-standard") is None
    assert get_models("openai-standard") == []
    with pytest.raises(KeyError):
        get_model("openai-standard", "gpt-4o-mini")
