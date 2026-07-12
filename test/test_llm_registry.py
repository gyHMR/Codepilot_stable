from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_api_registry_registers_and_overwrites_provider() -> None:
    from codepilot.llm.registry import ApiProvider, get_api_provider, register_api_provider
    from codepilot.llm.stream import AssistantMessageEventStream
    from codepilot.protocols import AssistantMessage

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


def test_provider_registry_instances_are_isolated() -> None:
    from codepilot.llm.registry import ApiProvider, ApiProviderRegistry

    first = ApiProvider(api="unit", stream=lambda *_: None, stream_simple=lambda *_: None)
    second = ApiProvider(api="unit", stream=lambda *_: None, stream_simple=lambda *_: None)
    left = ApiProviderRegistry()
    right = ApiProviderRegistry()

    left.register(first)
    right.register(second)

    assert left.get("unit") is first
    assert right.get("unit") is second
    assert left.get("missing") is None


def test_llm_package_does_not_reexport_protocol_types() -> None:
    import codepilot.llm as llm

    assert "AssistantMessage" not in llm.__all__
    assert "Model" not in llm.__all__
    assert not hasattr(llm, "AssistantMessage")
    assert not hasattr(llm, "Model")


def test_model_provider_identity_and_capabilities() -> None:
    from codepilot.llm.catalog import get_model, get_models, get_providers

    deepseek = get_model("deepseek", "deepseek-v4-pro")
    assert deepseek.api == "openai-compatible"
    assert deepseek.provider == "deepseek"
    assert deepseek.capabilities is not None
    assert deepseek.capabilities.tools
    assert deepseek.capabilities.reasoning

    assert {model.provider for model in get_models("openai")} == {"openai"}
    assert "deepseek" in get_providers()


def test_deepseek_api_key_uses_deepseek_env(monkeypatch) -> None:
    from codepilot.llm.catalog import get_env_api_key

    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    assert get_env_api_key("deepseek") == "deepseek-key"
    assert get_env_api_key("openai") == "openai-key"


def test_old_openai_standard_alias_is_removed() -> None:
    from codepilot.llm.catalog import get_model, get_models
    from codepilot.llm.registry import get_api_provider, reset_api_providers

    reset_api_providers()

    assert get_api_provider("openai-compatible") is not None
    assert get_api_provider("openai-standard") is None
    assert get_models("openai-standard") == []
    with pytest.raises(KeyError):
        get_model("openai-standard", "gpt-4o-mini")


def test_runtime_assembly_uses_isolated_builtin_provider_registry(tmp_path) -> None:
    from codepilot.llm.registry import clear_api_providers, get_api_provider
    from codepilot.runtime import SessionOpenIntent
    from codepilot.runtime.builder import build_runtime_session

    clear_api_providers()
    assert get_api_provider("openai-compatible") is None

    session = build_runtime_session(
        SessionOpenIntent(
            workspace_dir=tmp_path,
            provider="deepseek",
            model_id="deepseek-v4-pro",
            load_workspace_resources=False,
            memory_enabled=False,
        )
    )
    session.controller.close()

    assert get_api_provider("openai-compatible") is None
    assert session.model_port._registry.get("openai-compatible") is not None  # noqa: SLF001
