from __future__ import annotations

"""Built-in model catalog."""

from codepilot.protocols import Model, ModelCapabilities

_MODELS: dict[str, dict[str, Model]] = {
    "anthropic": {
        "claude-sonnet-4-5": Model(
            id="claude-sonnet-4-5",
            name="Claude Sonnet 4.5",
            api="anthropic-messages",
            provider="anthropic",
            base_url="https://api.anthropic.com",
            reasoning=True,
            input=["text", "image"],
            context_window=200_000,
            max_tokens=8192,
            capabilities=ModelCapabilities(
                tools=True,
                vision=True,
                streaming=True,
                reasoning=True,
                system_prompt=True,
            ),
        ),
        "glm-4.7": Model(
            id="glm-4.7",
            name="GLM-4.7",
            api="anthropic-messages",
            provider="anthropic",
            base_url="https://open.bigmodel.cn/api/anthropic",
            reasoning=True,
            input=["text", "image"],
            context_window=200_000,
            max_tokens=8192,
            capabilities=ModelCapabilities(
                tools=True,
                vision=True,
                streaming=True,
                reasoning=True,
                system_prompt=True,
            ),
        ),
    },
    "openai": {
        "gpt-4o-mini": Model(
            id="gpt-4o-mini",
            name="GPT-4o mini",
            api="openai-compatible",
            provider="openai",
            base_url="https://api.openai.com/v1",
            reasoning=False,
            input=["text", "image"],
            context_window=128_000,
            max_tokens=16_384,
            capabilities=ModelCapabilities(
                tools=True,
                vision=True,
                json_schema=True,
                streaming=True,
                reasoning=False,
                system_prompt=True,
                tool_choice=True,
                parallel_tool_calls=True,
            ),
        ),
    },
    "deepseek": {
        "deepseek-v4-pro": Model(
            id="deepseek-v4-pro",
            name="DeepSeek V4 Pro",
            api="openai-compatible",
            provider="deepseek",
            base_url="https://api.deepseek.com/v1",
            reasoning=True,
            input=["text", "image"],
            context_window=200_000,
            max_tokens=8192,
            capabilities=ModelCapabilities(
                tools=True,
                vision=True,
                streaming=True,
                reasoning=True,
                system_prompt=True,
            ),
        ),
    },
}

_PROVIDER_ALIASES = {
    "openai-standard": "openai",
}


def get_model(provider: str, model_id: str) -> Model:
    """获取单个模型，找不到会抛 KeyError。"""
    original_provider = provider
    provider = _PROVIDER_ALIASES.get(provider, provider)
    if original_provider == "openai-standard" and model_id.startswith("deepseek"):
        provider = "deepseek"
    try:
        return _MODELS[provider][model_id]
    except KeyError as exc:
        raise KeyError(f"Unknown model: {provider}/{model_id}") from exc


def get_models(provider: str) -> list[Model]:
    """获取某 provider 的全部模型。"""
    if provider == "openai-standard":
        return [*_MODELS.get("openai", {}).values(), *_MODELS.get("deepseek", {}).values()]
    provider = _PROVIDER_ALIASES.get(provider, provider)
    return list(_MODELS.get(provider, {}).values())


def get_providers() -> list[str]:
    """列出当前内置 provider。"""
    return list(_MODELS.keys())
