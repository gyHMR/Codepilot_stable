from __future__ import annotations

# 新手导读：models.py 是内置模型目录，描述 provider、api、上下文窗口和能力。
# 关注点：它只描述模型能力，不读取 API Key 或发送请求。

"""
内置模型目录模块。

定义了 Codepilot 内置支持的 LLM 模型配置，按 provider 分组。
每个模型包含完整的配置信息：ID、名称、API 协议、能力声明等。

使用方式：
    model = get_model("anthropic", "claude-sonnet-4-5")
    models = get_models("anthropic")
    providers = get_providers()
"""

from codepilot.protocols import Model, ModelCapabilities

# 内置模型注册表：provider -> model_id -> Model
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

def get_model(provider: str, model_id: str) -> Model:
    """获取单个模型，找不到会抛 KeyError。"""
    try:
        return _MODELS[provider][model_id]
    except KeyError as exc:
        raise KeyError(f"Unknown model: {provider}/{model_id}") from exc


def get_models(provider: str) -> list[Model]:
    """获取某 provider 的全部模型。"""
    return list(_MODELS.get(provider, {}).values())


def get_providers() -> list[str]:
    """列出当前内置 provider。"""
    return list(_MODELS.keys())
