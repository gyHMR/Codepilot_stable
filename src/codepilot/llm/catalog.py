"""内置模型目录 —— 描述 provider、api、上下文窗口和能力。

本文件定义了 Codepilot 内置支持的 LLM 模型配置，按 provider 分组。
每个模型包含完整的配置信息：ID、名称、API 协议、能力声明等。

注意：它只描述模型能力，不读取 API Key 或发送请求。

使用方式：
    model = get_model("anthropic", "claude-sonnet-4-5")
    models = get_models("anthropic")
    providers = get_providers()
"""

import os

from codepilot.protocols import Model, ModelCapabilities

# ── 内置模型注册表 ────────────────────────────────────────────────────────────
# 数据结构: provider -> model_id -> Model
# 每个 Model 包含完整的配置信息
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


# ── 查询函数 ──────────────────────────────────────────────────────────────────


def get_model(provider: str, model_id: str) -> Model:
    """获取指定 provider 和 ID 的模型配置。

    参数:
        provider: 提供商名称（如 "anthropic"、"openai"）
        model_id: 模型 ID（如 "claude-sonnet-4-5"）

    返回:
        Model 对象

    抛出:
        KeyError: 如果未找到指定的模型
    """
    try:
        return _MODELS[provider][model_id]
    except KeyError as exc:
        raise KeyError(f"Unknown model: {provider}/{model_id}") from exc


def get_models(provider: str) -> list[Model]:
    """获取指定 provider 的全部模型列表。

    参数:
        provider: 提供商名称

    返回:
        该 provider 的 Model 列表
    """
    return list(_MODELS.get(provider, {}).values())


def get_providers() -> list[str]:
    """列出当前所有已注册的 provider 名称。

    返回:
        provider 名称列表
    """
    return list(_MODELS.keys())


# ── API Key 辅助函数 ─────────────────────────────────────────────────────────


def get_env_api_key_name(provider: str) -> str | None:
    """返回标准的 API Key 环境变量名。

    参数:
        provider: 提供商名称

    返回:
        环境变量名（如 "ANTHROPIC_API_KEY"），或 None
    """
    if provider == "anthropic":
        return "ANTHROPIC_API_KEY"
    if provider == "deepseek":
        return "DEEPSEEK_API_KEY"
    if provider in {"openai", "openai-compatible"}:
        return "OPENAI_API_KEY"
    return None


def get_env_api_key(provider: str) -> str | None:
    """从环境变量中获取 API Key。

    参数:
        provider: 提供商名称

    返回:
        API Key 字符串，或 None（未设置时）
    """
    env_name = get_env_api_key_name(provider)
    return os.getenv(env_name) if env_name else None


__all__ = [
    "get_env_api_key",
    "get_env_api_key_name",
    "get_model",
    "get_models",
    "get_providers",
]