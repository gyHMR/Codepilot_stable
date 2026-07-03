from __future__ import annotations

# 新手导读：register_builtins.py 在导入时注册内置 provider。
# 关注点：应用启动时确保这里被加载，api_registry 才知道可用模型协议。

"""
内置 provider 注册入口。
"""

from ..api_registry import ApiProvider, clear_api_providers, register_api_provider
from .anthropic import stream_anthropic, stream_simple_anthropic
from .openai_compatible import stream_openai_compatible, stream_simple_openai_compatible


def register_builtin_api_providers() -> None:
    """注册内置的线路协议适配器（Anthropic Messages 和 OpenAI-compatible）。"""
    register_api_provider(
        ApiProvider(
            api="anthropic-messages",
            stream=stream_anthropic,
            stream_simple=stream_simple_anthropic,
            name="Anthropic Messages",
            provider_id="anthropic",
        )
    )
    register_api_provider(
        ApiProvider(
            api="openai-compatible",
            stream=stream_openai_compatible,
            stream_simple=stream_simple_openai_compatible,
            name="OpenAI-compatible Chat Completions",
            provider_id="openai-compatible",
        )
    )


def reset_api_providers() -> None:
    """重置并重新注册内置 provider。"""
    clear_api_providers()
    register_builtin_api_providers()


# 模块加载即注册，保证 stream() 可直接使用。
register_builtin_api_providers()
