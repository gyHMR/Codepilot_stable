from __future__ import annotations

# 新手导读：Provider registry 根据 model.api 分发到具体模型适配器。
# 关注点：新增模型协议时通常先注册新的 provider，再扩展 models 目录。

"""
api -> provider 实现的注册中心。

这样可以做到：
1) stream() 时按 model.api 动态分发；
2) 后续扩展新 provider 时只需注册，不改调用方代码。
"""

from dataclasses import dataclass, field
from typing import Callable, Protocol

from .stream import AssistantMessageEventStream, SimpleStreamOptions, StreamOptions
from codepilot.protocols import (
    AssistantMessage,
    Context,
    Model,
)

StreamFn = Callable[[Model, Context, StreamOptions | None], AssistantMessageEventStream]
SimpleStreamFn = Callable[[Model, Context, SimpleStreamOptions | None], AssistantMessageEventStream]


class LLMProvider(Protocol):
    """LLM Provider 协议：由各模型 API 适配器实现。"""

    api: str

    def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        ...

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        ...


@dataclass
class ApiProvider:
    """已注册的 API Provider 实现：对应一种模型线路协议（如 anthropic-messages、openai-compatible）。"""
    api: str                                    # 协议标识（如 "openai-compatible"）
    stream: StreamFn                            # 标准流式调用函数
    stream_simple: SimpleStreamFn               # 简化流式调用函数
    name: str = ""                              # 人类可读名称
    provider_id: str = ""                       # Provider 标识
    metadata: dict[str, object] = field(default_factory=dict)  # 附加元数据


# 全局 Provider 注册表：api -> ApiProvider
_REGISTRY: dict[str, ApiProvider] = {}


class ApiProviderRegistry:
    """Instance-scoped provider registry used by one runtime assembly."""

    def __init__(self) -> None:
        self._providers: dict[str, ApiProvider] = {}

    def register(self, provider: ApiProvider) -> None:
        self._providers[provider.api] = provider

    def get(self, api: str) -> ApiProvider | None:
        return self._providers.get(api)

    def require(self, api: str) -> ApiProvider:
        provider = self.get(api)
        if provider is None:
            raise RuntimeError(f"No API provider registered for api: {api}")
        return provider

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        return self.require(model.api).stream_simple(
            model, context, options or SimpleStreamOptions()
        )

    async def complete_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessage:
        return await self.stream_simple(model, context, options).result()


def register_api_provider(provider: ApiProvider) -> None:
    """注册或覆盖某个 api 的 provider。"""
    _REGISTRY[provider.api] = provider


def get_api_provider(api: str) -> ApiProvider | None:
    """按 api 获取 provider；不存在返回 None。"""
    return _REGISTRY.get(api)


def clear_api_providers() -> None:
    """清空注册中心（通常用于测试或重置）。"""
    _REGISTRY.clear()


def _resolve_provider(api: str) -> ApiProvider:
    """按 api 标识解析 Provider，未注册时抛出 RuntimeError。"""
    provider = get_api_provider(api)
    if provider is None:
        raise RuntimeError(f"No API provider registered for api: {api}")
    return provider


def stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    """按 model.api 分发到对应 Provider 的标准流式调用。"""
    return _resolve_provider(model.api).stream(model, context, options)


async def complete(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessage:
    """流式调用并等待最终结果（complete = stream + await result）。"""
    return await stream(model, context, options).result()


def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """按 model.api 分发到对应 Provider 的简化流式调用。"""
    effective_options = options or SimpleStreamOptions()
    return _resolve_provider(model.api).stream_simple(
        model,
        context,
        effective_options,
    )


async def complete_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessage:
    """简化流式调用并等待最终结果。"""
    return await stream_simple(model, context, options).result()


def register_builtin_api_providers(registry: ApiProviderRegistry | None = None) -> None:
    """Register built-in provider adapters explicitly during runtime assembly."""

    from .providers.anthropic import stream_anthropic, stream_simple_anthropic
    from .providers.openai import (
        stream_openai_compatible,
        stream_simple_openai_compatible,
    )

    target = registry.register if registry is not None else register_api_provider
    target(
        ApiProvider(
            api="anthropic-messages",
            stream=stream_anthropic,
            stream_simple=stream_simple_anthropic,
            name="Anthropic Messages",
            provider_id="anthropic",
        )
    )
    target(
        ApiProvider(
            api="openai-compatible",
            stream=stream_openai_compatible,
            stream_simple=stream_simple_openai_compatible,
            name="OpenAI-compatible Chat Completions",
            provider_id="openai-compatible",
        )
    )


def builtin_api_provider_registry() -> ApiProviderRegistry:
    registry = ApiProviderRegistry()
    register_builtin_api_providers(registry)
    return registry


def reset_api_providers() -> None:
    """Clear and re-register built-in providers."""

    clear_api_providers()
    register_builtin_api_providers()


__all__ = [
    "ApiProvider",
    "ApiProviderRegistry",
    "LLMProvider",
    "SimpleStreamFn",
    "StreamFn",
    "clear_api_providers",
    "complete",
    "complete_simple",
    "get_api_provider",
    "register_api_provider",
    "register_builtin_api_providers",
    "builtin_api_provider_registry",
    "reset_api_providers",
    "stream",
    "stream_simple",
]
