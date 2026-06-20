from __future__ import annotations

"""
api -> provider 实现的注册中心。

这样可以做到：
1) stream() 时按 model.api 动态分发；
2) 后续扩展新 provider 时只需注册，不改调用方代码。
"""

from dataclasses import dataclass, field
from typing import Callable, Protocol

from .event_stream import AssistantMessageEventStream
from codepilot.protocols import (
    AssistantMessage,
    Context,
    Model,
    SimpleStreamOptions,
    StreamOptions,
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
