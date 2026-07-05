from __future__ import annotations

# 新手导读：provider_types.py 只放 runtime 装配 provider bridge 时需要的回调类型。
# 关注点：sessions/runtime 配置可以引用这些类型，而不需要知道 concrete adapter。

from typing import Any, Awaitable, Callable

from codepilot.protocols import AssistantMessage, Context, Message, Model, SimpleStreamOptions


ProviderSimpleStreamFn = Callable[
    [Model, Context, SimpleStreamOptions | None],
    Any | Awaitable[Any],
]
ProviderCompleteFn = Callable[
    [Model, Context, SimpleStreamOptions | None],
    AssistantMessage | Awaitable[AssistantMessage],
]
ProviderMessageConverter = Callable[
    [list[Message]],
    list[Message] | Awaitable[list[Message]],
]
ProviderApiKeyResolver = Callable[
    [str],
    str | None | Awaitable[str | None],
]


__all__ = [
    "ProviderApiKeyResolver",
    "ProviderCompleteFn",
    "ProviderMessageConverter",
    "ProviderSimpleStreamFn",
]
