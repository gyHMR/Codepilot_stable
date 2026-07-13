"""Provider 回调类型 —— runtime 装配 provider bridge 时需要的回调类型定义。

本文件仅定义类型别名（TypeAlias），不包含具体实现。
sessions/runtime 配置可以引用这些类型，而不需要知道具体的 adapter 实现。
"""

from typing import Any, Awaitable, Callable

from codepilot.protocols import AssistantMessage, Context, Message, Model

from .stream import SimpleStreamOptions


# ProviderSimpleStreamFn: 简化的流式调用函数类型
#   参数: Model, Context, SimpleStreamOptions | None
#   返回: 可以是任何可迭代或可等待的流对象
ProviderSimpleStreamFn = Callable[
    [Model, Context, SimpleStreamOptions | None],
    Any | Awaitable[Any],
]

# ProviderCompleteFn: 非流式完成调用函数类型
#   参数: Model, Context, SimpleStreamOptions | None
#   返回: AssistantMessage（可直接 await）
ProviderCompleteFn = Callable[
    [Model, Context, SimpleStreamOptions | None],
    AssistantMessage | Awaitable[AssistantMessage],
]

# ProviderMessageConverter: 消息转换函数类型
#   用于在调用前转换消息格式
ProviderMessageConverter = Callable[
    [list[Message]],
    list[Message] | Awaitable[list[Message]],
]

# ProviderApiKeyResolver: API Key 解析函数类型
#   接收 provider 名称，返回对应的 API Key
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