from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from .event_stream import AssistantMessageEventStream
from .types import Context, Model, SimpleStreamOptions, StreamOptions


StreamFn = Callable[[Model, Context, StreamOptions | None], AssistantMessageEventStream]
SimpleStreamFn = Callable[[Model, Context, SimpleStreamOptions | None], AssistantMessageEventStream]


class LLMProvider(Protocol):
    """Protocol implemented by model API adapters."""

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
class LLMProviderDescriptor:
    """Registered provider implementation for one model wire protocol."""

    api: str
    stream: StreamFn
    stream_simple: SimpleStreamFn
    name: str = ""
    provider_id: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


__all__ = [
    "LLMProvider",
    "LLMProviderDescriptor",
    "SimpleStreamFn",
    "StreamFn",
]
