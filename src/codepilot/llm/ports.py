from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Literal, Protocol

from codepilot.protocols import (
    AssistantMessage,
    Context,
    ImageContent,
    LLMErrorInfo,
    Message,
    Model,
    SimpleStreamOptions,
    Usage,
)

from .api_registry import complete_simple, stream_simple
from .event_stream import AssistantMessageEventStream


@dataclass(frozen=True)
class ModelDescriptor:
    provider: str
    model_id: str
    capabilities: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMOptions:
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning: str | None = None


@dataclass(frozen=True)
class LLMRequest:
    model: ModelDescriptor
    messages: tuple[Message, ...]
    system_prompt: str = ""
    tools: tuple[Any, ...] = field(default_factory=tuple)
    options: LLMOptions = field(default_factory=LLMOptions)
    correlation: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMStarted:
    kind: Literal["started"] = "started"


@dataclass(frozen=True)
class LLMTextDelta:
    text: str
    kind: Literal["text_delta"] = "text_delta"


@dataclass(frozen=True)
class LLMCompleted:
    message: AssistantMessage
    usage: Usage | None = None
    kind: Literal["completed"] = "completed"


@dataclass(frozen=True)
class LLMFailed:
    error: Any
    kind: Literal["failed"] = "failed"


LLMEvent = LLMStarted | LLMTextDelta | LLMCompleted | LLMFailed


class ModelPort(Protocol):
    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        ...


StreamFn = Callable[
    [Model, Context, SimpleStreamOptions | None],
    AssistantMessageEventStream | Awaitable[AssistantMessageEventStream],
]
CompleteFn = Callable[
    [Model, Context, SimpleStreamOptions | None],
    AssistantMessage | Awaitable[AssistantMessage],
]
MessageConverter = Callable[[list[Message]], list[Message] | Awaitable[list[Message]]]
ApiKeyResolver = Callable[[str], str | None | Awaitable[str | None]]


class ProviderModelPort:
    """ModelPort adapter over the existing provider registry or injected stream function."""

    def __init__(
        self,
        *,
        model: Model,
        stream_fn: StreamFn | None = None,
        complete_fn: CompleteFn | None = None,
        convert_messages: MessageConverter | None = None,
        get_api_key: ApiKeyResolver | None = None,
    ) -> None:
        self._model = model
        self._stream_fn = stream_fn
        self._complete_fn = complete_fn
        self._convert_messages = convert_messages
        self._get_api_key = get_api_key

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        try:
            messages = list(request.messages)
            if self._convert_messages is not None:
                converted = self._convert_messages(messages)
                messages = await converted if _is_awaitable(converted) else converted
            capabilities = self._model.capabilities
            capability_error = _validate_capabilities(
                messages,
                model=self._model,
            )
            if capability_error is not None:
                yield LLMFailed(error=capability_error)
                return
            context = Context(
                system_prompt=request.system_prompt if capabilities.system_prompt else None,
                messages=messages,
                tools=[
                    tool.to_spec() if hasattr(tool, "to_spec") else tool
                    for tool in request.tools
                ] if capabilities.tools else [],
            )
            api_key = None
            if self._get_api_key is not None:
                resolved = self._get_api_key(self._model.provider)
                api_key = await resolved if _is_awaitable(resolved) else resolved
            options = SimpleStreamOptions(
                reasoning=(
                    request.options.reasoning  # type: ignore[arg-type]
                    if capabilities.reasoning
                    else None
                ),
                temperature=request.options.temperature,
                max_tokens=request.options.max_tokens,
                api_key=api_key,
                session_id=request.correlation.get("session_id"),
            )
            if not capabilities.streaming:
                complete_fn = self._complete_fn or complete_simple
                message = complete_fn(self._model, context, options)
                yield LLMCompleted(
                    message=await message if _is_awaitable(message) else message
                )
                return
            stream_fn = self._stream_fn or stream_simple
            response = stream_fn(self._model, context, options)
            response_stream = await response if _is_awaitable(response) else response
            async for event in response_stream:
                if event.get("type") == "text_delta":
                    yield LLMTextDelta(text=str(event.get("delta", "")))
            yield LLMCompleted(message=await response_stream.result())
        except Exception as exc:
            yield LLMFailed(error=exc)


def _is_awaitable(value: object) -> bool:
    return hasattr(value, "__await__")


def _validate_capabilities(
    messages: list[Message],
    *,
    model: Model,
) -> LLMErrorInfo | None:
    if model.capabilities.vision:
        return None
    has_images = any(
        isinstance(message.content, list)
        and any(isinstance(block, ImageContent) for block in message.content)
        for message in messages
    )
    if not has_images:
        return None
    return LLMErrorInfo(
        code="llm.unsupported_capability",
        message=f"Model {model.id} does not support image input",
        retryable=False,
        kind="unsupported_capability",
        provider=model.provider,
        model=model.id,
        details={"capability": "vision"},
    )


__all__ = [
    "ApiKeyResolver",
    "CompleteFn",
    "LLMCompleted",
    "LLMEvent",
    "LLMFailed",
    "LLMOptions",
    "LLMRequest",
    "LLMStarted",
    "LLMTextDelta",
    "MessageConverter",
    "ModelDescriptor",
    "ModelPort",
    "ProviderModelPort",
    "StreamFn",
]
