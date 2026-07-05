from __future__ import annotations

# 新手导读：adapters.py 把现有 provider registry 适配成 core 可消费的 ModelPort。
# 关注点：端口契约在 ports.py；这里才允许接触 provider registry、event stream 和模型能力细节。

from collections.abc import AsyncIterator
from typing import Awaitable, Callable

from codepilot.protocols import (
    AssistantMessage,
    Context,
    ImageContent,
    LLMErrorInfo,
    Message,
    Model,
    SimpleStreamOptions,
)

from .ports import (
    LLMCompleted,
    LLMEvent,
    LLMFailed,
    LLMRequest,
    LLMTextDelta,
    ModelPort,
)
from .registry import complete_simple, stream_simple
from .stream import AssistantMessageEventStream


ProviderSimpleStreamFn = Callable[
    [Model, Context, SimpleStreamOptions | None],
    AssistantMessageEventStream | Awaitable[AssistantMessageEventStream],
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


class ProviderModelPort(ModelPort):
    """ModelPort adapter over the existing provider registry or injected stream function."""

    def __init__(
        self,
        *,
        model: Model,
        stream_fn: ProviderSimpleStreamFn | None = None,
        complete_fn: ProviderCompleteFn | None = None,
        convert_messages: ProviderMessageConverter | None = None,
        get_api_key: ProviderApiKeyResolver | None = None,
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
                tools=list(request.tools) if capabilities.tools else [],
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
                session_id=request.correlation.session_id or None,
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
    "ProviderApiKeyResolver",
    "ProviderCompleteFn",
    "ProviderMessageConverter",
    "ProviderModelPort",
    "ProviderSimpleStreamFn",
]
