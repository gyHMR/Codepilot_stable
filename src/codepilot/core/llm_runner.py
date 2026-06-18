from __future__ import annotations

"""LLM capability-aware runner used by the core agent loop."""

from typing import Any, Awaitable, Callable

from codepilot.llm.event_stream import AssistantMessageEventStream
from codepilot.llm.stream import complete_simple, stream_simple
from codepilot.llm.types import (
    AssistantMessage,
    Context,
    ImageContent,
    Message,
    Model,
    ModelCapabilities,
    SimpleStreamOptions,
    UserMessage,
)
from codepilot.protocols.errors import LLMErrorInfo

from .events import AgentEventEmitter, maybe_await
from .types import AgentContext, AgentLoopConfig


StreamFn = Callable[
    [Model, Context, SimpleStreamOptions | None],
    AssistantMessageEventStream | Awaitable[AssistantMessageEventStream],
]
CompleteFn = Callable[
    [Model, Context, SimpleStreamOptions | None],
    AssistantMessage | Awaitable[AssistantMessage],
]


class LLMStreamRunner:
    """Converts provider stream events into Agent message events."""

    def __init__(
        self,
        *,
        config: AgentLoopConfig,
        emitter: AgentEventEmitter,
        stream_fn: StreamFn | None = None,
        complete_fn: CompleteFn | None = None,
    ) -> None:
        self._config = config
        self._emitter = emitter
        self._stream_fn = stream_fn
        self._complete_fn = complete_fn

    async def stream_assistant_response(
        self,
        context: AgentContext,
        *,
        signal: Any | None = None,
    ) -> AssistantMessage:
        messages = context.messages
        if self._config.transform_context:
            messages = await maybe_await(self._config.transform_context(messages, signal))

        llm_messages = await maybe_await(self._config.convert_to_llm(messages))
        capabilities = self._config.model.capabilities or ModelCapabilities()
        capability_error = self._validate_capabilities(llm_messages, capabilities)
        if capability_error is not None:
            return await self._finalize_direct_response(context, capability_error)

        llm_context = Context(
            system_prompt=context.system_prompt if capabilities.system_prompt else None,
            messages=llm_messages,
            tools=[tool.to_spec() for tool in context.tools] if capabilities.tools else [],
        )

        resolved_api_key = None
        if self._config.get_api_key:
            resolved_api_key = await maybe_await(
                self._config.get_api_key(self._config.model.provider)
            )
        options = SimpleStreamOptions(
            reasoning=self._config.reasoning if capabilities.reasoning else None,
            api_key=resolved_api_key,
            session_id=self._config.session_id,
        )

        if not capabilities.streaming:
            complete_fn = self._complete_fn or complete_simple
            final_message = await maybe_await(
                complete_fn(self._config.model, llm_context, options)
            )
            return await self._finalize_direct_response(context, final_message)

        stream_fn = self._stream_fn or stream_simple
        response_stream = await maybe_await(
            stream_fn(self._config.model, llm_context, options)
        )

        partial: AssistantMessage | None = None
        added_partial = False
        async for event in response_stream:
            event_type = event.get("type")
            if event_type == "start":
                partial = event["partial"]
                context.messages.append(partial)
                added_partial = True
                await self._emitter.emit({"type": "message_start", "message": partial})
            elif event_type in {
                "text_start",
                "text_delta",
                "text_end",
                "thinking_start",
                "thinking_delta",
                "thinking_end",
                "toolcall_start",
                "toolcall_delta",
                "toolcall_end",
            }:
                if partial is not None:
                    partial = event["partial"]
                    context.messages[-1] = partial
                    await self._emitter.emit(
                        {
                            "type": "message_update",
                            "message": partial,
                            "assistantMessageEvent": event,
                        }
                    )
            elif event_type in {"done", "error"}:
                return await self._finalize_stream_response(
                    response_stream,
                    context,
                    added_partial=added_partial,
                )

        return await self._finalize_stream_response(
            response_stream,
            context,
            added_partial=added_partial,
        )

    async def _finalize_stream_response(
        self,
        response_stream: AssistantMessageEventStream,
        context: AgentContext,
        *,
        added_partial: bool,
    ) -> AssistantMessage:
        final_message = await response_stream.result()
        if added_partial:
            context.messages[-1] = final_message
        else:
            context.messages.append(final_message)
            await self._emitter.emit(
                {"type": "message_start", "message": final_message}
            )
        await self._emitter.emit({"type": "message_end", "message": final_message})
        await self._emit_llm_error(final_message)
        return final_message

    async def _finalize_direct_response(
        self,
        context: AgentContext,
        final_message: AssistantMessage,
    ) -> AssistantMessage:
        context.messages.append(final_message)
        await self._emitter.emit({"type": "message_start", "message": final_message})
        await self._emitter.emit({"type": "message_end", "message": final_message})
        await self._emit_llm_error(final_message)
        return final_message

    def _validate_capabilities(
        self,
        messages: list[Message],
        capabilities: ModelCapabilities,
    ) -> AssistantMessage | None:
        if capabilities.vision:
            return None
        has_images = any(
            isinstance(message, UserMessage)
            and isinstance(message.content, list)
            and any(isinstance(block, ImageContent) for block in message.content)
            for message in messages
        )
        if not has_images:
            return None
        info = LLMErrorInfo(
            code="llm.unsupported_capability",
            message=f"Model {self._config.model.id} does not support image input",
            retryable=False,
            kind="unsupported_capability",
            provider=self._config.model.provider,
            model=self._config.model.id,
            details={"capability": "vision"},
        )
        return AssistantMessage(
            api=self._config.model.api,
            provider=self._config.model.provider,
            model=self._config.model.id,
            stop_reason="error",
            error_message=info.message,
            error_info=info,
        )

    async def _emit_llm_error(self, message: AssistantMessage) -> None:
        if message.stop_reason != "error":
            return
        info = message.error_info or LLMErrorInfo(
            code="llm.unknown",
            message=message.error_message or "Unknown LLM error",
            retryable=False,
            provider=self._config.model.provider,
            model=self._config.model.id,
        )
        await self._emitter.emit(
            {
                "type": "error",
                "error": info.code,
                "message": info.message,
                "source": "llm",
                "code": info.code,
                "retryable": info.retryable,
                "provider": info.provider,
                "model": info.model,
                "statusCode": info.status_code,
                "errorInfo": info,
            }
        )
