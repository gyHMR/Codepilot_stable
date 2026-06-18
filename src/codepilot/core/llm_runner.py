from __future__ import annotations

"""LLM streaming runner used by the core agent loop."""

from typing import Any, Awaitable, Callable

from codepilot.llm.stream import stream_simple
from codepilot.llm.types import AssistantMessage, Context, SimpleStreamOptions

from .events import AgentEventEmitter, maybe_await
from .types import AgentContext, AgentLoopConfig


StreamFn = Callable[[Any, Context, SimpleStreamOptions | None], Any | Awaitable[Any]]


class LLMStreamRunner:
    """Converts provider stream events into Agent message events."""

    def __init__(
        self,
        *,
        config: AgentLoopConfig,
        emitter: AgentEventEmitter,
        stream_fn: StreamFn | None = None,
    ) -> None:
        self._config = config
        self._emitter = emitter
        self._stream_fn = stream_fn

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
        llm_context = Context(
            system_prompt=context.system_prompt,
            messages=llm_messages,
            tools=context.tools,
        )

        resolved_api_key = None
        if self._config.get_api_key:
            resolved_api_key = await maybe_await(
                self._config.get_api_key(self._config.model.provider)
            )
        options = SimpleStreamOptions(
            reasoning=self._config.reasoning,
            api_key=resolved_api_key,
            session_id=self._config.session_id,
        )

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
        response_stream: Any,
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
        return final_message

