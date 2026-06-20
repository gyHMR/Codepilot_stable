from __future__ import annotations

"""LLM 能力感知运行器：由核心 Agent 循环调用，负责调用 LLM 并处理流式响应。"""

from typing import Any, Awaitable, Callable

from codepilot.llm.event_stream import AssistantMessageEventStream
from codepilot.llm.api_registry import complete_simple, stream_simple
from codepilot.protocols import (
    AssistantMessage,
    Context,
    ImageContent,
    Message,
    Model,
    ModelCapabilities,
    SimpleStreamOptions,
    UserMessage,
)
from codepilot.protocols import LLMErrorInfo

from .events import AgentEventEmitter, maybe_await
from .types import AgentContext, AgentLoopConfig, ContextPreparationRequest


# 流式调用函数类型：接收模型、上下文和选项，返回事件流
StreamFn = Callable[
    [Model, Context, SimpleStreamOptions | None],
    AssistantMessageEventStream | Awaitable[AssistantMessageEventStream],
]
# 非流式调用函数类型：接收模型、上下文和选项，返回完整的助手消息
CompleteFn = Callable[
    [Model, Context, SimpleStreamOptions | None],
    AssistantMessage | Awaitable[AssistantMessage],
]


class LLMStreamRunner:
    """LLM 流式运行器：将 provider 的流式事件转换为 Agent 消息事件。"""

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
        """调用 LLM 获取助手响应（流式或非流式，取决于模型能力）。"""
        prepared_context = _with_current_task_context(context)
        if self._config.prepare_context:
            prepared = await maybe_await(
                self._config.prepare_context(
                    context,
                    ContextPreparationRequest(
                        session_id=self._config.session_id,
                        model_context_window=self._config.model.context_window,
                        model_max_output_tokens=self._config.model.max_tokens,
                        signal=signal,
                    ),
                )
            )
            prepared_context = AgentContext(
                system_prompt=prepared.system_prompt,
                messages=list(prepared.messages),
                tools=list(prepared.tools),
                current_task=context.current_task,
                recovered_task=context.recovered_task,
            )
            await self._emitter.emit(
                {
                    "type": "context_prepared",
                    "report": prepared.report.to_dict(),
                }
            )

        messages = prepared_context.messages
        if self._config.transform_context:
            messages = await maybe_await(self._config.transform_context(messages, signal))

        llm_messages = await maybe_await(self._config.convert_to_llm(messages))
        capabilities = self._config.model.capabilities or ModelCapabilities()
        capability_error = self._validate_capabilities(llm_messages, capabilities)
        if capability_error is not None:
            return await self._finalize_direct_response(context, capability_error)

        llm_context = Context(
            system_prompt=prepared_context.system_prompt if capabilities.system_prompt else None,
            messages=llm_messages,
            tools=[tool.to_spec() for tool in prepared_context.tools] if capabilities.tools else [],
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
        """完成流式响应：获取最终消息，更新上下文，发射 message_end 事件。"""
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
        """完成非流式响应：将消息追加到上下文，发射 start/end 事件。"""
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
        """校验模型能力：如果消息包含图片但模型不支持视觉，则返回错误消息。"""
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
        """如果助手消息包含 LLM 错误，则发射 error 事件通知外部。"""
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


def _with_current_task_context(context: AgentContext) -> AgentContext:
    """将当前任务信息注入系统提示词末尾（如果尚未包含）。"""
    if not context.current_task:
        return context
    marker = "## Current Task"
    system_prompt = context.system_prompt.rstrip()
    if marker not in system_prompt:
        system_prompt = f"{system_prompt}\n\n{context.current_task}".strip()
    return AgentContext(
        system_prompt=system_prompt,
        messages=context.messages,
        tools=context.tools,
        current_task=context.current_task,
        recovered_task=context.recovered_task,
    )
