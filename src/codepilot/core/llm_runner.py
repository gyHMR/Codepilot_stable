from __future__ import annotations

"""
LLM 能力感知运行器模块

本模块是 Agent 系统与 LLM 服务之间的桥梁，负责：
1. 根据模型能力（如是否支持流式、视觉等）选择合适的调用方式
2. 处理 LLM 的流式响应，将 token 级别的事件转换为消息级别的事件
3. 管理上下文准备和转换，确保发送给 LLM 的消息格式正确
4. 处理错误和异常情况，提供统一的错误报告机制

核心类：
- LLMStreamRunner: 流式运行器，处理流式和非流式两种调用模式

典型使用场景：
- 由 AgentLoop（核心 Agent 循环）调用，获取 LLM 的助手响应
- 支持取消信号（signal），允许用户中断长时间运行的请求
"""

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
# 用于支持流式输出的模型，可以实时获取生成的 token
StreamFn = Callable[
    [Model, Context, SimpleStreamOptions | None],
    AssistantMessageEventStream | Awaitable[AssistantMessageEventStream],
]

# 非流式调用函数类型：接收模型、上下文和选项，返回完整的助手消息
# 用于不支持流式输出的模型，需要等待完整响应
CompleteFn = Callable[
    [Model, Context, SimpleStreamOptions | None],
    AssistantMessage | Awaitable[AssistantMessage],
]


class LLMStreamRunner:
    """
    LLM 流式运行器

    负责调用 LLM 并处理响应，支持两种模式：
    1. 流式模式：实时接收 token 事件，逐步构建助手消息
    2. 非流式模式：等待完整响应后一次性返回

    主要职责：
    - 根据模型能力选择流式或非流式调用
    - 将 LLM 的 token 级事件转换为 Agent 可理解的消息事件
    - 管理消息上下文的更新和维护
    - 处理错误并发射错误事件

    属性:
        _config: Agent 循环配置，包含模型信息、API 密钥获取函数等
        _emitter: 事件发射器，用于向外部广播消息生命周期事件
        _stream_fn: 可选的自定义流式调用函数，默认使用 stream_simple
        _complete_fn: 可选的自定义非流式调用函数，默认使用 complete_simple
    """

    def __init__(
        self,
        *,
        config: AgentLoopConfig,
        emitter: AgentEventEmitter,
        stream_fn: StreamFn | None = None,
        complete_fn: CompleteFn | None = None,
    ) -> None:
        """
        初始化 LLM 流式运行器

        参数:
            config: Agent 循环配置对象，包含模型、会话 ID、上下文准备函数等
            emitter: 事件发射器，用于广播消息开始、更新、结束等事件
            stream_fn: 自定义的流式调用函数，如果为 None 则使用默认的 stream_simple
            complete_fn: 自定义的非流式调用函数，如果为 None 则使用默认的 complete_simple
        """
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
        """
        调用 LLM 获取助手响应

        这是 LLMStreamRunner 的核心方法，执行完整的 LLM 调用流程：
        1. 准备上下文（如果配置了 prepare_context 函数）
        2. 转换消息格式（如果配置了 transform_context 函数）
        3. 校验模型能力（如是否支持视觉输入）
        4. 根据模型能力选择流式或非流式调用
        5. 处理响应并发射相应的事件

        参数:
            context: Agent 上下文，包含系统提示词、消息列表、工具列表等
            signal: 可选的取消信号，用于中断长时间运行的请求（如用户取消操作）

        返回:
            AssistantMessage: 助手消息对象，包含生成的文本、工具调用等信息

        事件发射:
            - context_prepared: 上下文准备完成时发射，包含准备报告
            - message_start: 消息开始时发射
            - message_update: 消息更新时发射（流式模式下，每个 token 事件都会触发）
            - message_end: 消息结束时发射
            - error: 发生错误时发射
        """
        # 步骤 1: 将当前任务信息注入系统提示词
        prepared_context = _with_current_task_context(context)

        # 步骤 2: 如果配置了上下文准备函数，执行上下文准备
        # 上下文准备可能包括：裁剪过长的消息列表、添加系统提示词等
        if self._config.prepare_context:
            prepared = await maybe_await(
                self._config.prepare_context(
                    prepared_context,
                    ContextPreparationRequest(
                        session_id=self._config.session_id,
                        model_context_window=self._config.model.context_window,
                        model_max_output_tokens=self._config.model.max_tokens,
                        signal=signal,
                    ),
                )
            )
            # 使用准备后的上下文替换原上下文
            prepared_context = AgentContext(
                system_prompt=prepared.system_prompt,
                messages=list(prepared.messages),
                tools=list(prepared.tools),
                current_task=prepared_context.current_task,
                task_recovery_projection=prepared_context.task_recovery_projection,
                task_signal=prepared_context.task_signal,
            )
            # 发射上下文准备完成事件，通知外部上下文已准备好
            await self._emitter.emit(
                {
                    "type": "context_prepared",
                    "report": prepared.report.to_dict(),
                }
            )

        # 步骤 3: 如果配置了上下文转换函数，对消息进行转换
        # 转换可能包括：消息格式转换、内容过滤等
        messages = prepared_context.messages
        if self._config.transform_context:
            messages = await maybe_await(self._config.transform_context(messages, signal))

        # 步骤 4: 将内部消息格式转换为 LLM 可理解的格式
        llm_messages = await maybe_await(self._config.convert_to_llm(messages))

        # 步骤 5: 获取模型能力并校验
        capabilities = self._config.model.capabilities or ModelCapabilities()
        capability_error = self._validate_capabilities(llm_messages, capabilities)
        if capability_error is not None:
            # 如果能力校验失败（如模型不支持图片），直接返回错误消息
            return await self._finalize_direct_response(context, capability_error)

        # 步骤 6: 构建 LLM 上下文对象
        # 根据模型能力决定是否包含系统提示词和工具
        llm_context = Context(
            system_prompt=prepared_context.system_prompt if capabilities.system_prompt else None,
            messages=llm_messages,
            tools=[tool.to_spec() for tool in prepared_context.tools] if capabilities.tools else [],
        )

        # 步骤 7: 解析 API 密钥
        resolved_api_key = None
        if self._config.get_api_key:
            resolved_api_key = await maybe_await(
                self._config.get_api_key(self._config.model.provider)
            )

        # 步骤 8: 构建调用选项
        options = SimpleStreamOptions(
            reasoning=self._config.reasoning if capabilities.reasoning else None,
            api_key=resolved_api_key,
            session_id=self._config.session_id,
        )

        # 步骤 9: 根据模型能力选择调用方式
        if not capabilities.streaming:
            # 模型不支持流式输出，使用非流式调用
            complete_fn = self._complete_fn or complete_simple
            final_message = await maybe_await(
                complete_fn(self._config.model, llm_context, options)
            )
            return await self._finalize_direct_response(context, final_message)

        # 步骤 10: 模型支持流式输出，使用流式调用
        stream_fn = self._stream_fn or stream_simple
        response_stream = await maybe_await(
            stream_fn(self._config.model, llm_context, options)
        )

        # 步骤 11: 处理流式响应事件
        partial: AssistantMessage | None = None
        added_partial = False
        async for event in response_stream:
            event_type = event.get("type")

            if event_type == "start":
                # 流式开始：创建初始的 partial 消息并添加到上下文
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
                # 流式更新：更新 partial 消息并发射更新事件
                # 这些事件包括：文本生成、思考过程、工具调用等
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
                # 流式结束或出错：完成流式响应处理
                return await self._finalize_stream_response(
                    response_stream,
                    context,
                    added_partial=added_partial,
                )

        # 如果流式事件循环正常结束（没有 done/error 事件），也需要完成响应
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
        """
        完成流式响应处理

        在流式响应结束后，执行以下操作：
        1. 从响应流中获取最终的完整消息
        2. 更新上下文中的消息（替换 partial 或追加新消息）
        3. 发射 message_end 事件
        4. 检查并处理可能的 LLM 错误

        参数:
            response_stream: 流式响应对象，包含所有流式事件的结果
            context: Agent 上下文，包含消息列表
            added_partial: 是否已经在上下文中添加了 partial 消息
                          True: 需要替换上下文中的最后一条消息
                          False: 需要将最终消息追加到上下文

        返回:
            AssistantMessage: 最终的助手消息对象
        """
        # 从响应流中获取最终的完整消息
        final_message = await response_stream.result()

        # 更新上下文中的消息
        if added_partial:
            # 如果已经添加了 partial 消息，用最终消息替换它
            context.messages[-1] = final_message
        else:
            # 如果没有添加过 partial 消息（异常情况），追加最终消息
            context.messages.append(final_message)
            await self._emitter.emit(
                {"type": "message_start", "message": final_message}
            )

        # 发射消息结束事件
        await self._emitter.emit({"type": "message_end", "message": final_message})

        # 检查并处理可能的 LLM 错误
        await self._emit_llm_error(final_message)

        return final_message

    async def _finalize_direct_response(
        self,
        context: AgentContext,
        final_message: AssistantMessage,
    ) -> AssistantMessage:
        """
        完成非流式响应处理

        处理非流式 LLM 调用的响应，执行以下操作：
        1. 将最终消息追加到上下文的消息列表
        2. 发射 message_start 和 message_end 事件
        3. 检查并处理可能的 LLM 错误

        参数:
            context: Agent 上下文，包含消息列表
            final_message: LLM 返回的完整助手消息

        返回:
            AssistantMessage: 助手消息对象（与输入相同）
        """
        # 将最终消息追加到上下文
        context.messages.append(final_message)

        # 发射消息生命周期事件
        await self._emitter.emit({"type": "message_start", "message": final_message})
        await self._emitter.emit({"type": "message_end", "message": final_message})

        # 检查并处理可能的 LLM 错误
        await self._emit_llm_error(final_message)

        return final_message

    def _validate_capabilities(
        self,
        messages: list[Message],
        capabilities: ModelCapabilities,
    ) -> AssistantMessage | None:
        """
        校验模型能力是否满足消息需求

        检查消息中是否包含模型不支持的内容类型，目前主要检查：
        - 视觉能力：如果消息包含图片但模型不支持视觉，返回错误消息

        参数:
            messages: 待发送给 LLM 的消息列表
            capabilities: 模型的能力描述对象

        返回:
            AssistantMessage | None:
            - None: 能力校验通过，可以继续调用 LLM
            - AssistantMessage: 能力校验失败，返回包含错误信息的助手消息
        """
        # 如果模型支持视觉能力，跳过检查
        if capabilities.vision:
            return None

        # 检查消息中是否包含图片内容
        has_images = any(
            isinstance(message, UserMessage)
            and isinstance(message.content, list)
            and any(isinstance(block, ImageContent) for block in message.content)
            for message in messages
        )

        # 如果没有图片，无需检查
        if not has_images:
            return None

        # 构建错误信息
        info = LLMErrorInfo(
            code="llm.unsupported_capability",
            message=f"Model {self._config.model.id} does not support image input",
            retryable=False,
            kind="unsupported_capability",
            provider=self._config.model.provider,
            model=self._config.model.id,
            details={"capability": "vision"},
        )

        # 返回包含错误信息的助手消息
        return AssistantMessage(
            api=self._config.model.api,
            provider=self._config.model.provider,
            model=self._config.model.id,
            stop_reason="error",
            error_message=info.message,
            error_info=info,
        )

    async def _emit_llm_error(self, message: AssistantMessage) -> None:
        """
        检查并发射 LLM 错误事件

        如果助手消息的停止原因是 "error"，则发射错误事件通知外部系统。
        错误事件包含详细的错误信息，便于外部系统进行错误处理和重试决策。

        参数:
            message: 助手消息对象，需要检查是否包含错误

        事件发射:
            当消息包含错误时，发射 type="error" 的事件，包含以下字段：
            - error: 错误代码（如 "llm.rate_limit"）
            - message: 错误描述信息
            - source: 错误来源，固定为 "llm"
            - code: 错误代码（与 error 字段相同）
            - retryable: 是否可重试
            - provider: LLM 提供商名称
            - model: 模型 ID
            - statusCode: HTTP 状态码（如果有）
            - errorInfo: 完整的错误信息对象
        """
        # 只处理停止原因为 "error" 的消息
        if message.stop_reason != "error":
            return

        # 获取或构建错误信息
        info = message.error_info or LLMErrorInfo(
            code="llm.unknown",
            message=message.error_message or "Unknown LLM error",
            retryable=False,
            provider=self._config.model.provider,
            model=self._config.model.id,
        )

        # 发射错误事件
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
    """
    将当前任务信息注入系统提示词

    如果 Agent 上下文包含当前任务信息（current_task），并且系统提示词中
    尚未包含任务信息（通过 "## Current Task" 标记判断），则将任务信息
    追加到系统提示词末尾。

    这确保 LLM 在生成响应时能够了解当前正在执行的任务，从而提供更相关的结果。

    参数:
        context: Agent 上下文，包含系统提示词和当前任务信息

    返回:
        AgentContext: 更新后的上下文对象
        - 如果没有当前任务或已包含任务信息，返回原上下文
        - 否则返回系统提示词已更新的新上下文

    示例:
        原始系统提示词: "你是一个有用的助手。"
        当前任务: "## Current Task\n请帮我修复这个 bug。"
        结果: "你是一个有用的助手。\n\n## Current Task\n请帮我修复这个 bug。"
    """
    # 如果没有当前任务信息，直接返回原上下文
    if not context.current_task:
        return context

    # 检查系统提示词是否已包含任务标记
    marker = "## Current Task"
    system_prompt = context.system_prompt.rstrip()

    # 如果系统提示词中没有任务标记，将任务信息追加到末尾
    if marker not in system_prompt:
        system_prompt = f"{system_prompt}\n\n{context.current_task}".strip()

    # 返回更新后的上下文
    return AgentContext(
        system_prompt=system_prompt,
        messages=context.messages,
        tools=context.tools,
        current_task=context.current_task,
        task_recovery_projection=context.task_recovery_projection,
        task_signal=context.task_signal,
    )
