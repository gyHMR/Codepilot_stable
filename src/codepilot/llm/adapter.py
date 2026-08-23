"""ModelPort 适配器 —— 将现有 provider registry 适配为 Core 可消费的 ModelPort。

ProviderModelPort 是 ModelPort 协议的标准实现，它将：
1. LLMRequest（core 层请求）转换为 provider 的流式调用
2. provider 的响应流转换为 core 层的 LLMEvent 事件流
3. 处理消息格式转换、API Key 解析、模型能力验证等跨切面逻辑
"""

from collections.abc import AsyncIterator
from dataclasses import replace

from codepilot.protocols import (
    Context,
    ImageContent,
    LLMErrorInfo,
    Message,
    Model,
)

from .ports import (
    LLMCompleted,
    LLMEvent,
    LLMFailed,
    LLMReasoningDelta,
    LLMRequest,
    LLMStarted,
    LLMTextDelta,
    LLMToolCallDelta,
    ModelPort,
)
from .provider_types import (
    ProviderApiKeyResolver,
    ProviderCompleteFn,
    ProviderMessageConverter,
    ProviderSimpleStreamFn,
)
from .registry import ApiProviderRegistry, complete_simple, stream_simple
from .stream import SimpleStreamOptions, classify_llm_error


class ProviderModelPort(ModelPort):
    """ModelPort 适配器 —— 在现有 provider registry 或注入的流函数之上适配。

    支持两种装配方式：
    1. 通过 registry: 传入 ApiProviderRegistry 实例
    2. 通过注入函数: 传入 stream_fn / complete_fn / convert_messages 等

    参数:
        model: 模型配置
        stream_fn: 流式调用函数（可选，不提供则使用 registry）
        complete_fn: 非流式调用函数（可选）
        convert_messages: 消息转换函数（可选，如 OpenAI ↔ Anthropic 互转）
        get_api_key: API Key 解析函数（可选）
        registry: Provider 注册中心（可选）
    """

    def __init__(
        self,
        *,
        model: Model,
        stream_fn: ProviderSimpleStreamFn | None = None,
        complete_fn: ProviderCompleteFn | None = None,
        convert_messages: ProviderMessageConverter | None = None,
        get_api_key: ProviderApiKeyResolver | None = None,
        proxy_url: str | None = None,
        registry: ApiProviderRegistry | None = None,
    ) -> None:
        self._model = model
        self._stream_fn = stream_fn
        self._complete_fn = complete_fn
        self._convert_messages = convert_messages
        self._get_api_key = get_api_key
        self._proxy_url = proxy_url
        self._registry = registry

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        """执行一次流式 LLM 调用。

        处理流程:
        1. 消息转换（如有 convert_messages）
        2. 验证模型能力（如图片支持）
        3. 构造 Context（包含 system_prompt、messages、tools）
        4. 解析 API Key
        5. 根据模型能力选择流式或非流式调用
        6. 将 provider 事件转换为 core 层 LLMEvent

        参数:
            request: LLM 请求

        返回:
            LLMEvent 的异步迭代器
        """
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
                    request.options.reasoning
                    if capabilities.reasoning
                    else None
                ),
                temperature=request.options.temperature,
                max_tokens=request.options.max_tokens,
                timeout_seconds=request.options.timeout_seconds,
                api_key=api_key,
                proxy_url=self._proxy_url,
                session_id=request.correlation.session_id or None,
            )
            if not capabilities.streaming:
                # 非流式模式：直接调用 complete 并返回
                complete_fn = self._complete_fn or (
                    self._registry.complete_simple if self._registry is not None else complete_simple
                )
                message = complete_fn(self._model, context, options)
                completed = await message if _is_awaitable(message) else message
                if completed.error_info is not None:
                    yield LLMFailed(error=_correlate_error(completed.error_info, request))
                else:
                    yield LLMCompleted(message=completed, usage=completed.usage)
                return
            # 流式模式
            stream_fn = self._stream_fn or (
                self._registry.stream_simple if self._registry is not None else stream_simple
            )
            response = stream_fn(self._model, context, options)
            response_stream = await response if _is_awaitable(response) else response
            try:
                async for event in response_stream:
                    event_type = event.get("type")
                    if event_type == "start":
                        yield LLMStarted()
                    elif event_type == "text_delta":
                        yield LLMTextDelta(text=str(event.get("delta", "")))
                    elif event_type == "thinking_delta":
                        yield LLMReasoningDelta(text=str(event.get("delta", "")))
                    elif event_type == "toolcall_end" and event.get("toolCall") is not None:
                        yield LLMToolCallDelta(tool_call=event["toolCall"])
                message = await response_stream.result()
                if message.error_info is not None:
                    yield LLMFailed(error=_correlate_error(message.error_info, request))
                    return
                yield LLMCompleted(message=message, usage=message.usage)
            finally:
                await response_stream.aclose()
        except Exception as exc:
            yield LLMFailed(
                error=_correlate_error(classify_llm_error(exc, self._model), request)
            )


def _is_awaitable(value: object) -> bool:
    """检查值是否为 awaitable（有 __await__ 方法）。"""
    return hasattr(value, "__await__")


def _correlate_error(error: LLMErrorInfo, request: LLMRequest) -> LLMErrorInfo:
    """将错误信息与请求关联 —— 注入 run_id 和 session_id。"""
    details = dict(error.details)
    if request.correlation.run_id:
        details["run_id"] = request.correlation.run_id
    if request.correlation.session_id:
        details["session_id"] = request.correlation.session_id
    return replace(error, details=details)


def _validate_capabilities(
    messages: list[Message],
    *,
    model: Model,
) -> LLMErrorInfo | None:
    """验证模型能力是否满足消息需求。

    当前检查：如果消息中包含图片但模型不支持 vision，返回错误。

    参数:
        messages: 消息列表
        model: 模型配置

    返回:
        如果能力不足，返回 LLMErrorInfo；否则返回 None
    """
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
