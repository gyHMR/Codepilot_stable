from __future__ import annotations

# 新手导读：AssistantMessageEventStream 统一封装 provider 的流式响应。
# 关注点：上层可以 async for 消费事件，也可以 await result() 获取最终消息。

"""
统一事件流容器模块。

本模块提供了 LLM 流式响应的统一事件流抽象。
各 provider 将自己的 SSE 事件推送到事件流中，上层消费者通过两种方式消费：

消费方式：
    1) async for 逐个消费事件（适用于流式渲染）
    2) await result() 获取最终 AssistantMessage（适用于非流式场景）

核心类：
    AssistantMessageEventStream: 异步事件流，支持推送、迭代和等待结果
"""

import asyncio
from collections.abc import Coroutine
from typing import Any, AsyncIterator, Optional, cast

import httpx

from codepilot.protocols import (
    AssistantMessage,
    Context,
    ImageContent,
    LLMErrorInfo,
    LLMErrorKind,
    LLMStreamEvent,
    LLMStreamEventType,
    Message,
    Model,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


# 哨兵对象：标记事件流结束
_SENTINEL = object()


def llm_event(event_type: LLMStreamEventType, **payload: object) -> LLMStreamEvent:
    """构建标准化的 LLM 流式事件。

    Args:
        event_type: 事件类型（如 "text_delta"、"toolcall_end"）。
        **payload: 事件负载数据。

    Returns:
        LLMStreamEvent: 标准化的流式事件字典。
    """
    return {"type": event_type, **payload}  # type: ignore[typeddict-item]


class AssistantMessageEventStream:
    """异步事件流：用于 LLM 流式响应的统一抽象。

    使用方式：
        # Provider 侧（推送事件）
        stream = AssistantMessageEventStream()
        stream.push({"type": "text_start", ...})
        stream.push({"type": "text_delta", "delta": "Hello"})
        stream.end(final_message)

        # 消费者侧（方式一：逐个消费）
        async for event in stream:
            print(event["type"])

        # 消费者侧（方式二：等待最终结果）
        message = await stream.result()
    """

    def __init__(self) -> None:
        # 事件队列：用于迭代消费
        self._queue: asyncio.Queue[LLMStreamEvent | object] = asyncio.Queue()
        # 最终结果 Future：用于一次性获取完整消息
        self._result: "asyncio.Future[AssistantMessage]" = asyncio.get_event_loop().create_future()
        self._closed = False

    def push(self, event: LLMStreamEvent) -> None:
        """推送一个事件到队列（text_delta/toolcall_delta/...）。

        如果流已关闭，事件会被静默丢弃。
        """
        if self._closed:
            return
        self._queue.put_nowait(event)

    def start_background(self, coroutine: Coroutine[Any, Any, None]) -> None:
        """启动 provider 后台任务，并将意外异常绑定到当前事件流。

        Args:
            coroutine: provider 的异步协程。
        """
        task = asyncio.create_task(coroutine)
        task.add_done_callback(self._handle_background_done)

    def _handle_background_done(self, task: asyncio.Task[None]) -> None:
        """处理后台任务完成：如果任务异常则调用 fail()。"""
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self.fail(error)

    def end(self, message: AssistantMessage) -> None:
        """正常结束：写入最终消息并关闭流。"""
        if self._closed:
            return
        self._closed = True
        if not self._result.done():
            self._result.set_result(message)
        # 推送哨兵对象标记结束
        self._queue.put_nowait(_SENTINEL)

    def fail(self, error: Exception, fallback: Optional[AssistantMessage] = None) -> None:
        """异常结束。

        Args:
            error: 导致失败的异常。
            fallback: 可选的降级消息。如果提供，result() 仍返回 fallback；
                      否则 result() 会抛出异常。
        """
        if self._closed:
            return
        self._closed = True
        if fallback is not None:
            if not self._result.done():
                self._result.set_result(fallback)
        else:
            if not self._result.done():
                self._result.set_exception(error)
        self._queue.put_nowait(_SENTINEL)

    async def result(self) -> AssistantMessage:
        """等待并返回最终 AssistantMessage。"""
        return await self._result

    def __aiter__(self) -> AsyncIterator[LLMStreamEvent]:
        """支持 async for 迭代消费事件。"""
        return self._iter_events()

    async def _iter_events(self) -> AsyncIterator[LLMStreamEvent]:
        """内部迭代器：从队列中逐个取出事件，遇到哨兵对象则结束。"""
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                break
            yield cast(LLMStreamEvent, item)


def _safe_response_text(response: httpx.Response, limit: int = 1000) -> str:
    try:
        return response.text[:limit]
    except (httpx.ResponseNotRead, httpx.StreamConsumed):
        return ""


def classify_llm_error(exc: Exception, model: Model) -> LLMErrorInfo:
    """Convert provider/http exceptions into protocol-level LLMErrorInfo."""

    kind: LLMErrorKind = "unknown"
    retryable = False
    status_code: int | None = None
    details: dict[str, Any] = {"exception": type(exc).__name__}

    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        response_text = _safe_response_text(exc.response)
        if response_text:
            details["response_text"] = response_text
        if status_code in {401, 403}:
            kind = "auth"
        elif status_code == 429:
            kind = "rate_limit"
            retryable = True
        elif status_code in {408, 500, 502, 503, 504}:
            kind = "provider_response"
            retryable = True
        elif status_code in {400, 413} and "context" in response_text.lower():
            kind = "context_length"
        else:
            kind = "provider_response"
    elif isinstance(exc, httpx.TimeoutException):
        kind = "timeout"
        retryable = True
    elif isinstance(exc, httpx.NetworkError):
        kind = "network"
        retryable = True
    elif isinstance(exc, RuntimeError) and "api_key" in str(exc).lower():
        kind = "auth"

    return LLMErrorInfo(
        code=f"llm.{kind}",
        message=str(exc),
        retryable=retryable,
        kind=kind,
        provider=model.provider,
        model=model.id,
        status_code=status_code,
        details=details,
    )


CHARS_PER_TOKEN = 4
IMAGE_TOKEN_ESTIMATE = 1000
TOOL_SCHEMA_TOKEN_ESTIMATE = 200


def estimate_message_tokens(msg: Message) -> int:
    total = 0
    if isinstance(msg, UserMessage):
        if isinstance(msg.content, str):
            total += len(msg.content)
        else:
            for block in msg.content:
                if isinstance(block, TextContent):
                    total += len(block.text)
                elif isinstance(block, ImageContent):
                    total += IMAGE_TOKEN_ESTIMATE * CHARS_PER_TOKEN
    elif isinstance(msg, AssistantMessage):
        for block in msg.content:
            if isinstance(block, TextContent):
                total += len(block.text)
            elif isinstance(block, ThinkingContent):
                total += len(block.thinking)
            elif isinstance(block, ToolCall):
                total += len(str(block.arguments)) + len(block.name) + 20
    elif isinstance(msg, ToolResultMessage):
        for block in msg.content:
            if isinstance(block, TextContent):
                total += len(block.text)
            elif isinstance(block, ImageContent):
                total += IMAGE_TOKEN_ESTIMATE * CHARS_PER_TOKEN
    return max(1, total // CHARS_PER_TOKEN)


def estimate_context_tokens(
    messages: list[Message],
    system_prompt: str = "",
    tools: list | None = None,
) -> int:
    total = len(system_prompt) // CHARS_PER_TOKEN
    for msg in messages:
        total += estimate_message_tokens(msg)
    if tools:
        total += len(tools) * TOOL_SCHEMA_TOKEN_ESTIMATE
    return total


def is_context_overflow(
    model: Model,
    context: Context,
    *,
    safety_margin: float = 0.95,
) -> bool:
    limit = int(model.context_window * safety_margin)
    estimated = estimate_context_tokens(
        context.messages,
        context.system_prompt or "",
        context.tools,
    )
    return estimated > limit


def overflow_ratio(model: Model, context: Context) -> float:
    estimated = estimate_context_tokens(
        context.messages,
        context.system_prompt or "",
        context.tools,
    )
    if model.context_window <= 0:
        return 0.0
    return estimated / model.context_window


__all__ = [
    "AssistantMessageEventStream",
    "CHARS_PER_TOKEN",
    "IMAGE_TOKEN_ESTIMATE",
    "TOOL_SCHEMA_TOKEN_ESTIMATE",
    "classify_llm_error",
    "estimate_context_tokens",
    "estimate_message_tokens",
    "is_context_overflow",
    "llm_event",
    "overflow_ratio",
]
