"""统一事件流容器 —— 封装 provider 的流式响应。

本模块提供了 LLM 流式响应的统一事件流抽象。
各 provider 将自己的 SSE 事件推送到事件流中，上层消费者通过两种方式消费：

消费方式：
    1) async for 逐个消费事件（适用于流式渲染）
    2) await result() 获取最终 AssistantMessage（适用于非流式场景）

核心类：
    AssistantMessageEventStream: 异步事件流，支持推送、迭代和等待结果

辅助函数：
    classify_llm_error: 将 HTTP/provider 异常转换为协议层 LLMErrorInfo
    redact_llm_error_text: 移除错误文本中的凭据信息
"""

import asyncio
import re
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any, AsyncIterator, Literal, Optional, TypedDict, cast

import httpx

from codepilot.protocols import (
    AssistantMessage,
    LLMErrorInfo,
    LLMErrorKind,
    Model,
    ThinkingLevel,
    ToolCall,
)

from .estimation import (
    CHARS_PER_TOKEN,
    IMAGE_TOKEN_ESTIMATE,
    TOOL_SCHEMA_TOKEN_ESTIMATE,
    estimate_context_tokens,
    estimate_message_tokens,
    is_context_overflow,
    overflow_ratio,
)


# ── 哨兵对象 ──────────────────────────────────────────────────────────────────
# 标记事件流结束，用于内部队列通信
_SENTINEL = object()


# ── 事件类型 ──────────────────────────────────────────────────────────────────


LLMStreamEventType = Literal[
    "start",          # 流开始
    "text_start",     # 文本块开始
    "text_delta",     # 文本增量
    "text_end",       # 文本块结束
    "thinking_start", # 思考块开始
    "thinking_delta",  # 思考增量
    "thinking_end",   # 思考块结束
    "toolcall_start", # 工具调用块开始
    "toolcall_delta", # 工具调用增量
    "toolcall_end",   # 工具调用块结束
    "done",           # 流结束（成功）
    "error",          # 流结束（错误）
]


@dataclass
class StreamOptions:
    """流式调用选项 —— 完整的请求配置。

    参数:
        temperature: 温度参数
        max_tokens: 最大输出 token 数
        api_key: API 密钥（可选，不提供则从环境变量读取）
        headers: 自定义 HTTP 头
        timeout_seconds: 超时时间（秒）
        session_id: 会话 ID（用于追踪）
    """
    temperature: float | None = None
    max_tokens: int | None = None
    api_key: str | None = None
    headers: dict[str, str] | None = None
    timeout_seconds: float = 120.0
    session_id: str | None = None


@dataclass
class SimpleStreamOptions(StreamOptions):
    """简化的流式调用选项 —— 在 StreamOptions 基础上添加推理级别。

    参数:
        reasoning: 推理级别（None=不启用，其他值见 ThinkingLevel）
    """
    reasoning: ThinkingLevel | None = None


class LLMStreamEvent(TypedDict, total=False):
    """LLM 流式事件字典 —— 标准化的事件格式。

    provider 的流式响应被转换为此类字典，
    AssistantMessageEventStream 的事件队列以这种格式传递。
    """
    type: LLMStreamEventType
    partial: AssistantMessage
    contentIndex: int
    delta: str
    content: str
    toolCall: ToolCall
    reason: str
    message: AssistantMessage
    error: AssistantMessage
    errorInfo: LLMErrorInfo
    raw: Any


def llm_event(event_type: LLMStreamEventType, **payload: object) -> LLMStreamEvent:
    """构建标准化的 LLM 流式事件。

    参数:
        event_type: 事件类型（如 "text_delta"、"toolcall_end"）
        **payload: 事件负载数据

    返回:
        标准化的流式事件字典
    """
    return {"type": event_type, **payload}  # type: ignore[typeddict-item]


# ── 流容器 ──────────────────────────────────────────────────────────────────


class AssistantMessageEventStream:
    """异步事件流 —— 用于 LLM 流式响应的统一抽象。

    Provider 将 SSE 数据解析后通过 push() 推送到事件队列中，
    消费者通过两种方式消费：

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

    设计要点：
    - 使用 asyncio.Queue 做事件缓冲，支持生产/消费解耦
    - 使用 Future 让 await result() 可以等待最终消息
    - 支持异常结束（fail）：消费者可以选择捕获或降级
    - 支持后台任务绑定：provider 的 long-running 协程异常会自动触发 fail
    """

    def __init__(self) -> None:
        # 事件队列：用于迭代消费（async for）
        self._queue: asyncio.Queue[LLMStreamEvent | object] = asyncio.Queue()
        # 最终结果 Future：用于一次性获取完整消息（await result()）
        self._result: "asyncio.Future[AssistantMessage]" = asyncio.get_running_loop().create_future()
        self._closed = False
        self._background_task: asyncio.Task[None] | None = None

    def push(self, event: LLMStreamEvent) -> None:
        """推送一个事件到队列（text_delta/toolcall_delta/...）。

        如果流已关闭，事件会被静默丢弃。

        参数:
            event: 标准化的事件字典
        """
        if self._closed:
            return
        self._queue.put_nowait(event)

    def start_background(self, coroutine: Coroutine[Any, Any, None]) -> None:
        """启动 provider 后台任务，并将意外异常绑定到当前事件流。

        provider 通过此方法启动异步 SSE 消费循环，
        如果该协程抛出异常，会自动调用 self.fail() 传播到消费者。

        参数:
            coroutine: provider 的异步协程

        抛出:
            RuntimeError: 如果后台任务已经启动
        """
        if self._background_task is not None:
            raise RuntimeError("Provider background task already started")
        task = asyncio.create_task(coroutine)
        self._background_task = task
        task.add_done_callback(self._handle_background_done)

    async def aclose(self) -> None:
        """取消尚未完成的 Provider 工作并关闭事件流；重复调用保持幂等。"""
        task = self._background_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if self._closed:
            return
        self._closed = True
        if not self._result.done():
            self._result.cancel()
        self._queue.put_nowait(_SENTINEL)

    def _handle_background_done(self, task: asyncio.Task[None]) -> None:
        """处理后台任务完成：如果任务异常则调用 fail()。

        start_background 的 done_callback，在后台协程异常退出时
        自动将异常传播到事件流。
        """
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self.fail(error)

    def end(self, message: AssistantMessage) -> None:
        """正常结束：写入最终消息并关闭流。

        参数:
            message: 完整的 AssistantMessage（包含所有累积的内容）
        """
        if self._closed:
            return
        self._closed = True
        if not self._result.done():
            self._result.set_result(message)
        # 推送哨兵对象标记结束（消费者迭代到哨兵会停止）
        self._queue.put_nowait(_SENTINEL)

    def fail(self, error: Exception, fallback: Optional[AssistantMessage] = None) -> None:
        """异常结束。

        参数:
            error: 导致失败的异常
            fallback: 可选的降级消息。如果提供，result() 仍返回 fallback；
                      否则 result() 会抛出异常
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
        """等待并返回最终的 AssistantMessage。

        用于非流式场景：等待 end() 设置最终结果后返回。
        如果流异常结束且没有 fallback，会抛出异常。

        返回:
            完整的 AssistantMessage
        """
        return await self._result

    def __aiter__(self) -> AsyncIterator[LLMStreamEvent]:
        """支持 async for 迭代消费事件。

        使用方式：
            async for event in stream:
                if event["type"] == "text_delta":
                    print(event["delta"])
        """
        return self._iter_events()

    async def _iter_events(self) -> AsyncIterator[LLMStreamEvent]:
        """内部迭代器：从队列中逐个取出事件，遇到哨兵对象则结束。"""
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                break
            yield cast(LLMStreamEvent, item)


# ── 辅助函数 ──────────────────────────────────────────────────────────────────


def _safe_response_text(response: httpx.Response, limit: int = 1000) -> str:
    """安全地从 HTTP 响应中提取文本内容，最多取 limit 字节。"""
    try:
        content = response.content
        return content[:limit].decode(response.encoding or "utf-8", errors="replace")
    except (httpx.ResponseNotRead, httpx.StreamConsumed):
        return ""


def redact_llm_error_text(value: object, limit: int = 1000) -> str:
    """移除 provider 错误文本中的常见凭据格式。

    脱敏规则：
    - Bearer token（不区分大小写）
    - API Key / x-api-key / Authorization
    - sk- 开头的密钥（OpenAI 风格）

    参数:
        value: 原始错误文本
        limit: 截断长度

    返回:
        脱敏后的文本
    """
    text = str(value)[:limit]
    text = re.sub(r"(?i)(bearer\s+)[^\s\"']+", r"\1[REDACTED]", text)
    text = re.sub(
        r"(?i)((?:api[_-]?key|x-api-key|authorization)[\"']?\s*[:=]\s*[\"']?)[^\s\"',}]+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    return text


def classify_llm_error(exc: Exception, model: Model) -> LLMErrorInfo:
    """将 provider/HTTP 异常转换为协议层 LLMErrorInfo。

    根据异常类型和 HTTP 状态码分类：
    - 401/403 → auth（认证错误）
    - 429 → rate_limit（限流，可重试）
    - 408/500/502/503/504 → provider_response（服务端错误，可重试）
    - 400/413 + "context" → context_length（上下文超长）
    - 超时 → timeout（可重试）
    - 网络错误 → network（可重试）

    参数:
        exc: 发生的异常
        model: 调用的模型（用于填充 provider/model 信息）

    返回:
        标准化后的 LLMErrorInfo
    """
    kind: LLMErrorKind = "unknown"
    retryable = False
    status_code: int | None = None
    details: dict[str, Any] = {"exception_type": type(exc).__name__}

    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        response_text = redact_llm_error_text(
            getattr(exc, "_response_text", "") or _safe_response_text(exc.response)
        )
        if response_text:
            details["response_excerpt"] = response_text
            details["response_text"] = response_text
        request_id = (
            exc.response.headers.get("x-request-id")
            or exc.response.headers.get("request-id")
            or exc.response.headers.get("cf-ray")
        )
        if request_id:
            details["provider_request_id"] = request_id
        retry_after = exc.response.headers.get("retry-after")
        if retry_after:
            details["retry_after"] = retry_after
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
        message=redact_llm_error_text(exc),
        retryable=retryable,
        kind=kind,
        provider=model.provider,
        model=model.id,
        status_code=status_code,
        details=details,
    )


__all__ = [
    "AssistantMessageEventStream",
    "CHARS_PER_TOKEN",
    "IMAGE_TOKEN_ESTIMATE",
    "LLMStreamEvent",
    "LLMStreamEventType",
    "SimpleStreamOptions",
    "StreamOptions",
    "TOOL_SCHEMA_TOKEN_ESTIMATE",
    "classify_llm_error",
    "estimate_context_tokens",
    "estimate_message_tokens",
    "is_context_overflow",
    "llm_event",
    "overflow_ratio",
    "redact_llm_error_text",
]
