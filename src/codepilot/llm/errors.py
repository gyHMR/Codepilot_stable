from __future__ import annotations

"""LLM 错误分类：将 provider/http 异常转换为结构化的 LLMErrorInfo。"""

from typing import Any

import httpx

from codepilot.protocols import LLMErrorInfo, LLMErrorKind, Model


def _safe_response_text(response: httpx.Response, limit: int = 1000) -> str:
    """安全读取已缓存的 HTTP 响应体，避免错误处理再次抛出异常。"""

    try:
        return response.text[:limit]
    except (httpx.ResponseNotRead, httpx.StreamConsumed):
        return ""


def classify_llm_error(exc: Exception, model: Model) -> LLMErrorInfo:
    """将 provider/http 异常分类为结构化的 LLMErrorInfo（含错误类型、是否可重试等）。"""

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


__all__ = ["classify_llm_error", "LLMErrorInfo", "LLMErrorKind"]
