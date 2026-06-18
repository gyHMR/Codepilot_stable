from __future__ import annotations

from typing import Any

import httpx

from codepilot.protocols import LLMErrorInfo, LLMErrorKind, Model


def classify_llm_error(exc: Exception, model: Model) -> LLMErrorInfo:
    """Convert provider/http exceptions into structured LLM error info."""

    kind: LLMErrorKind = "unknown"
    retryable = False
    status_code: int | None = None
    details: dict[str, Any] = {"exception": type(exc).__name__}

    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        details["response_text"] = exc.response.text[:1000]
        if status_code in {401, 403}:
            kind = "auth"
        elif status_code == 429:
            kind = "rate_limit"
            retryable = True
        elif status_code in {408, 500, 502, 503, 504}:
            kind = "provider_response"
            retryable = True
        elif status_code in {400, 413} and "context" in exc.response.text.lower():
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
