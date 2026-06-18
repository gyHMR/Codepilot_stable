from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ErrorSource = Literal["llm", "tool", "runtime", "session", "interface"]
LLMErrorKind = Literal[
    "auth",
    "rate_limit",
    "timeout",
    "network",
    "context_length",
    "provider_response",
    "unsupported_capability",
    "unknown",
]


@dataclass
class ErrorInfo:
    """Structured error payload that can travel across layers."""

    code: str
    message: str
    retryable: bool = False
    source: ErrorSource = "runtime"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMErrorInfo(ErrorInfo):
    """Structured model-provider error information."""

    kind: LLMErrorKind = "unknown"
    source: ErrorSource = "llm"
    provider: str = ""
    model: str = ""
    status_code: int | None = None


__all__ = [
    "ErrorInfo",
    "ErrorSource",
    "LLMErrorInfo",
    "LLMErrorKind",
]
