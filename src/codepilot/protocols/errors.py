from __future__ import annotations

# 新手导读：errors.py 定义跨层可共享的错误信息结构。
# 关注点：稳定错误结构能帮助 CLI/Web/RPC 用一致方式展示失败。

"""
错误信息类型定义。

定义了跨层传递的结构化错误载荷：
- ErrorInfo: 通用错误信息基类
- LLMErrorInfo: LLM provider 特有的错误信息（继承自 ErrorInfo）

通过结构化的错误信息，上层可以根据错误类型（认证、限流、超时等）
做出不同的处理策略（重试、降级、报错等）。
"""

from dataclasses import dataclass, field
from typing import Any, Literal, cast


# 错误来源：标识错误发生在系统的哪个层面
ErrorSource = Literal["llm", "tool", "runtime", "session", "interface"]

# LLM 错误类型：细分 LLM 调用过程中可能遇到的错误类别
LLMErrorKind = Literal[
    "auth",                    # 认证失败（API Key 无效或缺失）
    "rate_limit",              # 速率限制（429）
    "timeout",                 # 请求超时
    "network",                 # 网络连接错误
    "context_length",          # 上下文长度超限
    "provider_response",       # 服务端错误（5xx 等）
    "unsupported_capability",  # 模型不支持的能力
    "unknown",                 # 未知错误
]
_ERROR_SOURCES = frozenset({"llm", "tool", "runtime", "session", "interface"})
_LLM_ERROR_KINDS = frozenset(
    {
        "auth",
        "rate_limit",
        "timeout",
        "network",
        "context_length",
        "provider_response",
        "unsupported_capability",
        "unknown",
    }
)


@dataclass
class ErrorInfo:
    """通用结构化错误信息。

    可跨系统各层传递，携带错误代码、消息、是否可重试等关键信息。

    Attributes:
        code: 错误代码（如 "llm.auth"、"tool.not_found"）。
        message: 人类可读的错误描述。
        retryable: 是否可重试（True 时上层可尝试自动重试）。
        source: 错误来源层面。
        details: 附加详情字典（如异常类型、响应文本等）。
    """

    code: str
    message: str
    retryable: bool = False
    source: ErrorSource = "runtime"
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "code",
            _require_error_text(self.code, field_name="error code"),
        )
        object.__setattr__(
            self,
            "message",
            _require_error_text(self.message, field_name="error message"),
        )
        object.__setattr__(
            self,
            "source",
            _ensure_error_source(self.source),
        )
        if not isinstance(self.details, dict):
            raise TypeError("Error details must be a dict")


@dataclass
class LLMErrorInfo(ErrorInfo):
    """LLM Provider 特有的结构化错误信息。

    继承自 ErrorInfo，额外包含 provider、model、HTTP 状态码等 LLM 特有字段。

    Attributes:
        kind: LLM 错误类型细分。
        source: 固定为 "llm"。
        provider: 出错的 provider 名称。
        model: 出错的模型 ID。
        status_code: HTTP 状态码（非 HTTP 错误时为 None）。
    """

    kind: LLMErrorKind = "unknown"
    source: ErrorSource = "llm"
    provider: str = ""
    model: str = ""
    status_code: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.source != "llm":
            raise ValueError("LLM error source must be llm")
        object.__setattr__(self, "kind", _ensure_llm_error_kind(self.kind))
        object.__setattr__(self, "provider", _clean_error_text(self.provider))
        object.__setattr__(self, "model", _clean_error_text(self.model))
        if (
            self.status_code is not None
            and (isinstance(self.status_code, bool) or not isinstance(self.status_code, int))
        ):
            raise TypeError("LLM status_code must be int or None")


def _clean_error_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _require_error_text(value: object, *, field_name: str) -> str:
    text = _clean_error_text(value)
    if not text:
        raise ValueError(f"Structured {field_name} cannot be empty")
    return text


def _ensure_error_source(value: object) -> ErrorSource:
    text = _clean_error_text(value)
    if text not in _ERROR_SOURCES:
        raise ValueError(f"Unknown error source: {value}")
    return cast(ErrorSource, text)


def _ensure_llm_error_kind(value: object) -> LLMErrorKind:
    text = _clean_error_text(value)
    if text not in _LLM_ERROR_KINDS:
        raise ValueError(f"Unknown LLM error kind: {value}")
    return cast(LLMErrorKind, text)


__all__ = [
    "ErrorInfo",
    "ErrorSource",
    "LLMErrorInfo",
    "LLMErrorKind",
]
