from __future__ import annotations

"""
错误信息类型定义。

定义了跨层传递的结构化错误载荷：
- ErrorInfo: 通用错误信息基类
- LLMErrorInfo: LLM provider 特有的错误信息（继承自 ErrorInfo）

通过结构化的错误信息，上层可以根据错误类型（认证、限流、超时等）
做出不同的处理策略（重试、降级、报错等）。
"""

from dataclasses import dataclass, field
from typing import Any, Literal


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


__all__ = [
    "ErrorInfo",
    "ErrorSource",
    "LLMErrorInfo",
    "LLMErrorKind",
]
