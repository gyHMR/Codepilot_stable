from __future__ import annotations

# 新手导读：redact.py 集中做敏感字段脱敏，避免日志和报告泄露 token/key。
# 关注点：新增审计输出前，应先确认是否需要经过这里。

"""Small, shared redaction helper for run artifacts."""

import os
import re
from typing import Any


_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "secret",
    "password",
    "passwd",
    "cookie",
    "credential",
    "private_key",
)
_SENSITIVE_KEYS = {
    "token",
    "access_token",
    "refresh_token",
    "id_token",
    "api_token",
    "auth_token",
}
_INLINE_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|password|passwd|secret|token)\s*=\s*[^,\s;]+"
)


def redact_artifact(value: Any) -> Any:
    """Redact common secret keys and in-process secret values."""

    secret_values = {
        secret
        for key, secret in os.environ.items()
        if secret and _is_sensitive_key(key)
    }
    return _redact(value, secret_values, key="")


def _redact(value: Any, secret_values: set[str], *, key: str) -> Any:
    if _is_sensitive_key(key):
        return "<redacted>"
    if isinstance(value, dict):
        return {
            str(item_key): _redact(item, secret_values, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_redact(item, secret_values, key="") for item in value]
    if isinstance(value, str):
        redacted = value
        for secret in secret_values:
            redacted = redacted.replace(secret, "<redacted>")
        redacted = _INLINE_SECRET_RE.sub(lambda match: f"{match.group(1)}=<redacted>", redacted)
        return redacted
    return value


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower()
    return normalized in _SENSITIVE_KEYS or any(
        part in normalized for part in _SENSITIVE_KEY_PARTS
    )


__all__ = ["redact_artifact"]
