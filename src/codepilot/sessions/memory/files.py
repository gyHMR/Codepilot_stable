from __future__ import annotations

# 新手导读：memory/files.py 负责记忆文件路径和 JSONL 读写细节。
# 关注点：业务判断不放这里，避免文件 IO 和策略混在一起。

"""记忆文本脱敏。"""

import logging
import re


logger = logging.getLogger("codepilot.sessions.memory")

_SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|token|password|secret|cookie)\s*[:=]\s*\S+"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
    re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?i)authorization:\s*bearer\s+\S+"),
]


def sanitize_memory_text(text: str, *, limit: int) -> str:
    """脱敏记忆文本：移除密钥、token 等敏感信息后截断。"""
    safe = text
    for pattern in _SECRET_PATTERNS:
        safe = pattern.sub("[REDACTED]", safe)
    safe = safe.replace("\x00", "").strip()
    return safe[:limit]


__all__ = [
    "sanitize_memory_text",
]
