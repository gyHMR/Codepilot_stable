from __future__ import annotations

"""固定 MEMORY.md 辅助工具和记忆文本脱敏。"""

import logging
import re
from pathlib import Path


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


def load_global_memory(workspace_dir: str | Path) -> str:
    """加载用户维护的固定记忆文件 `.codepilot/MEMORY.md`。"""

    path = Path(workspace_dir) / ".codepilot" / "MEMORY.md"
    return _read_memory_file(path)


def save_global_memory(workspace_dir: str | Path, content: str) -> None:
    """保存固定 MEMORY.md（自动记忆机制不会调用此函数，仅用户手动操作）。"""

    path = Path(workspace_dir) / ".codepilot" / "MEMORY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    logger.info("global memory saved chars=%d", len(content))


def _read_memory_file(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            logger.debug("loaded memory file=%s chars=%d", path, len(text))
        return text
    except Exception as exc:
        logger.warning("failed to read memory file=%s: %s", path, exc)
        return ""


__all__ = [
    "load_global_memory",
    "sanitize_memory_text",
    "save_global_memory",
]
