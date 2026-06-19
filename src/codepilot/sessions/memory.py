from __future__ import annotations

"""Memory helpers for global MEMORY.md files."""

import logging
from pathlib import Path

logger = logging.getLogger("codepilot.sessions.memory")


def load_global_memory(workspace_dir: str | Path) -> str:
    """Load global MEMORY.md."""
    path = Path(workspace_dir) / ".codepilot" / "MEMORY.md"
    return _read_memory_file(path)


def save_global_memory(workspace_dir: str | Path, content: str) -> None:
    """Save global MEMORY.md."""
    path = Path(workspace_dir) / ".codepilot" / "MEMORY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
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
