from __future__ import annotations

"""Memory helpers for global and channel-level MEMORY.md files."""

import logging
from pathlib import Path

logger = logging.getLogger("codepilot.sessions.memory")


def load_global_memory(workspace_dir: str | Path) -> str:
    """Load global MEMORY.md."""
    path = Path(workspace_dir) / ".codepilot" / "MEMORY.md"
    return _read_memory_file(path)


def load_channel_memory(workspace_dir: str | Path, channel_id: str) -> str:
    """Load channel-level MEMORY.md."""
    path = Path(workspace_dir) / ".codepilot" / "im" / channel_id / "MEMORY.md"
    return _read_memory_file(path)


def load_merged_memory(workspace_dir: str | Path, channel_id: str | None = None) -> str:
    """Load and merge global and channel memory."""
    sections: list[str] = []
    global_mem = load_global_memory(workspace_dir)
    if global_mem:
        sections.append(f"## Global Memory\n{global_mem}")

    if channel_id:
        channel_mem = load_channel_memory(workspace_dir, channel_id)
        if channel_mem:
            sections.append(f"## Channel Memory ({channel_id})\n{channel_mem}")

    if not sections:
        return ""
    return "\n\n".join(sections)


def save_channel_memory(workspace_dir: str | Path, channel_id: str, content: str) -> None:
    """Save channel-level MEMORY.md."""
    path = Path(workspace_dir) / ".codepilot" / "im" / channel_id / "MEMORY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("channel memory saved channel_id=%s chars=%d", channel_id, len(content))


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
