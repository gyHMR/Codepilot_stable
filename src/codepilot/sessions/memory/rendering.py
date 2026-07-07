from __future__ import annotations

# 新手导读：rendering.py 把结构化记忆渲染成模型可读文本。
# 关注点：它控制记忆进入 prompt 时的表达方式。

"""Render canonical Memory v4 records for prompt and CLI surfaces."""

from .records import MemoryRecord


def render_memory(record: MemoryRecord) -> str:
    return (
        f"[{record.type}/{record.scope}/{record.confidence}] "
        f"{record.content} ({record.id})"
    )


__all__ = ["render_memory"]
