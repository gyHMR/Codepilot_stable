from __future__ import annotations

"""
兼容导出：上下文压缩已迁移到 sessions.context.compaction。

TODO(sessions-cleanup): 后续清理旧导入时，可以删除本模块。
"""

from .context.compaction import (
    COMPACTION_SYSTEM_PROMPT,
    ContextCompactionResult,
    build_compacted_context,
    extract_text_from_assistant,
    extract_text_from_tool_result,
    extract_text_from_user,
    fallback_summary,
    format_messages_for_summary,
)


__all__ = [
    "COMPACTION_SYSTEM_PROMPT",
    "ContextCompactionResult",
    "build_compacted_context",
    "extract_text_from_assistant",
    "extract_text_from_tool_result",
    "extract_text_from_user",
    "fallback_summary",
    "format_messages_for_summary",
]
