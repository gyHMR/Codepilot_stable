from __future__ import annotations

"""
兼容导出：上下文状态已迁移到 sessions.context.state。

TODO(sessions-cleanup): 后续清理旧导入时，可以删除本模块。
"""

from .context.state import ActiveFile, ContextEvidence, FileSummary, SessionContextState


__all__ = [
    "ActiveFile",
    "ContextEvidence",
    "FileSummary",
    "SessionContextState",
]
