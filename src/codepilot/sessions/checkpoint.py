from __future__ import annotations

"""
兼容导出：检查点逻辑已迁移到 sessions.history.checkpoint。

TODO(sessions-cleanup): 后续清理旧导入时，可以删除本模块。
"""

from .history.checkpoint import SessionCheckpoint, record_checkpoint


__all__ = ["SessionCheckpoint", "record_checkpoint"]
