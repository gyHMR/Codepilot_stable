from __future__ import annotations

"""
兼容导出：SessionStore 已迁移到 sessions.persistence.store。

TODO(sessions-cleanup): 后续清理旧导入时，可以删除本模块。
"""

from .persistence.store import SessionStore, new_session_id


__all__ = ["SessionStore", "new_session_id"]
