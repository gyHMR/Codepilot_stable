from __future__ import annotations

"""
兼容导出：RunStore 已迁移到 sessions.persistence.run_store。

TODO(sessions-cleanup): 后续清理旧导入时，可以删除本模块。
"""

from .persistence.run_store import FreshnessResult, RunStore


__all__ = ["FreshnessResult", "RunStore"]
