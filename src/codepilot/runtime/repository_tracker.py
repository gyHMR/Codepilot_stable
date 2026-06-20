from __future__ import annotations

"""
兼容导出：动态仓库快照追踪已迁移到 sessions.repository_tracker。

RuntimeAssembly 只保存创建会话时的静态 repository bootstrap；每次模型调用前的
仓库 fingerprint、顶层目录变化、dirty 文件变化和 instruction 变化，属于
Session 动态上下文治理。

TODO(runtime-cleanup): 后续清理旧导入时，可以删除本模块，并将调用方统一改为
codepilot.sessions.repository_tracker。
"""

from codepilot.sessions.repository_tracker import (
    RepositoryTracker,
    compare_snapshots,
    render_repository_snapshot,
)


__all__ = [
    "RepositoryTracker",
    "compare_snapshots",
    "render_repository_snapshot",
]
