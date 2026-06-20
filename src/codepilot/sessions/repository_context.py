from __future__ import annotations

"""
兼容导出：仓库启动上下文已迁移到 sessions.context.repository_context。

TODO(sessions-cleanup): 后续清理旧导入时，可以删除本模块。
"""

from .context.repository_context import (
    GitInfo,
    RepositoryBootstrap,
    build_repository_bootstrap,
    render_repository_context,
)


__all__ = [
    "GitInfo",
    "RepositoryBootstrap",
    "build_repository_bootstrap",
    "render_repository_context",
]
