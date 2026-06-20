from __future__ import annotations

"""
兼容导出：动态上下文编译已迁移到 sessions.context.compiler。

TODO(sessions-cleanup): 后续清理旧导入时，可以删除本模块。
"""

from .context.compiler import ContextCompiler, ContextPolicy


__all__ = ["ContextCompiler", "ContextPolicy"]
