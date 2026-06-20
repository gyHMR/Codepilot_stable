from __future__ import annotations

"""
兼容导出：动态任务上下文编译已迁移到 sessions.context_compiler。

Runtime 只负责在装配时把 ContextCompiler 接入 AgentSessionOptions.prepare_context；
具体的 active files、recent evidence、memory retrieval、仓库新鲜度和 token 预算裁剪
属于 Session/Run 期间的动态上下文治理。

TODO(runtime-cleanup): 后续清理旧导入时，可以删除本模块，并将调用方统一改为
codepilot.sessions.context.compiler。
"""

from codepilot.sessions.context.compiler import ContextCompiler, ContextPolicy


__all__ = ["ContextCompiler", "ContextPolicy"]
