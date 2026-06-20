from __future__ import annotations

"""
兼容导出：runtime.factory 已改名为 runtime.assembly。

TODO(runtime-cleanup): 后续清理旧导入时，可以删除本模块，并将调用方统一改为
codepilot.runtime.assembly。
"""

from .assembly import (
    UnknownRuntimeConfigKeyError,
    assemble_runtime,
    create_agent_session,
    explain_runtime_config,
)


__all__ = [
    "UnknownRuntimeConfigKeyError",
    "assemble_runtime",
    "create_agent_session",
    "explain_runtime_config",
]
