from __future__ import annotations

# 新手导读：包门面文件：集中导出本层最常用的类型和入口，降低学习时的导入成本。
# 关注点：sessions 层是会话事实源，负责消息、run、记忆、上下文投影和任务恢复。

"""Session orchestration package.

The sessions layer owns four small domains:

- persistence: session/run facts and filesystem layout
- context: per-turn ContextGovernor prompt projection
- memory: durable project/session memory
- history: task recovery, branching, and lightweight git rollback
"""

from .context import ContextGovernor, RepositoryBootstrap, RepositoryTracker
from .history import GitRollbackAction, GitRollbackPlan, GitRollbackResult, SessionCheckpoint
from .layout import SessionLayout
from .memory import (
    MemoryQuery,
    MemoryRecord,
    MemoryRetriever,
    MemoryStore,
    MemoryWriter,
    RetrievedMemory,
    load_global_memory,
    save_global_memory,
)
from .persistence import FreshnessResult, RunStore, SessionStore, new_session_id
from .session import AgentSession
from .types import (
    AgentSessionOptions,
    CommandHandler,
    ConvertToLlmFn,
    LifecycleHook,
    RegisteredCommand,
    SessionCommandContext,
    SessionLifecycleContext,
)

__all__ = [
    "AgentSession",
    "AgentSessionOptions",
    "CommandHandler",
    "ConvertToLlmFn",
    "LifecycleHook",
    "RegisteredCommand",
    "SessionCommandContext",
    "SessionLifecycleContext",
    "ContextGovernor",
    "SessionCheckpoint",
    "GitRollbackAction",
    "GitRollbackPlan",
    "GitRollbackResult",
    "SessionLayout",
    "SessionStore",
    "FreshnessResult",
    "RunStore",
    "RepositoryBootstrap",
    "RepositoryTracker",
    "new_session_id",
    "load_global_memory",
    "save_global_memory",
    "MemoryQuery",
    "MemoryRecord",
    "MemoryRetriever",
    "MemoryStore",
    "MemoryWriter",
    "RetrievedMemory",
]
