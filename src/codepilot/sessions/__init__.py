"""
Codepilot 会话管理模块。

本包负责长期运行的 AgentSession 编排、持久化存储、消息序列化、
结构化记忆管理和上下文压缩等能力。
"""

from .context import ContextCompiler, ContextPolicy, RepositoryBootstrap, RepositoryTracker
from .history import SessionCheckpoint
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
from .types import AgentSessionOptions, ConvertToLlmFn

__all__ = [
    "AgentSession",
    "AgentSessionOptions",
    "ConvertToLlmFn",
    "ContextCompiler",
    "ContextPolicy",
    "SessionCheckpoint",
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
