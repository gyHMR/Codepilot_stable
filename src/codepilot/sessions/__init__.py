"""
Session management for Codepilot.

This package owns long-lived AgentSession orchestration, persistence,
serialization, memory files, and context compaction helpers.
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
