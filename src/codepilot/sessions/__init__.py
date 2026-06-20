"""
Session management for Codepilot.

This package owns long-lived AgentSession orchestration, persistence,
serialization, memory files, and context compaction helpers.
"""

from .context_compiler import ContextCompiler, ContextPolicy
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
from .repository_context import RepositoryBootstrap
from .repository_tracker import RepositoryTracker
from .session import AgentSession
from .store import SessionStore, new_session_id
from .run_store import FreshnessResult, RunStore
from .checkpoint import SessionCheckpoint
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
