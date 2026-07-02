from __future__ import annotations

"""Session orchestration package.

The sessions layer owns four small domains:

- persistence: session/run facts and filesystem layout
- context: per-turn ContextGovernor prompt projection
- memory: durable project/session memory
- history: task recovery, branching, and lightweight git rollback
"""

from .context import ContextGovernor, RepositoryBootstrap, RepositoryTracker
from .history import SessionCheckpoint
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
from .types import AgentSessionOptions, ConvertToLlmFn

__all__ = [
    "AgentSession",
    "AgentSessionOptions",
    "ConvertToLlmFn",
    "ContextGovernor",
    "SessionCheckpoint",
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
