"""
Session management for Codepilot.

This package owns long-lived AgentSession orchestration, persistence,
serialization, memory files, and context compaction helpers.
"""

from .memory import load_channel_memory, load_global_memory, load_merged_memory, save_channel_memory, save_global_memory
from .session import AgentSession
from .store import SessionStore, new_session_id
from .run_store import FreshnessResult, RunStore
from .checkpoint import SessionCheckpoint
from .types import AgentSessionOptions, ConvertToLlmFn

__all__ = [
    "AgentSession",
    "AgentSessionOptions",
    "ConvertToLlmFn",
    "SessionCheckpoint",
    "SessionStore",
    "FreshnessResult",
    "RunStore",
    "new_session_id",
    "load_channel_memory",
    "load_global_memory",
    "load_merged_memory",
    "save_channel_memory",
    "save_global_memory",
]
