from __future__ import annotations

"""Dynamic session context governance."""

from .compaction import (
    COMPACTION_SYSTEM_PROMPT,
    ContextCompactionResult,
    build_compacted_context,
    fallback_summary,
    format_messages_for_summary,
)
from .compiler import ContextCompiler, ContextPolicy
from .repository_context import (
    GitInfo,
    RepositoryBootstrap,
    build_repository_bootstrap,
    render_repository_context,
)
from .repository_tracker import (
    RepositoryTracker,
    compare_snapshots,
    render_repository_snapshot,
)
from .state import ActiveFile, ContextEvidence, FileSummary, SessionContextState


__all__ = [
    "ActiveFile",
    "COMPACTION_SYSTEM_PROMPT",
    "ContextCompiler",
    "ContextCompactionResult",
    "ContextEvidence",
    "ContextPolicy",
    "FileSummary",
    "GitInfo",
    "RepositoryBootstrap",
    "RepositoryTracker",
    "SessionContextState",
    "build_compacted_context",
    "build_repository_bootstrap",
    "compare_snapshots",
    "fallback_summary",
    "format_messages_for_summary",
    "render_repository_context",
    "render_repository_snapshot",
]
