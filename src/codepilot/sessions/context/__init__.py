from __future__ import annotations

"""Dynamic context governance for Agent sessions."""

from .checkpoint import ContextCheckpointManager
from .governor import ContextGovernor
from .ledger import ToolArtifactLedger, ToolLedgerEntry
from .policy import ContextPressurePolicy
from .projector import ContextProjection, ContextProjector
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
from .snapshot import SessionSnapshot, SessionSnapshotBuilder
from .state import (
    ActiveFile,
    ContextEvidence,
    ContextEvidenceKind,
    ContextFileRole,
    FileSummary,
    SessionContextState,
)


__all__ = [
    "ActiveFile",
    "ContextEvidence",
    "ContextEvidenceKind",
    "ContextFileRole",
    "FileSummary",
    "SessionContextState",
    "ContextGovernor",
    "ContextPressurePolicy",
    "ContextProjection",
    "ContextProjector",
    "SessionSnapshot",
    "SessionSnapshotBuilder",
    "ContextCheckpointManager",
    "ToolArtifactLedger",
    "ToolLedgerEntry",
    "GitInfo",
    "RepositoryBootstrap",
    "build_repository_bootstrap",
    "render_repository_context",
    "RepositoryTracker",
    "compare_snapshots",
    "render_repository_snapshot",
]
