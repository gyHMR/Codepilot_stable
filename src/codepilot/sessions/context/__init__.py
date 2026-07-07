from __future__ import annotations

# 新手导读：包门面文件：集中导出本层最常用的类型和入口，降低学习时的导入成本。
# 关注点：sessions 层是会话事实源，负责消息、run、记忆、上下文投影和任务恢复。

"""Context preparation for Agent sessions."""

from .governor import ContextGovernor
from .ledger import ToolArtifactLedger, ToolLedgerEntry
from .policy import ContextPressurePolicy
from .repository_tracker import (
    RepositoryTracker,
    compare_snapshots,
    render_repository_snapshot,
)
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
    "ToolArtifactLedger",
    "ToolLedgerEntry",
    "RepositoryTracker",
    "compare_snapshots",
    "render_repository_snapshot",
]
