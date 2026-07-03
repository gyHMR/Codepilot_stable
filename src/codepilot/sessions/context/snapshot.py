from __future__ import annotations

# 新手导读：SessionSnapshotBuilder 收集仓库、工具结果、artifact、checkpoint 和上下文状态快照。
# 关注点：它是 projector 之前的事实整理阶段。

"""SessionSnapshotBuilder：从完整 session 状态读取本轮投影事实源。"""

from dataclasses import dataclass
from pathlib import Path

from codepilot.core import AgentContext
from codepilot.protocols import (
    ContextArtifactRef,
    ContextCheckpoint,
    RepositoryDelta,
    RepositorySnapshot,
    ToolResultMessage,
)

from .checkpoint import ContextCheckpointManager
from .ledger import ToolArtifactLedger
from .repository_tracker import RepositoryTracker
from .state import SessionContextState


@dataclass(frozen=True)
class SessionSnapshot:
    """一次上下文准备可读取的完整事实快照。"""

    repository_snapshot: RepositorySnapshot
    repository_delta: RepositoryDelta
    stale_items: list[str]
    artifact_refs: list[ContextArtifactRef]
    latest_checkpoint: ContextCheckpoint | None
    active_files: list[str]
    changed_files: list[str]


class SessionSnapshotBuilder:
    """读取 repository、tool ledger、checkpoint 和 session context state。"""

    def __init__(
        self,
        *,
        workspace_dir: str | Path,
        state: SessionContextState,
        repository: RepositoryTracker,
        ledger: ToolArtifactLedger,
        checkpoints: ContextCheckpointManager,
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.state = state
        self.repository = repository
        self.ledger = ledger
        self.checkpoints = checkpoints

    def build(self, context: AgentContext) -> SessionSnapshot:
        snapshot, delta = self.repository.refresh(self.state.last_repository_snapshot)
        self.state.last_repository_snapshot = snapshot
        if delta.changed:
            self.state.invalidate_paths([*delta.modified_paths, *delta.deleted_paths])
            self.state.invalidate_verification()

        for message in context.messages:
            if not isinstance(message, ToolResultMessage):
                continue
            self.state.observe_tool_result(
                message,
                repository_fingerprint=snapshot.fingerprint,
            )
            self.ledger.record_tool_result(
                run_id=None,
                message=message,
            )

        stale_items = self.state.validate_sources(snapshot.fingerprint)
        return SessionSnapshot(
            repository_snapshot=snapshot,
            repository_delta=delta,
            stale_items=stale_items,
            artifact_refs=self.ledger.artifact_refs(),
            latest_checkpoint=self.checkpoints.load_latest(),
            active_files=sorted(self.state.active_files),
            changed_files=sorted({*delta.modified_paths, *delta.added_paths}),
        )


__all__ = ["SessionSnapshot", "SessionSnapshotBuilder"]
