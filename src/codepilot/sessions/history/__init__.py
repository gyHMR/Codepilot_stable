from __future__ import annotations

"""会话分支、切换和检查点。"""

from .branching import (
    build_session_options_from_existing,
    create_fresh_session,
    fork_session,
    switch_session,
    switch_to_entry,
)
from .checkpoint import SessionCheckpoint, record_checkpoint
from .task_recovery import TaskRecoveryStore
from .git_rollback import (
    GitRollbackBaseline,
    GitRollbackResult,
    build_rollback_metadata,
    capture_git_baseline,
    revert_run_changes,
)


__all__ = [
    "SessionCheckpoint",
    "TaskRecoveryStore",
    "GitRollbackBaseline",
    "GitRollbackResult",
    "build_session_options_from_existing",
    "build_rollback_metadata",
    "capture_git_baseline",
    "create_fresh_session",
    "fork_session",
    "record_checkpoint",
    "revert_run_changes",
    "switch_session",
    "switch_to_entry",
]
