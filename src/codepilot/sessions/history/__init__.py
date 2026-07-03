from __future__ import annotations

# 新手导读：包门面文件：集中导出本层最常用的类型和入口，降低学习时的导入成本。
# 关注点：sessions 层是会话事实源，负责消息、run、记忆、上下文投影和任务恢复。

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
    GitRollbackAction,
    GitRollbackBaseline,
    GitRollbackPlan,
    GitRollbackResult,
    build_rollback_metadata,
    capture_git_baseline,
    plan_run_rollback,
    revert_run_changes,
)


__all__ = [
    "SessionCheckpoint",
    "TaskRecoveryStore",
    "GitRollbackAction",
    "GitRollbackBaseline",
    "GitRollbackPlan",
    "GitRollbackResult",
    "build_session_options_from_existing",
    "build_rollback_metadata",
    "capture_git_baseline",
    "create_fresh_session",
    "fork_session",
    "plan_run_rollback",
    "record_checkpoint",
    "revert_run_changes",
    "switch_session",
    "switch_to_entry",
]
