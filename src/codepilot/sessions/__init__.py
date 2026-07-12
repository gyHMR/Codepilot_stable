from __future__ import annotations

# 新手导读：包门面文件：集中导出本层最常用的类型和入口，降低学习时的导入成本。
# 关注点：sessions 层保存 Session、Run、消息和 Checkpoint 事实，不决定任务如何推进。

"""Session state and recovery contracts.

The sessions layer owns four facts:

- Session state
- Run state and checkpoints
- Canonical messages
- Non-authoritative audit events
"""

from .contracts import (
    CHECKPOINT_SCHEMA_VERSION,
    MESSAGE_RECORD_SCHEMA_VERSION,
    RUN_STATE_SCHEMA_VERSION,
    SESSION_STATE_SCHEMA_VERSION,
    CancelRunIntent,
    ComponentCheckpoint,
    MessageCursor,
    MessageRecord,
    ModelRef,
    PreparedAgentRun,
    RecoveryBundle,
    RecoveryIssue,
    RecoveryRequest,
    RecoveryResult,
    RollbackBaselineRef,
    RunCheckpoint,
    RunState,
    SessionCommandIntent,
    SessionCommandRecord,
    SessionIntent,
    SessionResumeIntent,
    SessionRunIntent,
    SessionRunRecord,
    SessionOptions,
    SessionState,
    SessionStateConflictError,
    SessionStateValidationError,
    SessionView,
    WaitingState,
    WorkspaceCheckpoint,
    WorkspaceEffectsSnapshot,
    WorkspaceRecoveryState,
    validate_run_state,
    validate_run_transition,
)
from .repository import FileSessionRepository
from .service import (
    BeginRunRequest,
    BeginRunResult,
    CommitBoundaryRequest,
    CommitBoundaryResult,
    CreateSessionRequest,
    FinishRunRequest,
    FinishRunResult,
    ResumeRunRequest,
    SessionStateService,
    new_session_id,
)
__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "MESSAGE_RECORD_SCHEMA_VERSION",
    "RUN_STATE_SCHEMA_VERSION",
    "SESSION_STATE_SCHEMA_VERSION",
    "SessionState",
    "RunState",
    "RunCheckpoint",
    "MessageRecord",
    "MessageCursor",
    "ModelRef",
    "WaitingState",
    "ComponentCheckpoint",
    "WorkspaceCheckpoint",
    "WorkspaceEffectsSnapshot",
    "WorkspaceRecoveryState",
    "RecoveryRequest",
    "RecoveryResult",
    "RecoveryBundle",
    "RecoveryIssue",
    "SessionStateValidationError",
    "SessionStateConflictError",
    "validate_run_state",
    "validate_run_transition",
    "FileSessionRepository",
    "SessionStateService",
    "CreateSessionRequest",
    "BeginRunRequest",
    "BeginRunResult",
    "CommitBoundaryRequest",
    "CommitBoundaryResult",
    "ResumeRunRequest",
    "FinishRunRequest",
    "FinishRunResult",
    "SessionRunIntent",
    "SessionResumeIntent",
    "SessionCommandIntent",
    "CancelRunIntent",
    "SessionIntent",
    "PreparedAgentRun",
    "RollbackBaselineRef",
    "SessionRunRecord",
    "SessionCommandRecord",
    "SessionView",
    "SessionOptions",
    "new_session_id",
]
