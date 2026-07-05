from __future__ import annotations

# 新手导读：包门面文件：集中导出本层最常用的类型和入口，降低学习时的导入成本。
# 关注点：sessions 层是会话事实源，负责消息、run、记忆、上下文投影和任务恢复。

"""Session orchestration package.

The sessions layer owns four small domains:

- persistence: session/run facts and filesystem layout
- context: per-turn ContextGovernor prompt projection
- memory: durable project/session memory
- history: task recovery, branching, and lightweight git rollback
"""

from .controller import SessionController
from .contracts import (
    CancelRunIntent,
    PreparedAgentRun,
    RollbackBaselineRef,
    SessionCommandIntent,
    SessionCommandRecord,
    SessionIntent,
    SessionResumeIntent,
    SessionRunIntent,
    SessionRunRecord,
    SessionOptions,
    SessionView,
)
from .storage import (
    RepositoryBootstrap,
    SessionOpenMetadata,
    build_repository_bootstrap,
    load_session_open_metadata,
    new_session_id,
)

__all__ = [
    "SessionController",
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
    "SessionOpenMetadata",
    "load_session_open_metadata",
    "RepositoryBootstrap",
    "build_repository_bootstrap",
    "SessionOptions",
    "new_session_id",
]
