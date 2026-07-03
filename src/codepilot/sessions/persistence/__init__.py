from __future__ import annotations

# 新手导读：包门面文件：集中导出本层最常用的类型和入口，降低学习时的导入成本。
# 关注点：sessions 层是会话事实源，负责消息、run、记忆、上下文投影和任务恢复。

"""会话持久化层：消息存储、事件记录、Run 产物和序列化。"""

from .run_store import FreshnessResult, FreshnessStatus, RunStore
from .serde import message_from_dict, message_to_dict
from .store import SessionStore, new_session_id


__all__ = [
    "FreshnessResult",
    "FreshnessStatus",
    "RunStore",
    "SessionStore",
    "message_from_dict",
    "message_to_dict",
    "new_session_id",
]
