from __future__ import annotations

"""会话检查点辅助工具。

检查点目前刻意采用基于事件的方式实现；后续阶段可以附加文件快照和 diff 记录，
而无需修改 AgentSession 的接口。
"""

from dataclasses import dataclass, field
from typing import Any

from ..persistence.store import SessionStore


@dataclass(frozen=True)
class SessionCheckpoint:
    """会话检查点：标记一个有意义的时间点。"""
    session_id: str
    label: str
    details: dict[str, Any] = field(default_factory=dict)


def record_checkpoint(
    store: SessionStore,
    *,
    session_id: str,
    label: str,
    details: dict[str, Any] | None = None,
) -> SessionCheckpoint:
    """记录一个检查点事件到会话存储。"""
    checkpoint = SessionCheckpoint(
        session_id=session_id,
        label=label,
        details=dict(details or {}),
    )
    store.append_event(
        {
            "type": "session_checkpoint",
            "session_id": checkpoint.session_id,
            "label": checkpoint.label,
            "details": checkpoint.details,
        }
    )
    return checkpoint
