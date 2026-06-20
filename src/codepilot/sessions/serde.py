from __future__ import annotations

"""
兼容导出：消息序列化已迁移到 sessions.persistence.serde。

TODO(sessions-cleanup): 后续清理旧导入时，可以删除本模块。
"""

from .persistence.serde import message_from_dict, message_to_dict


__all__ = ["message_from_dict", "message_to_dict"]
