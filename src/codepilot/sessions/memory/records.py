from __future__ import annotations

"""结构化记忆记录和共享值类型。"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


# 记忆类型：任务/文件/失败教训/决策/项目知识
MemoryKind = Literal["task", "file", "failure", "decision", "project"]
# 记忆作用域：会话级/项目级
MemoryScope = Literal["session", "project"]
# 记忆信任度：观察到/已验证/用户给出/模型声称
MemoryTrust = Literal["observed", "verified", "user_given", "model_claim"]
# 记忆状态：活跃/过时/已被替代/已删除
MemoryStatus = Literal["active", "stale", "superseded", "deleted"]

MEMORY_SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryRecord:
    """结构化记忆记录。"""
    id: str                              # 记忆唯一标识
    kind: MemoryKind                     # 记忆类型
    scope: MemoryScope                   # 作用域
    content: dict[str, Any]              # 记忆内容
    source: str                          # 来源
    source_run_id: str | None = None     # 来源 Run ID
    related_paths: list[str] = field(default_factory=list)       # 关联文件路径
    source_hashes: dict[str, str] = field(default_factory=dict)  # 文件哈希快照
    trust: MemoryTrust = "observed"      # 信任度
    status: MemoryStatus = "active"      # 状态
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MemoryRecord":
        return cls(
            id=str(value.get("id", "")),
            kind=_memory_kind(value.get("kind")),
            scope=_memory_scope(value.get("scope")),
            content=dict(value.get("content", {})) if isinstance(value.get("content"), dict) else {},
            source=str(value.get("source", "unknown")),
            source_run_id=value.get("source_run_id") if isinstance(value.get("source_run_id"), str) else None,
            related_paths=[str(item) for item in value.get("related_paths", []) if isinstance(item, str)],
            source_hashes={
                str(key): str(item)
                for key, item in value.get("source_hashes", {}).items()
            } if isinstance(value.get("source_hashes"), dict) else {},
            trust=_memory_trust(value.get("trust")),
            status=_memory_status(value.get("status")),
            created_at=str(value.get("created_at", utc_now_iso())),
            updated_at=str(value.get("updated_at", utc_now_iso())),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryQuery:
    """记忆检索查询。"""
    text: str                     # 查询文本
    active_paths: list[str]       # 当前活跃文件路径
    limit: int = 8                # 返回数量上限


@dataclass(frozen=True)
class RetrievedMemory:
    """检索到的记忆（含评分和匹配原因）。"""
    record: MemoryRecord
    score: int
    reasons: list[str]


def _memory_kind(value: object) -> MemoryKind:
    return value if value in {"task", "file", "failure", "decision", "project"} else "project"  # type: ignore[return-value]


def _memory_scope(value: object) -> MemoryScope:
    return value if value in {"session", "project"} else "session"  # type: ignore[return-value]


def _memory_trust(value: object) -> MemoryTrust:
    return value if value in {"observed", "verified", "user_given", "model_claim"} else "observed"  # type: ignore[return-value]


def _memory_status(value: object) -> MemoryStatus:
    return value if value in {"active", "stale", "superseded", "deleted"} else "active"  # type: ignore[return-value]


__all__ = [
    "MEMORY_SCHEMA_VERSION",
    "MemoryKind",
    "MemoryQuery",
    "MemoryRecord",
    "MemoryScope",
    "MemoryStatus",
    "MemoryTrust",
    "RetrievedMemory",
    "utc_now_iso",
]
