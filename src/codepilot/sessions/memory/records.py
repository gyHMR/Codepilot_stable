from __future__ import annotations

"""结构化记忆记录和共享值类型。

本模块只描述“什么是记忆”，不负责决定何时写入记忆。
写入策略在 ``MemoryWriter``，检索策略在 ``MemoryRetriever``；
但“哪些记录属于可复用长期记忆”是数据模型自身的语义，集中放在这里，
避免调用方用零散的字符串判断重复解释。
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, cast


# 记忆类型：
# - task/file/failure 是早期运行态记忆，仅为兼容旧存储保留，不再进入检索。
# - experience/decision/project 是可复用的 durable memory。
MemoryKind = Literal["task", "file", "failure", "experience", "decision", "project"]
# 记忆作用域：会话级/项目级
MemoryScope = Literal["session", "project"]
# 记忆信任度：观察到/已验证/用户给出/模型声称
MemoryTrust = Literal["observed", "verified", "user_given", "model_claim"]
# 记忆状态：活跃/过时/已被替代/已删除
MemoryStatus = Literal["active", "stale", "superseded", "deleted"]

MEMORY_SCHEMA_VERSION = 1
DURABLE_MEMORY_KINDS = frozenset({"project", "decision", "experience"})
LEGACY_MEMORY_KINDS = frozenset({"task", "file", "failure"})
_MEMORY_KINDS = frozenset(
    {"task", "file", "failure", "experience", "decision", "project"}
)
_MEMORY_SCOPES = frozenset({"session", "project"})
_MEMORY_TRUST_VALUES = frozenset(
    {"observed", "verified", "user_given", "model_claim"}
)
_MEMORY_STATUS_VALUES = frozenset({"active", "stale", "superseded", "deleted"})


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryRecord:
    """一条结构化记忆记录。

    设计边界：
    - ``kind`` 表示这条记录的语义类型，而不是存储位置。
    - ``scope`` 表示生命周期边界：当前会话可见，还是整个项目可见。
    - ``trust`` 表示证据强度，检索时只影响排序，不替代新鲜度检查。
    - ``status`` 表示当前是否仍可使用；非 active 记录不会进入上下文。
    """
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

    def __post_init__(self) -> None:
        _ensure_memory_kind(self.kind)
        _ensure_memory_scope(self.scope)
        _ensure_memory_trust(self.trust)
        _ensure_memory_status(self.status)

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

    @property
    def is_active(self) -> bool:
        """记录是否仍可用于后续推理。"""

        return self.status == "active"

    @property
    def is_durable(self) -> bool:
        """记录是否属于可复用长期记忆。"""

        return self.kind in DURABLE_MEMORY_KINDS

    @property
    def is_legacy_state(self) -> bool:
        """记录是否只是旧版运行态记忆，不应再被检索注入上下文。"""

        return self.kind in LEGACY_MEMORY_KINDS

    @property
    def is_retrievable(self) -> bool:
        """记录是否有资格参与记忆检索。"""

        return self.retrieval_exclusion_reason() is None

    def retrieval_exclusion_reason(self) -> str | None:
        """返回记录不能参与检索的原因；可检索时返回 ``None``。

        这个方法只表达数据模型层面的硬约束。相关性评分、数量限制和
        查询模式仍由 ``MemoryRetriever`` 负责。
        """

        if not self.is_active:
            return f"status:{self.status}"
        if self.is_legacy_state:
            return f"transient_kind:{self.kind}"
        if not self.is_durable:
            return f"unsupported_kind:{self.kind}"
        if (
            self.kind == "experience"
            and self.content.get("maturity") == "candidate"
        ):
            return "candidate_experience"
        return None


@dataclass(frozen=True)
class MemoryQuery:
    """记忆检索查询。"""
    text: str                     # 查询文本
    active_paths: list[str]       # 当前活跃文件路径
    limit: int = 8                # 返回数量上限
    task_phase: str | None = None
    action_intent: str | None = None
    recent_error: str | None = None
    retrieval_mode: str | None = None


@dataclass(frozen=True)
class RetrievedMemory:
    """检索到的记忆（含评分和匹配原因）。"""
    record: MemoryRecord
    score: int
    reasons: list[str]


def _memory_kind(value: object) -> MemoryKind:
    return _ensure_memory_kind(value) if value in _MEMORY_KINDS else "project"


def _memory_scope(value: object) -> MemoryScope:
    return _ensure_memory_scope(value) if value in _MEMORY_SCOPES else "session"


def _memory_trust(value: object) -> MemoryTrust:
    return _ensure_memory_trust(value) if value in _MEMORY_TRUST_VALUES else "observed"


def _memory_status(value: object) -> MemoryStatus:
    return _ensure_memory_status(value) if value in _MEMORY_STATUS_VALUES else "active"


def _ensure_memory_kind(value: object) -> MemoryKind:
    if value not in _MEMORY_KINDS:
        raise ValueError(f"Unknown memory kind: {value}")
    return cast(MemoryKind, value)


def _ensure_memory_scope(value: object) -> MemoryScope:
    if value not in _MEMORY_SCOPES:
        raise ValueError(f"Unknown memory scope: {value}")
    return cast(MemoryScope, value)


def _ensure_memory_trust(value: object) -> MemoryTrust:
    if value not in _MEMORY_TRUST_VALUES:
        raise ValueError(f"Unknown memory trust: {value}")
    return cast(MemoryTrust, value)


def _ensure_memory_status(value: object) -> MemoryStatus:
    if value not in _MEMORY_STATUS_VALUES:
        raise ValueError(f"Unknown memory status: {value}")
    return cast(MemoryStatus, value)


__all__ = [
    "MEMORY_SCHEMA_VERSION",
    "DURABLE_MEMORY_KINDS",
    "LEGACY_MEMORY_KINDS",
    "MemoryKind",
    "MemoryQuery",
    "MemoryRecord",
    "MemoryScope",
    "MemoryStatus",
    "MemoryTrust",
    "RetrievedMemory",
    "utc_now_iso",
]
