from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Literal, Protocol, TypeAlias, cast


MemoryType = Literal["profile", "feedback", "project", "experience", "reference"]
MemoryScope = Literal["user", "project"]
MemorySource = Literal[
    "user_explicit",
    "user_feedback",
    "agent_extracted",
    "verified_run",
]
MemoryStatus = Literal[
    "candidate",
    "active",
    "disabled",
    "superseded",
    "deleted",
]
MemoryProposalOrigin = Literal["user_explicit", "agent_finalization"]

_MEMORY_TYPES = frozenset({"profile", "feedback", "project", "experience", "reference"})
_MEMORY_SCOPES = frozenset({"user", "project"})
_MEMORY_SOURCES = frozenset(
    {"user_explicit", "user_feedback", "agent_extracted", "verified_run"}
)
_MEMORY_STATUSES = frozenset(
    {"candidate", "active", "disabled", "superseded", "deleted"}
)
_RECORD_FIELDS = frozenset(
    {"id", "scope", "type", "key", "content", "source", "status", "updated_at"}
)
_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+$")


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    scope: MemoryScope
    type: MemoryType
    key: str
    content: str
    source: MemorySource
    status: MemoryStatus
    updated_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_text(self.id, "memory id"))
        object.__setattr__(self, "scope", _memory_scope(self.scope))
        object.__setattr__(self, "type", _memory_type(self.type))
        key = _required_text(self.key, "memory key").lower()
        if not _KEY_PATTERN.fullmatch(key):
            raise ValueError("memory key must be lower-case dot notation")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "content", _required_text(self.content, "memory content"))
        object.__setattr__(self, "source", _memory_source(self.source))
        object.__setattr__(self, "status", _memory_status(self.status))
        if not isinstance(self.updated_at, datetime):
            raise TypeError("updated_at must be datetime")
        if self.updated_at.tzinfo is None or self.updated_at.utcoffset() is None:
            raise ValueError("updated_at must be timezone-aware")
        object.__setattr__(self, "updated_at", self.updated_at.astimezone(timezone.utc))

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "scope": self.scope,
            "type": self.type,
            "key": self.key,
            "content": self.content,
            "source": self.source,
            "status": self.status,
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "MemoryRecord":
        if not isinstance(raw, Mapping):
            raise TypeError("memory record must be a mapping")
        keys = set(raw)
        missing = sorted(_RECORD_FIELDS - keys)
        unknown = sorted(keys - _RECORD_FIELDS)
        if missing:
            raise ValueError("missing memory fields: " + ", ".join(missing))
        if unknown:
            raise ValueError("unknown memory fields: " + ", ".join(unknown))
        return cls(
            id=_required_text(raw.get("id"), "memory id"),
            scope=_memory_scope(raw.get("scope")),
            type=_memory_type(raw.get("type")),
            key=_required_text(raw.get("key"), "memory key"),
            content=_required_text(raw.get("content"), "memory content"),
            source=_memory_source(raw.get("source")),
            status=_memory_status(raw.get("status")),
            updated_at=_datetime(raw.get("updated_at")),
        )


@dataclass(frozen=True)
class MemoryProposal:
    scope: MemoryScope
    type: MemoryType
    key: str
    content: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", _memory_scope(self.scope))
        object.__setattr__(self, "type", _memory_type(self.type))
        key = _required_text(self.key, "memory proposal key").lower()
        if not _KEY_PATTERN.fullmatch(key):
            raise ValueError("memory proposal key must be lower-case dot notation")
        object.__setattr__(self, "key", key)
        object.__setattr__(
            self,
            "content",
            _required_text(self.content, "memory proposal content"),
        )


@dataclass(frozen=True)
class MemoryProposalBatch:
    session_id: str
    run_id: str
    origin: MemoryProposalOrigin
    verification_passed: bool
    proposals: tuple[MemoryProposal, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _required_text(self.session_id, "session_id"))
        object.__setattr__(self, "run_id", _required_text(self.run_id, "run_id"))
        if self.origin not in {"user_explicit", "agent_finalization"}:
            raise ValueError(f"unknown memory proposal origin: {self.origin}")
        if not isinstance(self.verification_passed, bool):
            raise TypeError("verification_passed must be bool")
        proposals = tuple(self.proposals)
        if len(proposals) > 3:
            raise ValueError("a run can propose at most three memories")
        if any(not isinstance(item, MemoryProposal) for item in proposals):
            raise TypeError("proposals must contain MemoryProposal values")
        object.__setattr__(self, "proposals", proposals)


@dataclass(frozen=True)
class MemoryProposalReceipt:
    records: tuple[MemoryRecord, ...] = ()
    rejected: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if any(not isinstance(record, MemoryRecord) for record in records):
            raise TypeError("proposal receipt records must contain MemoryRecord values")
        object.__setattr__(self, "records", records)
        object.__setattr__(
            self,
            "rejected",
            tuple(_required_text(reason, "rejection reason") for reason in self.rejected),
        )


@dataclass(frozen=True)
class MemoryQuery:
    user_request: str
    task_goal: str
    current_step: str | None = None
    active_paths: tuple[str, ...] = ()
    limit: int = 5

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_request", str(self.user_request or "").strip())
        object.__setattr__(self, "task_goal", str(self.task_goal or "").strip())
        object.__setattr__(self, "current_step", _optional_text(self.current_step))
        object.__setattr__(
            self,
            "active_paths",
            tuple(_required_text(path, "active path") for path in self.active_paths),
        )
        if not isinstance(self.limit, int) or isinstance(self.limit, bool):
            raise TypeError("memory recall limit must be int")
        if self.limit < 1 or self.limit > 5:
            raise ValueError("memory recall limit must be between 1 and 5")


@dataclass(frozen=True)
class RecalledMemory:
    memory_id: str
    scope: MemoryScope
    type: MemoryType
    key: str
    content: str
    source: MemorySource
    rank_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory_id", _required_text(self.memory_id, "memory id"))
        object.__setattr__(self, "scope", _memory_scope(self.scope))
        object.__setattr__(self, "type", _memory_type(self.type))
        object.__setattr__(self, "key", _required_text(self.key, "memory key").lower())
        object.__setattr__(self, "content", _required_text(self.content, "memory content"))
        object.__setattr__(self, "source", _memory_source(self.source))
        object.__setattr__(
            self,
            "rank_reasons",
            tuple(_required_text(reason, "rank reason") for reason in self.rank_reasons),
        )


@dataclass(frozen=True)
class MemoryRecallResult:
    retrieved: tuple[RecalledMemory, ...] = ()
    dropped: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        retrieved = tuple(self.retrieved)
        if any(not isinstance(item, RecalledMemory) for item in retrieved):
            raise TypeError("retrieved must contain RecalledMemory values")
        object.__setattr__(self, "retrieved", retrieved)
        if not isinstance(self.dropped, Mapping):
            raise TypeError("dropped must be a mapping")
        object.__setattr__(
            self,
            "dropped",
            MappingProxyType(
                {
                    _required_text(memory_id, "dropped memory id"): _required_text(
                        reason, "drop reason"
                    )
                    for memory_id, reason in self.dropped.items()
                }
            ),
        )


@dataclass(frozen=True)
class MemoryActor:
    user_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_id", _required_text(self.user_id, "user_id"))


@dataclass(frozen=True)
class AddMemory:
    scope: MemoryScope
    type: MemoryType
    key: str
    content: str

    def __post_init__(self) -> None:
        proposal = MemoryProposal(self.scope, self.type, self.key, self.content)
        object.__setattr__(self, "scope", proposal.scope)
        object.__setattr__(self, "type", proposal.type)
        object.__setattr__(self, "key", proposal.key)
        object.__setattr__(self, "content", proposal.content)


@dataclass(frozen=True)
class ListMemory:
    scope: MemoryScope | None = None
    status: MemoryStatus | None = None

    def __post_init__(self) -> None:
        if self.scope is not None:
            object.__setattr__(self, "scope", _memory_scope(self.scope))
        if self.status is not None:
            object.__setattr__(self, "status", _memory_status(self.status))


@dataclass(frozen=True)
class _MemoryIdCommand:
    memory_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory_id", _required_text(self.memory_id, "memory id"))


@dataclass(frozen=True)
class ShowMemory(_MemoryIdCommand):
    pass


@dataclass(frozen=True)
class ApproveMemory(_MemoryIdCommand):
    pass


@dataclass(frozen=True)
class RejectMemory(_MemoryIdCommand):
    pass


@dataclass(frozen=True)
class EditMemory(_MemoryIdCommand):
    content: str

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "content", _required_text(self.content, "memory content"))


@dataclass(frozen=True)
class DisableMemory(_MemoryIdCommand):
    pass


@dataclass(frozen=True)
class EnableMemory(_MemoryIdCommand):
    pass


@dataclass(frozen=True)
class DeleteMemory(_MemoryIdCommand):
    pass


@dataclass(frozen=True)
class HistoryMemory:
    scope: MemoryScope
    key: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", _memory_scope(self.scope))
        key = _required_text(self.key, "memory key").lower()
        if not _KEY_PATTERN.fullmatch(key):
            raise ValueError("memory key must be lower-case dot notation")
        object.__setattr__(self, "key", key)


@dataclass(frozen=True)
class PurgeMemory(_MemoryIdCommand):
    pass


MemoryCommand: TypeAlias = (
    AddMemory
    | ListMemory
    | ShowMemory
    | ApproveMemory
    | RejectMemory
    | EditMemory
    | DisableMemory
    | EnableMemory
    | DeleteMemory
    | HistoryMemory
    | PurgeMemory
)


@dataclass(frozen=True)
class MemoryCommandResult:
    records: tuple[MemoryRecord, ...] = ()
    message: str = ""

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if any(not isinstance(record, MemoryRecord) for record in records):
            raise TypeError("command result records must contain MemoryRecord values")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "message", str(self.message or "").strip())


class MemoryRecallPort(Protocol):
    def recall(self, query: MemoryQuery) -> MemoryRecallResult: ...


class MemoryProposalPort(Protocol):
    def submit_proposals(
        self,
        batch: MemoryProposalBatch,
    ) -> MemoryProposalReceipt: ...


class MemoryManagementPort(Protocol):
    def execute(
        self,
        command: MemoryCommand,
        actor: MemoryActor,
    ) -> MemoryCommandResult: ...


def _required_text(value: object, field_name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise TypeError("updated_at must be datetime or ISO datetime string")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("updated_at must be an ISO datetime string") from exc


def _memory_type(value: object) -> MemoryType:
    if value not in _MEMORY_TYPES:
        raise ValueError(f"unknown memory type: {value}")
    return cast(MemoryType, value)


def _memory_scope(value: object) -> MemoryScope:
    if value not in _MEMORY_SCOPES:
        raise ValueError(f"unknown memory scope: {value}")
    return cast(MemoryScope, value)


def _memory_source(value: object) -> MemorySource:
    if value not in _MEMORY_SOURCES:
        raise ValueError(f"unknown memory source: {value}")
    return cast(MemorySource, value)


def _memory_status(value: object) -> MemoryStatus:
    if value not in _MEMORY_STATUSES:
        raise ValueError(f"unknown memory status: {value}")
    return cast(MemoryStatus, value)


__all__ = [
    "AddMemory",
    "ApproveMemory",
    "DeleteMemory",
    "DisableMemory",
    "EditMemory",
    "EnableMemory",
    "HistoryMemory",
    "ListMemory",
    "MemoryActor",
    "MemoryCommand",
    "MemoryCommandResult",
    "MemoryManagementPort",
    "MemoryProposal",
    "MemoryProposalBatch",
    "MemoryProposalOrigin",
    "MemoryProposalPort",
    "MemoryProposalReceipt",
    "MemoryQuery",
    "MemoryRecallPort",
    "MemoryRecallResult",
    "MemoryRecord",
    "MemoryScope",
    "MemorySource",
    "MemoryStatus",
    "MemoryType",
    "PurgeMemory",
    "RecalledMemory",
    "RejectMemory",
    "ShowMemory",
]
