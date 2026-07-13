from __future__ import annotations

"""Stable Runtime execution contracts shared by orchestration components."""

from dataclasses import dataclass
from typing import Literal


RuntimeExecutionState = Literal[
    "new",
    "preparing",
    "executing",
    "waiting",
    "resuming",
    "cancelling",
    "finalizing",
    "terminal",
    "released",
]
TerminalOutcome = Literal["completed", "failed", "cancelled"]
CommitKind = Literal["progress", "waiting", "terminal"]


@dataclass(frozen=True)
class RunCommitIdentity:
    commit_id: str
    expected_revision: int
    kind: CommitKind

    def __post_init__(self) -> None:
        object.__setattr__(self, "commit_id", _required_text(self.commit_id, "commit_id"))
        if isinstance(self.expected_revision, bool) or not isinstance(self.expected_revision, int):
            raise TypeError("expected_revision must be an int")
        if self.expected_revision < 0:
            raise ValueError("expected_revision cannot be negative")
        if self.kind not in {"progress", "waiting", "terminal"}:
            raise ValueError(f"Unknown commit kind: {self.kind}")


@dataclass(frozen=True)
class CommitReceipt:
    commit_id: str
    revision: int
    kind: CommitKind

    def __post_init__(self) -> None:
        object.__setattr__(self, "commit_id", _required_text(self.commit_id, "commit_id"))
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("revision must be an int")
        if self.revision < 1:
            raise ValueError("revision must be positive")
        if self.kind not in {"progress", "waiting", "terminal"}:
            raise ValueError(f"Unknown commit kind: {self.kind}")

    def matches(self, identity: RunCommitIdentity) -> bool:
        return self.commit_id == identity.commit_id and self.kind == identity.kind


def terminal_outcome_for_status(status: str) -> TerminalOutcome | None:
    normalized = _required_text(status, "status")
    if normalized == "completed":
        return "completed"
    if normalized == "failed":
        return "failed"
    if normalized == "cancelled":
        return "cancelled"
    if normalized in {"waiting", "waiting_approval", "waiting_user"}:
        return None
    raise ValueError(f"Unknown execution outcome status: {status}")


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


__all__ = [
    "CommitKind",
    "CommitReceipt",
    "RunCommitIdentity",
    "RuntimeExecutionState",
    "TerminalOutcome",
    "terminal_outcome_for_status",
]
