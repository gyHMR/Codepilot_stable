from __future__ import annotations

"""Stable context-governance data contracts shared across layers."""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ContextFreshness = Literal["fresh", "stale", "missing", "unknown"]
ContextTrust = Literal["observed", "derived", "user_given", "model_claim"]
DroppedContextReason = Literal[
    "duplicate",
    "stale",
    "low_relevance",
    "over_budget",
    "superseded",
    "missing_source",
]


@dataclass(frozen=True)
class RepositorySnapshot:
    workspace_root: str
    project_type: str | None
    manifest_files: list[str]
    top_level_entries: list[str]
    test_directories: list[str]
    instruction_files: list[str]
    branch: str | None
    head_sha: str | None
    git_status: list[str]
    fingerprint: str
    instruction_hashes: dict[str, str] = field(default_factory=dict)
    dirty_path_hashes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RepositoryDelta:
    added_paths: list[str] = field(default_factory=list)
    modified_paths: list[str] = field(default_factory=list)
    deleted_paths: list[str] = field(default_factory=list)
    branch_changed: bool = False
    head_changed: bool = False
    instructions_changed: bool = False

    @property
    def changed(self) -> bool:
        return bool(
            self.added_paths
            or self.modified_paths
            or self.deleted_paths
            or self.branch_changed
            or self.head_changed
            or self.instructions_changed
        )


@dataclass
class ContextItem:
    id: str
    kind: str
    content: str
    source: str
    trust: ContextTrust
    priority: int
    estimated_tokens: int
    path: str | None = None
    source_hash: str | None = None
    freshness: ContextFreshness = "unknown"


@dataclass
class ContextSection:
    name: str
    budget_tokens: int
    min_tokens: int
    items: list[ContextItem] = field(default_factory=list)
    reduction_policy: str = "drop_low_priority"


@dataclass(frozen=True)
class ContextSectionReport:
    name: str
    budget_tokens: int
    candidate_items: int
    selected_items: int
    estimated_tokens_before: int
    estimated_tokens_after: int
    reduction_policy: str


@dataclass(frozen=True)
class DroppedContextItem:
    item_id: str
    section: str
    reason: DroppedContextReason
    source: str


@dataclass
class ContextReport:
    context_id: str
    repository_fingerprint: str
    total_budget_tokens: int
    estimated_tokens_before: int
    estimated_tokens_after: int
    sections: list[ContextSectionReport] = field(default_factory=list)
    stale_items: list[str] = field(default_factory=list)
    dropped_items: list[DroppedContextItem] = field(default_factory=list)
    repository_delta: RepositoryDelta = field(default_factory=RepositoryDelta)
    retrieved_memory_ids: list[str] = field(default_factory=list)
    memory_retrieval_reasons: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "ContextFreshness",
    "ContextItem",
    "ContextReport",
    "ContextSection",
    "ContextSectionReport",
    "ContextTrust",
    "DroppedContextItem",
    "DroppedContextReason",
    "RepositoryDelta",
    "RepositorySnapshot",
]
