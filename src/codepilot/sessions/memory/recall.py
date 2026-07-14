from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .contracts import MemoryQuery, MemoryRecallResult, MemoryRecord, RecalledMemory


_BASE_TYPE_SCORE = {
    "profile": 40,
    "feedback": 35,
    "project": 0,
    "experience": 0,
    "reference": 0,
}
_ALWAYS_RECALL_TYPES = frozenset({"profile", "feedback"})


@dataclass(frozen=True)
class _RankedMemory:
    record: MemoryRecord
    score: int
    reasons: tuple[str, ...]


class _RecordRepository(Protocol):
    def all_records(self) -> list[MemoryRecord]: ...


class MemoryRecallEngine:
    """Deterministic Active-only recall across User and Project stores."""

    def __init__(
        self,
        user_repository: _RecordRepository,
        project_repository: _RecordRepository,
    ) -> None:
        self.user_repository = user_repository
        self.project_repository = project_repository

    def recall(self, query: MemoryQuery) -> MemoryRecallResult:
        dropped: dict[str, str] = {}
        active: list[MemoryRecord] = []
        for record in (
            *self.user_repository.all_records(),
            *self.project_repository.all_records(),
        ):
            if record.status != "active":
                dropped[record.id] = f"status:{record.status}"
                continue
            active.append(record)

        project_keys = {
            record.key for record in active if record.scope == "project"
        }
        visible: list[MemoryRecord] = []
        for record in active:
            if record.scope == "user" and record.key in project_keys:
                dropped[record.id] = "shadowed_by_project"
                continue
            visible.append(record)

        ranked: list[_RankedMemory] = []
        for record in visible:
            item = _rank(record, query)
            if item is None:
                dropped[record.id] = "low_relevance"
                continue
            ranked.append(item)
        ranked.sort(
            key=lambda item: (
                -item.score,
                0 if item.record.scope == "project" else 1,
                item.record.key,
                item.record.id,
            )
        )
        selected = ranked[: query.limit]
        for item in ranked[query.limit :]:
            dropped[item.record.id] = "over_limit"
        return MemoryRecallResult(
            retrieved=tuple(
                RecalledMemory(
                    memory_id=item.record.id,
                    scope=item.record.scope,
                    type=item.record.type,
                    key=item.record.key,
                    content=item.record.content,
                    source=item.record.source,
                    rank_reasons=item.reasons,
                )
                for item in selected
            ),
            dropped=dropped,
        )


def _rank(record: MemoryRecord, query: MemoryQuery) -> _RankedMemory | None:
    query_text = " ".join(
        part
        for part in (
            query.user_request,
            query.task_goal,
            query.current_step or "",
            " ".join(query.active_paths),
        )
        if part
    )
    query_terms = _terms(query_text)
    record_terms = _terms(f"{record.key} {record.content}")
    matches = sorted(query_terms & record_terms)
    path_matches = sorted(
        path
        for path in query.active_paths
        if path.lower().replace("\\", "/")
        and path.lower().replace("\\", "/")
        in f"{record.key} {record.content}".lower().replace("\\", "/")
    )
    if record.type not in _ALWAYS_RECALL_TYPES and not matches and not path_matches:
        return None

    score = _BASE_TYPE_SCORE[record.type]
    reasons = [f"type:{record.type}"]
    if record.scope == "project":
        score += 20
        reasons.append("scope:project")
    for term in matches[:4]:
        score += min(20, 4 + len(term) * 2)
        reasons.append(f"term:{term}")
    for path in path_matches[:2]:
        score += 20
        reasons.append(f"path:{path}")
    return _RankedMemory(record=record, score=score, reasons=tuple(reasons))


def _terms(text: str) -> set[str]:
    lowered = text.lower().replace("\\", "/")
    terms: set[str] = set()
    for token in re.findall(r"[a-z0-9_./-]+", lowered):
        if len(token) >= 2:
            terms.add(token)
        terms.update(part for part in re.split(r"[./_-]+", token) if len(part) >= 2)
    for sequence in re.findall(r"[\u4e00-\u9fff]+", lowered):
        if len(sequence) == 1:
            terms.add(sequence)
            continue
        terms.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return terms


__all__ = ["MemoryRecallEngine"]
