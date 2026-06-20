from __future__ import annotations

"""Retrieve relevant structured memory for dynamic context compilation."""

import re
from pathlib import Path

from .files import load_global_memory
from .records import MemoryQuery, MemoryRecord, RetrievedMemory
from .rendering import render_memory
from .store import MemoryStore
from .writer import MemoryWriter


class MemoryRetriever:
    def __init__(self, *, store: MemoryStore, workspace_dir: str | Path) -> None:
        self.store = store
        self.workspace_dir = Path(workspace_dir)

    def retrieve(self, query: MemoryQuery) -> list[RetrievedMemory]:
        records = [*self.store.load_session(), *self.store.load_project()]
        query_terms = _terms(query.text)
        active_paths = {Path(path).as_posix() for path in query.active_paths}
        ranked: list[RetrievedMemory] = []
        for record in records:
            if record.status != "active":
                continue
            if (
                record.kind == "failure"
                and int(record.content.get("occurrence_count", 1)) < 2
                and not record.content.get("resolution")
            ):
                continue
            score = 0
            reasons: list[str] = []
            if record.kind == "task":
                score += 100
                reasons.append("task_memory")
            related = active_paths.intersection(record.related_paths)
            if related:
                score += 40
                reasons.append(f"related_path:{sorted(related)[0]}")
            record_terms = _terms(render_memory(record))
            keyword_matches = sorted(query_terms.intersection(record_terms))
            if keyword_matches:
                score += min(30, len(keyword_matches) * 10)
                reasons.append(f"keyword:{keyword_matches[0]}")
            if record.trust in {"verified", "observed"}:
                score += 20
                reasons.append(f"trust:{record.trust}")
            if record.scope == "project":
                score += 10
                reasons.append("project_memory")
            if record.trust == "model_claim":
                score -= 20
                reasons.append("model_claim_penalty")
            if score > 0:
                ranked.append(RetrievedMemory(record=record, score=score, reasons=reasons))

        ranked.sort(
            key=lambda item: (item.score, item.record.updated_at),
            reverse=True,
        )
        return _apply_kind_limits(ranked, query.limit)

    def validate_freshness(self) -> list[MemoryRecord]:
        return MemoryWriter(
            store=self.store,
            workspace_dir=self.workspace_dir,
        ).validate_freshness()

    def pinned_memory(self) -> str:
        return load_global_memory(self.workspace_dir)


def _terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[\w./-]{2,}", text, flags=re.UNICODE)
    }


def _apply_kind_limits(
    ranked: list[RetrievedMemory],
    total_limit: int,
) -> list[RetrievedMemory]:
    limits = {"task": 1, "file": 3, "failure": 2, "decision": 2, "project": 3}
    counts: dict[str, int] = {}
    selected: list[RetrievedMemory] = []
    for item in ranked:
        kind = item.record.kind
        if counts.get(kind, 0) >= limits.get(kind, total_limit):
            continue
        selected.append(item)
        counts[kind] = counts.get(kind, 0) + 1
        if len(selected) >= total_limit:
            break
    return selected


__all__ = ["MemoryRetriever"]
