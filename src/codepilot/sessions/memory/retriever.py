from __future__ import annotations

"""Layered durable memory recall."""

import re
from pathlib import Path

from .files import load_global_memory, sanitize_memory_text
from .records import MemoryQuery, MemoryRecall, MemoryRecord, RetrievedMemory
from .rendering import render_memory
from .store import MemoryStore


PINNED_MEMORY_LIMIT = 2000
EXPERIENCE_LIMIT = 3


class MemoryRetriever:
    """Recall durable memory for ContextGovernor."""

    def __init__(self, *, store: MemoryStore, workspace_dir: str | Path) -> None:
        self.store = store
        self.workspace_dir = Path(workspace_dir)

    def recall(self, query: MemoryQuery) -> MemoryRecall:
        dropped: dict[str, str] = {}
        active: list[MemoryRecord] = []
        for record in self.store.all_records():
            reason = record.retrieval_exclusion_reason()
            if reason is not None:
                dropped[record.id] = reason
                continue
            active.append(record)

        corrections = [
            RetrievedMemory(record, 1000, ["layer:correction"])
            for record in active
            if record.kind == "correction"
        ]
        always_constraints = [
            item
            for record in active
            if record.kind == "constraint" and "always" in record.triggers
            if (item := score_memory_record(record, query, force_reason="layer:always_constraint"))
            is not None
        ]
        selected_candidates = [
            scored
            for record in active
            if record.kind not in {"correction"}
            and not (record.kind == "constraint" and "always" in record.triggers)
            if (scored := score_memory_record(record, query)) is not None
        ]
        selected_candidates.sort(
            key=lambda item: (_kind_order(item.record.kind), item.score, item.record.updated_at),
            reverse=True,
        )
        selected = _apply_selected_limits(selected_candidates, query.limit)
        return MemoryRecall(
            pinned_text=self.pinned_memory(),
            always=[*corrections, *always_constraints],
            selected=selected,
            dropped=dropped,
        )

    def retrieve(self, query: MemoryQuery) -> list[RetrievedMemory]:
        """Compatibility adapter for callers that only need selected records."""

        return self.recall(query).retrieved

    def validate_freshness(self) -> list[MemoryRecord]:
        return []

    def pinned_memory(self) -> str:
        text = load_global_memory(self.workspace_dir)
        return sanitize_memory_text(text, limit=PINNED_MEMORY_LIMIT)


def score_memory_record(
    record: MemoryRecord,
    query: MemoryQuery,
    *,
    force_reason: str | None = None,
) -> RetrievedMemory | None:
    if record.retrieval_exclusion_reason() is not None:
        return None

    score = 0
    reasons: list[str] = []
    if force_reason:
        score += 500
        reasons.append(force_reason)

    trigger_score, trigger_reasons = _trigger_score(record, query)
    score += trigger_score
    reasons.extend(trigger_reasons)

    keyword_matches = sorted(_keyword_matches(_terms(query.text), _terms(render_memory(record))))
    if keyword_matches:
        score += min(40, len(keyword_matches) * 10)
        reasons.append(f"keyword:{keyword_matches[0]}")

    if record.scope == "project":
        score += 10
        reasons.append("scope:project")

    mode = query.retrieval_mode or ""
    if record.kind == "decision" and mode in {"qa", "plan", "design"}:
        score += 40
        reasons.append(f"mode:{mode}_decision")
    if record.kind == "constraint" and (mode in {"qa", "plan", "design"} or keyword_matches):
        score += 25
        reasons.append("constraint_relevant")
    if record.kind == "experience":
        if mode in {"repair", "verify"}:
            score += 35
            reasons.append(f"mode:{mode}_experience")
        if query.recent_error and f"error:{query.recent_error}" in record.triggers:
            score += 60
            reasons.append(f"error:{query.recent_error}")
        if query.action_intent and f"intent:{query.action_intent}" in record.triggers:
            score += 40
            reasons.append(f"intent:{query.action_intent}")
        if record.occurrences > 1:
            score += min(30, record.occurrences * 10)
            reasons.append(f"occurrences:{record.occurrences}")

    if score <= 0:
        return None
    return RetrievedMemory(record=record, score=score, reasons=_dedupe(reasons))


def _trigger_score(record: MemoryRecord, query: MemoryQuery) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    active_paths = {Path(path).as_posix() for path in query.active_paths}
    for trigger in record.triggers:
        if trigger == "always":
            score += 100
            reasons.append("trigger:always")
        elif trigger.startswith("path:"):
            path = trigger.removeprefix("path:")
            if path in active_paths:
                score += 50
                reasons.append(f"path:{path}")
        elif trigger.startswith("topic:"):
            topic = trigger.removeprefix("topic:")
            if topic and (topic in query.text.lower() or topic in _terms(query.text)):
                score += 30
                reasons.append(f"topic:{topic}")
        elif trigger.startswith("phase:") and query.task_phase:
            phase = trigger.removeprefix("phase:")
            if phase == query.task_phase:
                score += 25
                reasons.append(f"phase:{phase}")
        elif trigger.startswith("intent:") and query.action_intent:
            intent = trigger.removeprefix("intent:")
            if intent == query.action_intent:
                score += 35
                reasons.append(f"intent:{intent}")
        elif trigger.startswith("error:") and query.recent_error:
            error = trigger.removeprefix("error:")
            if error == query.recent_error:
                score += 50
                reasons.append(f"error:{error}")
    return score, reasons


def _apply_selected_limits(
    candidates: list[RetrievedMemory],
    total_limit: int,
) -> list[RetrievedMemory]:
    limits = {
        "constraint": 3,
        "decision": 2,
        "experience": EXPERIENCE_LIMIT,
    }
    counts: dict[str, int] = {}
    selected: list[RetrievedMemory] = []
    for item in candidates:
        kind = item.record.kind
        if counts.get(kind, 0) >= limits.get(kind, total_limit):
            continue
        selected.append(item)
        counts[kind] = counts.get(kind, 0) + 1
        if len(selected) >= total_limit:
            break
    selected.sort(key=lambda item: _output_order(item.record.kind))
    return selected


def _output_order(kind: str) -> int:
    return {
        "constraint": 0,
        "decision": 1,
        "experience": 2,
    }.get(kind, 9)


def _kind_order(kind: str) -> int:
    return {
        "constraint": 1,
        "decision": 2,
        "experience": 3,
    }.get(kind, 0)


def _terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[\w./-]{2,}", text, flags=re.UNICODE)
    }


def _keyword_matches(query_terms: set[str], record_terms: set[str]) -> set[str]:
    matches = set(query_terms.intersection(record_terms))
    for query_term in query_terms:
        for record_term in record_terms:
            if query_term == record_term:
                continue
            if query_term in record_term or record_term in query_term:
                matches.add(record_term)
    return matches


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


__all__ = ["MemoryRetriever", "score_memory_record"]
