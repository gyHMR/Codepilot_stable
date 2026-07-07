from __future__ import annotations

# 新手导读：MemoryRetriever 只读 memories.jsonl，给 ContextGovernor 返回可解释召回结果。
# 关注点：candidate/disabled/superseded/deleted 都不会进入上下文。

"""Canonical Memory v4 retrieval."""

import re
from pathlib import Path

from .records import MemoryQuery, MemoryRecall, MemoryRecord, RetrievedMemory
from .rendering import render_memory
from .store import MemoryStore


MEMORY_RECALL_MIN = 3
MEMORY_RECALL_MAX = 5


class MemoryRetriever:
    """Recall active durable memory for context projection."""

    def __init__(self, *, store: MemoryStore, workspace_dir: str | Path) -> None:
        self.store = store
        self.workspace_dir = Path(workspace_dir)

    def recall(self, query: MemoryQuery) -> MemoryRecall:
        dropped: dict[str, str] = {}
        scored: list[RetrievedMemory] = []
        for record in self.store.all_records():
            reason = _drop_reason(record, query)
            if reason is not None:
                dropped[record.id] = reason
                continue
            item = score_memory_record(record, query)
            if item is None:
                dropped[record.id] = "low_score"
                continue
            scored.append(item)

        ranked = _dedupe_by_subject(
            sorted(
                scored,
                key=lambda item: (
                    item.score,
                    item.record.priority,
                    item.record.updated_at,
                ),
                reverse=True,
            ),
            dropped,
        )
        limit = min(MEMORY_RECALL_MAX, max(MEMORY_RECALL_MIN, query.limit))
        selected = ranked[:limit]
        selected_ids = {item.record.id for item in selected}
        for item in ranked[limit:]:
            if item.record.id not in dropped:
                dropped[item.record.id] = "over_limit"
        return MemoryRecall(
            retrieved=selected,
            dropped={
                memory_id: reason
                for memory_id, reason in dropped.items()
                if memory_id not in selected_ids
            },
        )


def score_memory_record(record: MemoryRecord, query: MemoryQuery) -> RetrievedMemory | None:
    score = 0
    reasons: list[str] = []
    query_terms = _terms(query.text)
    rendered_terms = _terms(render_memory(record))

    if record.scope == "project":
        score += 20
        reasons.append("scope:project")
    elif record.scope == "workspace":
        score += 12
        reasons.append("scope:workspace")
    elif record.scope == "global":
        score += 6
        reasons.append("scope:global")

    type_score = {
        "constraint": 35,
        "correction": 35,
        "preference": 25,
        "decision": 22,
        "workflow": 20,
        "experience": 18,
    }.get(record.type, 0)
    score += type_score
    reasons.append(f"type:{record.type}")

    subject_matches = _keyword_matches(query_terms, _terms(record.subject))
    if subject_matches:
        score += min(35, len(subject_matches) * 12)
        reasons.append(f"subject:{sorted(subject_matches)[0]}")

    keyword_matches = _keyword_matches(query_terms, set(_keywords(record)) | rendered_terms)
    if keyword_matches:
        score += min(45, len(keyword_matches) * 10)
        reasons.append(f"keyword:{sorted(keyword_matches)[0]}")

    path_matches = _path_matches(record, query)
    if path_matches:
        score += min(40, len(path_matches) * 15)
        reasons.append(f"path:{path_matches[0]}")

    task_score, task_reasons = _task_signal_score(record, query)
    score += task_score
    reasons.extend(task_reasons)

    if record.priority:
        score += record.priority * 8
        reasons.append(f"priority:{record.priority}")
    if record.occurrences > 1:
        score += min(20, record.occurrences * 5)
        reasons.append(f"occurrences:{record.occurrences}")

    if score <= 0:
        return None
    return RetrievedMemory(record=record, score=score, reasons=_dedupe(reasons))


def _drop_reason(record: MemoryRecord, query: MemoryQuery) -> str | None:
    if record.status != "active":
        return f"status:{record.status}"
    if record.scope not in {"project", "workspace", "global"}:
        return f"scope:{record.scope}"
    if _conflicts_with_current_instruction(record, query):
        return "conflict:latest_instruction"
    return None


def _path_matches(record: MemoryRecord, query: MemoryQuery) -> list[str]:
    active = {Path(path).as_posix() for path in [*query.active_paths, *query.changed_paths]}
    matches: list[str] = []
    for path in record.paths:
        normalized = Path(path).as_posix()
        if normalized in active:
            matches.append(normalized)
    for keyword in _keywords(record):
        if keyword.startswith("path:"):
            path = keyword.removeprefix("path:")
            if path in active:
                matches.append(path)
    return _dedupe(matches)


def _task_signal_score(record: MemoryRecord, query: MemoryQuery) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    keywords = _keywords(record)
    if query.current_mode and f"mode:{query.current_mode}" in keywords:
        score += 25
        reasons.append(f"mode:{query.current_mode}")
    if query.current_step_kind and f"kind:{query.current_step_kind}" in keywords:
        score += 25
        reasons.append(f"kind:{query.current_step_kind}")
    if query.verification_status and record.type == "experience":
        score += 10
        reasons.append(f"verification:{query.verification_status}")
    if query.blocked_reason:
        matches = _keyword_matches(_terms(query.blocked_reason), set(keywords) | _terms(record.content))
        if matches:
            score += 35
            reasons.append(f"blocked:{sorted(matches)[0]}")
    return score, reasons


def _dedupe_by_subject(
    items: list[RetrievedMemory],
    dropped: dict[str, str],
) -> list[RetrievedMemory]:
    best: dict[str, RetrievedMemory] = {}
    order: list[str] = []
    for item in items:
        subject = item.record.subject.lower()
        previous = best.get(subject)
        if previous is None:
            best[subject] = item
            order.append(subject)
            continue
        if (item.score, item.record.priority, item.record.updated_at) > (
            previous.score,
            previous.record.priority,
            previous.record.updated_at,
        ):
            dropped[previous.record.id] = "duplicate_subject"
            best[subject] = item
        else:
            dropped[item.record.id] = "duplicate_subject"
    return [best[subject] for subject in order]


def _conflicts_with_current_instruction(record: MemoryRecord, query: MemoryQuery) -> bool:
    text = query.latest_user_message.lower()
    if not text:
        return False
    if not any(
        marker in text
        for marker in (
            "不要",
            "别",
            "不是",
            "而是",
            "改成",
            "纠正",
            "更正",
            "not ",
            "instead",
            "never",
            "correction",
        )
    ):
        return False
    query_terms = _terms(query.latest_user_message)
    record_terms = _terms(record.subject) | _terms(record.value) | set(_keywords(record))
    return bool(_keyword_matches(query_terms, record_terms))


def _terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[\w./-]{2,}", text, flags=re.UNICODE)
    }


def _keyword_matches(query_terms: set[str], record_terms: set[str]) -> set[str]:
    matches = set(query_terms.intersection({term.lower() for term in record_terms}))
    for query_term in query_terms:
        for record_term in record_terms:
            normalized = record_term.lower()
            if query_term == normalized:
                continue
            if query_term in normalized or normalized in query_term:
                matches.add(normalized)
    return matches


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _keywords(record: MemoryRecord) -> list[str]:
    return list(getattr(record, "keywords"))


__all__ = ["MemoryRetriever", "score_memory_record"]
