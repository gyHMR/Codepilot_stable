from __future__ import annotations

# 新手导读：memory_retrieval.py 是离线记忆检索评测器。
# 关注点：它评估 MemoryRetriever 的排序/过滤能力，不跑模型，也不改正常 Agent 记忆链路。

"""Offline memory retrieval benchmark.

The benchmark uses a shared memory corpus plus structured retrieval queries.
Each case asks the same retriever used by runtime to rank memories, then scores
the ranking against gold, forbidden, stale/superseded, and no-relevant labels.
"""

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from codepilot.sessions.memory import (
    MemoryQuery,
    MemoryRecord,
    MemoryRetriever,
)


DEFAULT_MEMORY_CASES_PATH = Path("benchmarks/evaluation_memory/cases.json")
DEFAULT_MEMORY_CORPUS_PATH = Path(
    "benchmarks/evaluation_memory/corpus/issue_tracker_memory.jsonl"
)

MEMORY_RETRIEVAL_METRICS = [
    "memory.recall@1",
    "memory.recall@3",
    "memory.recall@5",
    "memory.precision@1",
    "memory.precision@3",
    "memory.precision@5",
    "memory.mrr",
    "memory.ndcg@5",
    "memory.noise_rate@5",
    "memory.forbidden_retrieval_rate",
    "memory.stale_retrieval_rate",
    "memory.no_relevant_rejection_rate",
]

_HIGHER_IS_BETTER = {
    "memory.recall@1",
    "memory.recall@3",
    "memory.recall@5",
    "memory.precision@1",
    "memory.precision@3",
    "memory.precision@5",
    "memory.mrr",
    "memory.ndcg@5",
    "memory.no_relevant_rejection_rate",
}


class _InMemoryStore:
    def __init__(self, records: list[MemoryRecord]) -> None:
        self._records = records

    def all_records(self) -> list[MemoryRecord]:
        return list(self._records)


def load_memory_retrieval_cases(path: Path | str = DEFAULT_MEMORY_CASES_PATH) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        cases = payload.get("cases")
    else:
        cases = payload
    if not isinstance(cases, list):
        raise ValueError(f"Memory retrieval cases must be a list or object with cases: {path}")
    return [item for item in cases if isinstance(item, dict)]


def load_memory_corpus(path: Path | str = DEFAULT_MEMORY_CORPUS_PATH) -> list[MemoryRecord]:
    records: list[MemoryRecord] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"Memory corpus line {line_number} is not an object: {path}")
        records.append(MemoryRecord.from_dict(raw))
    return records


def run_memory_retrieval_benchmark(
    cases: list[dict[str, Any]],
    corpus: list[MemoryRecord],
    *,
    workspace_dir: Path | str = ".",
) -> dict[str, Any]:
    rows = [
        _run_memory_case(case, corpus, workspace_dir=workspace_dir)
        for case in cases
    ]
    return {
        "schema_version": 1,
        "module": "memory",
        "kind": "offline_memory_retrieval_benchmark",
        "case_count": len(rows),
        "corpus_size": len(corpus),
        "metrics": _aggregate_memory_rows(rows),
        "cases": rows,
    }


def _run_memory_case(
    case: dict[str, Any],
    corpus: list[MemoryRecord],
    *,
    workspace_dir: Path | str,
) -> dict[str, Any]:
    query = _memory_query(case)
    retriever = MemoryRetriever(
        store=_InMemoryStore(corpus),  # type: ignore[arg-type]
        workspace_dir=workspace_dir,
    )
    recall = retriever.recall(query)
    ranked = [
        {
            "id": item.record.id,
            "score": item.score,
            "reasons": list(item.reasons),
            "type": item.record.type,
            "status": item.record.status,
            "subject": item.record.subject,
            "paths": list(item.record.paths),
        }
        for item in recall.retrieved
    ]
    expected = _expected(case)
    selected_ids = [item["id"] for item in ranked]
    metrics = _score_ranking(selected_ids, expected)
    return {
        "case_id": str(case.get("id") or "case"),
        "scenario": str(case.get("scenario") or ""),
        "query": _query_payload(query),
        "expected": expected,
        "retrieved": ranked,
        "dropped": dict(recall.dropped),
        "metrics": metrics,
    }


def _memory_query(case: dict[str, Any]) -> MemoryQuery:
    query = case.get("query") if isinstance(case.get("query"), dict) else {}
    return MemoryQuery(
        latest_user_message=str(query.get("latest_user_message") or query.get("text") or ""),
        raw_user_request=str(query.get("raw_user_request") or ""),
        goal=str(query.get("goal") or ""),
        current_mode=_optional_text(query.get("current_mode")),
        current_step_title=_optional_text(query.get("current_step_title")),
        current_step_kind=_optional_text(query.get("current_step_kind")),
        verification_status=_optional_text(query.get("verification_status")),
        blocked_reason=_optional_text(query.get("blocked_reason")),
        active_paths=_string_list(query.get("active_paths")),
        changed_paths=_string_list(query.get("changed_paths")),
        artifact_summaries=_string_list(query.get("artifact_summaries")),
        session_id=_optional_text(query.get("session_id")),
        run_id=_optional_text(query.get("run_id")),
        limit=_positive_int(query.get("limit"), default=5),
    )


def _query_payload(query: MemoryQuery) -> dict[str, Any]:
    return {
        "latest_user_message": query.latest_user_message,
        "raw_user_request": query.raw_user_request,
        "goal": query.goal,
        "current_mode": query.current_mode,
        "current_step_title": query.current_step_title,
        "current_step_kind": query.current_step_kind,
        "verification_status": query.verification_status,
        "blocked_reason": query.blocked_reason,
        "active_paths": list(query.active_paths),
        "changed_paths": list(query.changed_paths),
        "limit": query.limit,
    }


def _expected(case: dict[str, Any]) -> dict[str, Any]:
    raw = case.get("expected") if isinstance(case.get("expected"), dict) else {}
    gold = _string_list(raw.get("gold_memory_ids") or raw.get("memory_ids"))
    relevance = {
        str(key): float(value)
        for key, value in (raw.get("relevance_grades") or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    for memory_id in gold:
        relevance.setdefault(memory_id, 1.0)
    return {
        "gold_memory_ids": gold,
        "forbidden_memory_ids": _string_list(raw.get("forbidden_memory_ids")),
        "stale_memory_ids": _string_list(raw.get("stale_memory_ids")),
        "relevance_grades": relevance,
        "no_relevant": bool(raw.get("no_relevant")),
    }


def _score_ranking(selected_ids: list[str], expected: dict[str, Any]) -> dict[str, float | None]:
    gold = set(_string_list(expected.get("gold_memory_ids")))
    forbidden = set(_string_list(expected.get("forbidden_memory_ids")))
    stale = set(_string_list(expected.get("stale_memory_ids")))
    no_relevant = bool(expected.get("no_relevant")) or not gold
    relevance = {
        str(key): float(value)
        for key, value in _dict(expected.get("relevance_grades")).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    relevant = gold | {
        memory_id
        for memory_id, grade in relevance.items()
        if grade > 0
    }
    return {
        "memory.recall@1": None if no_relevant else _recall_at(selected_ids, gold, 1),
        "memory.recall@3": None if no_relevant else _recall_at(selected_ids, gold, 3),
        "memory.recall@5": None if no_relevant else _recall_at(selected_ids, gold, 5),
        "memory.precision@1": None if no_relevant else _precision_at(selected_ids, relevant, 1),
        "memory.precision@3": None if no_relevant else _precision_at(selected_ids, relevant, 3),
        "memory.precision@5": None if no_relevant else _precision_at(selected_ids, relevant, 5),
        "memory.mrr": None if no_relevant else _mrr(selected_ids, gold),
        "memory.ndcg@5": None if no_relevant else _ndcg_at(selected_ids, relevance, 5),
        "memory.noise_rate@5": _noise_rate_at(selected_ids, relevant, 5),
        "memory.forbidden_retrieval_rate": _label_retrieval_rate(selected_ids, forbidden),
        "memory.stale_retrieval_rate": _label_retrieval_rate(selected_ids, stale),
        "memory.no_relevant_rejection_rate": (
            1.0 if no_relevant and not selected_ids else 0.0 if no_relevant else None
        ),
    }


def _aggregate_memory_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for name in MEMORY_RETRIEVAL_METRICS:
        values = [
            float(metric)
            for row in rows
            if isinstance((metric := _dict(row.get("metrics")).get(name)), (int, float))
            and not isinstance(metric, bool)
        ]
        avg = mean(values) if values else None
        result[name] = {
            "avg": avg,
            "count": len(values),
            "higher_is_better": name in _HIGHER_IS_BETTER,
        }
    return result


def _recall_at(selected_ids: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(selected_ids[:k]).intersection(gold)) / len(gold)


def _precision_at(selected_ids: list[str], gold: set[str], k: int) -> float:
    selected = selected_ids[:k]
    if not selected:
        return 0.0
    return len(set(selected).intersection(gold)) / len(selected)


def _mrr(selected_ids: list[str], gold: set[str]) -> float:
    for index, memory_id in enumerate(selected_ids, start=1):
        if memory_id in gold:
            return 1.0 / index
    return 0.0


def _ndcg_at(selected_ids: list[str], relevance: dict[str, float], k: int) -> float:
    if not relevance:
        return 0.0
    dcg = _dcg([relevance.get(memory_id, 0.0) for memory_id in selected_ids[:k]])
    ideal = _dcg(sorted(relevance.values(), reverse=True)[:k])
    if ideal <= 0:
        return 0.0
    return dcg / ideal


def _dcg(grades: list[float]) -> float:
    return sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))


def _noise_rate_at(selected_ids: list[str], gold: set[str], k: int) -> float:
    selected = selected_ids[:k]
    if not selected:
        return 0.0
    noise = sum(1 for memory_id in selected if memory_id not in gold)
    return noise / len(selected)


def _label_retrieval_rate(selected_ids: list[str], labels: set[str]) -> float | None:
    if not labels:
        return None
    return len(set(selected_ids).intersection(labels)) / len(labels)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _positive_int(value: object, *, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


__all__ = [
    "DEFAULT_MEMORY_CASES_PATH",
    "DEFAULT_MEMORY_CORPUS_PATH",
    "MEMORY_RETRIEVAL_METRICS",
    "load_memory_corpus",
    "load_memory_retrieval_cases",
    "run_memory_retrieval_benchmark",
]
