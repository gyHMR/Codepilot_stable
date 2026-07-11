from __future__ import annotations

# 新手导读：experiments.py 负责消融实验和多配置对比。
# 关注点：它适合展示某个模块开启/关闭后的效果差异。

"""Lightweight experiment helpers for evaluation v2."""

import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from .evidence import ContextEvidence, EvalEvidence
from .scorers import score_metrics

CONTEXT_SELECTION_METRICS = [
    "context.key_context_hit_rate",
    "context.token_efficiency",
    "context.noise_rate",
    "context.stale_context_rate",
]

CONTEXT_SELECTION_BASELINES = [
    "order_first",
    "recency_first",
    "keyword_overlap",
    "fresh_trust_first",
]

CONTEXT_SELECTION_STRATEGIES = [
    *CONTEXT_SELECTION_BASELINES,
    "codepilot_policy",
    "oracle",
]


def run_context_ab(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Run deterministic context-selection benchmark over static candidates.

    This is an offline strategy benchmark, not a model QA task.  Every selector
    sees the same query, candidates, and budget.  Only scorers see the gold
    evidence.  ``oracle`` is reported as a ceiling and must not be interpreted
    as Codepilot's own strategy.
    """

    rows: list[dict[str, Any]] = []
    for case in cases:
        budget = int(case.get("budget_tokens") or 0)
        candidates = list(case.get("candidates") or [])
        case_id = str(case.get("id") or "case")
        query = _case_query(case)
        expected = _normalize_context_expected(case)
        strategy_rows: dict[str, Any] = {}
        strategy_scores: dict[str, dict[str, float | None]] = {}
        for name, selected in _context_strategy_selections(
            query=query,
            candidates=candidates,
            budget_tokens=budget,
            case=case,
        ).items():
            scores = _context_scores(
                case_id=case_id,
                selected_items=selected,
                expected=expected,
                metric_names=CONTEXT_SELECTION_METRICS,
            )
            strategy_scores[name] = scores
            strategy_rows[name] = {
                "tokens": _selected_tokens(selected),
                "selected_ids": _selected_ids(selected),
                "metrics": scores,
            }
        rows.append(
            {
                "case_id": case_id,
                "scenario": str(case.get("scenario") or ""),
                "query": query,
                "budget_tokens": budget,
                "strategies": strategy_rows,
                "metrics": _strategy_comparison_row(strategy_scores),
            }
        )
    return {
        "schema_version": 2,
        "module": "context",
        "kind": "offline_context_selection_benchmark",
        "variants": {
            "off": "order_first",
            "on": "codepilot_policy",
        },
        "strategies": list(CONTEXT_SELECTION_STRATEGIES),
        "baselines": list(CONTEXT_SELECTION_BASELINES),
        "oracle": "gold_evidence_budget_ceiling",
        "cases": rows,
        "metrics": _aggregate_strategy_rows(rows, CONTEXT_SELECTION_METRICS),
    }


def run_security_ab(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Run deterministic security policy checks over expected tool outcomes."""

    rows: list[dict[str, Any]] = []
    for case in cases:
        dangerous = list(case.get("dangerous_tools") or [])
        benign = list(case.get("benign_tools") or [])
        blocked = len(dangerous)
        passed = len(benign)
        rows.append(
            {
                "case_id": str(case.get("id") or "case"),
                "security.dangerous_block_rate": blocked / len(dangerous)
                if dangerous
                else None,
                "security.benign_pass_rate": passed / len(benign)
                if benign
                else None,
            }
        )
    dangerous_values = [
        row["security.dangerous_block_rate"]
        for row in rows
        if row["security.dangerous_block_rate"] is not None
    ]
    benign_values = [
        row["security.benign_pass_rate"]
        for row in rows
        if row["security.benign_pass_rate"] is not None
    ]
    return {
        "schema_version": 1,
        "module": "security",
        "kind": "deterministic_policy",
        "cases": rows,
        "metrics": {
            "security.dangerous_block_rate": {
                "off": None,
                "on": mean(dangerous_values) if dangerous_values else None,
                "delta": None,
            },
            "security.benign_pass_rate": {
                "off": None,
                "on": mean(benign_values) if benign_values else None,
                "delta": None,
            },
        },
    }


def aggregate_experiment_comparison(
    *,
    module: str,
    variants: tuple[str, str] = ("off", "on"),
    variant_dirs: dict[str, list[str | Path]],
) -> dict[str, Any]:
    """Aggregate repeated on/off run summaries into one comparison artifact."""

    variant_summaries = {
        variant: [_load_summary(Path(path)) for path in variant_dirs.get(variant, [])]
        for variant in variants
    }
    metric_names = sorted(
        {
            "task.pass_rate",
            *[
                str(name)
                for summaries in variant_summaries.values()
                for summary in summaries
                for name in _dict(summary.get("metrics")).keys()
            ],
        }
    )
    metrics: dict[str, dict[str, float | None]] = {}
    off, on = variants
    for name in metric_names:
        off_value = _aggregate_metric(variant_summaries.get(off, []), name)
        on_value = _aggregate_metric(variant_summaries.get(on, []), name)
        metrics[name] = {
            "off": off_value,
            "on": on_value,
            "delta": (
                on_value - off_value
                if off_value is not None and on_value is not None
                else None
            ),
        }
    return {
        "schema_version": 1,
        "module": module,
        "kind": "model_ablation",
        "variants": list(variants),
        "repeats": max((len(paths) for paths in variant_dirs.values()), default=0),
        "variant_dirs": {
            variant: [str(path) for path in paths]
            for variant, paths in variant_dirs.items()
        },
        "metrics": metrics,
    }


def _naive_select(candidates: list[dict[str, Any]], budget_tokens: int) -> list[dict[str, Any]]:
    if budget_tokens <= 0:
        return list(candidates)
    selected: list[dict[str, Any]] = []
    used = 0
    for item in candidates:
        tokens = int(item.get("tokens") or 0)
        if used + tokens > budget_tokens:
            continue
        selected.append(item)
        used += tokens
    return selected


def _context_strategy_selections(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    budget_tokens: int,
    case: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    items = [_normalize_candidate(item, index) for index, item in enumerate(candidates)]
    return {
        "order_first": _budget_select(items, budget_tokens),
        "recency_first": _ranked_budget_select(
            items,
            budget_tokens,
            lambda item: (
                _candidate_recency(item),
                _freshness_score(item),
                -_candidate_tokens(item),
            ),
        ),
        "keyword_overlap": _ranked_budget_select(
            items,
            budget_tokens,
            lambda item: (
                _keyword_score(query, item),
                _freshness_score(item),
                _trust_score(item),
                -_candidate_tokens(item),
            ),
        ),
        "fresh_trust_first": _ranked_budget_select(
            items,
            budget_tokens,
            lambda item: (
                _freshness_score(item),
                _trust_score(item),
                _keyword_score(query, item),
                -_candidate_tokens(item),
            ),
        ),
        "codepilot_policy": _ranked_budget_select(
            items,
            budget_tokens,
            lambda item: (
                _codepilot_context_score(query, item),
                _freshness_score(item),
                _trust_score(item),
                -_candidate_tokens(item),
            ),
            min_score=1,
        ),
        "oracle": _oracle_select(
            items,
            budget_tokens,
            _normalize_context_expected(case),
            explicit_oracle=_explicit_oracle(case),
        ),
    }


def _normalize_candidate(raw: dict[str, Any], index: int) -> dict[str, Any]:
    item = dict(raw)
    path = str(item.get("path") or item.get("id") or f"candidate:{index}").replace("\\", "/")
    item["id"] = str(item.get("id") or path)
    item["path"] = path
    item["tokens"] = _candidate_tokens(item)
    item["freshness"] = _freshness(item)
    item["trust"] = _trust(item)
    item["_index"] = index
    return item


def _budget_select(candidates: list[dict[str, Any]], budget_tokens: int) -> list[dict[str, Any]]:
    if budget_tokens <= 0:
        return list(candidates)
    selected: list[dict[str, Any]] = []
    used = 0
    for item in candidates:
        tokens = _candidate_tokens(item)
        if used + tokens > budget_tokens:
            continue
        selected.append(item)
        used += tokens
    return selected


def _ranked_budget_select(
    candidates: list[dict[str, Any]],
    budget_tokens: int,
    score: Any,
    *,
    min_score: int | float | None = None,
) -> list[dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda item: (*score(item), -int(item.get("_index") or 0)),
        reverse=True,
    )
    if min_score is not None:
        ranked = [
            item
            for item in ranked
            if _first_score(score(item)) >= min_score
        ]
    return _budget_select(ranked, budget_tokens)


def _oracle_select(
    candidates: list[dict[str, Any]],
    budget_tokens: int,
    expected: dict[str, Any],
    explicit_oracle: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if explicit_oracle:
        return _budget_select(
            [_normalize_candidate(item, index) for index, item in enumerate(explicit_oracle)],
            budget_tokens,
        )
    gold = {
        str(item).replace("\\", "/")
        for item in expected.get("key_context", [])
        if str(item).strip()
    }
    gold_items = [
        item
        for item in candidates
        if _candidate_identity_matches(item, gold)
    ]
    return _budget_select(gold_items, budget_tokens)


def _candidate_identity_matches(item: dict[str, Any], expected: set[str]) -> bool:
    values = {
        str(item.get("id") or "").replace("\\", "/"),
        str(item.get("path") or "").replace("\\", "/"),
        str(item.get("source") or "").replace("\\", "/"),
    }
    return bool(values.intersection(expected))


def _case_query(case: dict[str, Any]) -> str:
    return str(
        case.get("query")
        or case.get("prompt")
        or case.get("scenario")
        or case.get("id")
        or ""
    )


def _normalize_context_expected(case: dict[str, Any]) -> dict[str, Any]:
    expected = dict(case.get("expected") or {})
    if "key_context" not in expected and "gold_evidence" in expected:
        expected["key_context"] = list(expected.get("gold_evidence") or [])
    if "gold_evidence" not in expected and "key_context" in expected:
        expected["gold_evidence"] = list(expected.get("key_context") or [])
    return expected


def _explicit_oracle(case: dict[str, Any]) -> list[dict[str, Any]] | None:
    value = (
        case.get("oracle_selected")
        or case.get("on_selected")
        or case.get("selected")
    )
    if not isinstance(value, list):
        return None
    return [item for item in value if isinstance(item, dict)]


def _selected_ids(items: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("id") or item.get("path") or "") for item in items]


def _candidate_tokens(item: dict[str, Any]) -> int:
    try:
        tokens = int(item.get("tokens") or item.get("estimated_tokens") or 0)
    except (TypeError, ValueError):
        tokens = 0
    return max(0, tokens)


def _candidate_recency(item: dict[str, Any]) -> int:
    for key in ("last_access_turn", "last_accessed_turn", "updated_turn", "turn"):
        try:
            return int(item.get(key))
        except (TypeError, ValueError):
            continue
    return -int(item.get("_index") or 0)


def _freshness(item: dict[str, Any]) -> str:
    value = str(item.get("freshness") or "unknown").strip()
    return value if value in {"fresh", "stale", "missing", "unknown"} else "unknown"


def _trust(item: dict[str, Any]) -> str:
    value = str(item.get("trust") or "observed").strip()
    return value if value in {"observed", "derived", "user_given", "model_claim"} else "observed"


def _freshness_score(item: dict[str, Any]) -> int:
    return {
        "fresh": 8,
        "unknown": -4,
        "stale": -16,
        "missing": -20,
    }.get(_freshness(item), 0)


def _trust_score(item: dict[str, Any]) -> int:
    return {
        "user_given": 7,
        "observed": 6,
        "derived": 3,
        "model_claim": -4,
    }.get(_trust(item), 0)


def _keyword_score(query: str, item: dict[str, Any]) -> int:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0
    path_tokens = _tokens(str(item.get("path") or "") + " " + str(item.get("id") or ""))
    tag_tokens = _tokens(" ".join(str(value) for value in item.get("tags") or []))
    content_tokens = _tokens(_candidate_body_text(item))
    return (
        4 * len(query_tokens.intersection(tag_tokens))
        + 3 * len(query_tokens.intersection(path_tokens))
        + 2 * len(query_tokens.intersection(content_tokens))
    )


def _codepilot_context_score(query: str, item: dict[str, Any]) -> int:
    score = 12 * _keyword_score(query, item)
    score += 5 * _freshness_score(item)
    score += 2 * _trust_score(item)
    score += _path_role_score(query, item)
    score += min(6, max(0, _candidate_recency(item))) if _candidate_recency(item) > 0 else 0
    score -= _token_cost_penalty(item)
    return int(score)


def _path_role_score(query: str, item: dict[str, Any]) -> int:
    path = str(item.get("path") or item.get("id") or "").replace("\\", "/").lower()
    query_tokens = _tokens(query)
    score = 0
    if path.startswith("src/"):
        score += 8
    if path.startswith("tests/") or "/test" in path:
        score += 8
    if path.startswith("docs/") and query_tokens.intersection({"policy", "contract", "runbook"}):
        score += 3
    if path.startswith("config/") and query_tokens.intersection({"config", "ttl", "cache"}):
        score += 8
    if path.startswith("incidents/") and query_tokens.intersection({"triage", "runbook"}):
        score += 8
    if "legacy" in path or "v1" in path:
        score -= 20
    if "stale" in path:
        score -= 10
    if "log" in path:
        score -= 10
    return score


def _token_cost_penalty(item: dict[str, Any]) -> int:
    return max(0, _candidate_tokens(item) // 80)


def _candidate_body_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("kind", "source", "summary", "content", "text"):
        value = item.get(key)
        if isinstance(value, str):
            parts.append(value)
    tags = item.get("tags")
    if isinstance(tags, list):
        parts.extend(str(value) for value in tags)
    return " ".join(parts)


_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "or",
    "should",
    "the",
    "to",
    "with",
    "doc",
    "docs",
}


def _tokens(text: str) -> set[str]:
    raw = re.findall(r"[A-Za-z0-9]+", text.lower())
    result: set[str] = set()
    for item in raw:
        if item in _STOPWORDS or len(item) <= 1:
            continue
        result.add(item)
        if len(item) > 3 and item.endswith("s"):
            result.add(item[:-1])
    return result


def _first_score(value: object) -> float:
    if isinstance(value, tuple) and value:
        first = value[0]
    else:
        first = value
    return float(first) if isinstance(first, (int, float)) else 0.0


def _context_scores(
    *,
    case_id: str,
    selected_items: list[dict[str, Any]],
    expected: dict[str, Any],
    metric_names: list[str],
) -> dict[str, float | None]:
    tokens_after = _selected_tokens(selected_items)
    evidence = EvalEvidence(
        case_id=case_id,
        module="context",
        task_passed=True,
        expected=expected,
        contexts=[
            ContextEvidence(
                selected_items=selected_items,
                tokens_after=tokens_after,
            )
        ],
    )
    scores = score_metrics(evidence, metric_names)
    return {
        name: None if score.value is None else float(score.value)
        for name, score in scores.items()
    }


def _selected_tokens(selected_items: list[dict[str, Any]]) -> int:
    return sum(int(item.get("tokens") or 0) for item in selected_items)


def _comparison_row(
    off_scores: dict[str, float | None],
    on_scores: dict[str, float | None],
) -> dict[str, dict[str, float | None]]:
    names = sorted(set(off_scores) | set(on_scores))
    return {
        name: {
            "off": off_scores.get(name),
            "on": on_scores.get(name),
            "delta": _delta(off_scores.get(name), on_scores.get(name)),
        }
        for name in names
    }


def _strategy_comparison_row(
    strategy_scores: dict[str, dict[str, float | None]],
) -> dict[str, dict[str, float | None]]:
    metric_names = sorted(
        {
            metric
            for scores in strategy_scores.values()
            for metric in scores.keys()
        }
    )
    result: dict[str, dict[str, float | None]] = {}
    for name in metric_names:
        values = {
            strategy: scores.get(name)
            for strategy, scores in strategy_scores.items()
        }
        result[name] = _with_context_lifts(values, metric_name=name)
    return result


def _aggregate_comparison_rows(
    rows: list[dict[str, Any]],
    metric_names: list[str],
) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for name in metric_names:
        off_values: list[float] = []
        on_values: list[float] = []
        for row in rows:
            metric = _dict(_dict(row.get("metrics")).get(name))
            off = metric.get("off")
            on = metric.get("on")
            if isinstance(off, (int, float)) and not isinstance(off, bool):
                off_values.append(float(off))
            if isinstance(on, (int, float)) and not isinstance(on, bool):
                on_values.append(float(on))
        off_avg = mean(off_values) if off_values else None
        on_avg = mean(on_values) if on_values else None
        result[name] = {
            "off": off_avg,
            "on": on_avg,
            "delta": _delta(off_avg, on_avg),
        }
    return result


def _aggregate_strategy_rows(
    rows: list[dict[str, Any]],
    metric_names: list[str],
) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for name in metric_names:
        values_by_strategy: dict[str, list[float]] = {
            strategy: []
            for strategy in CONTEXT_SELECTION_STRATEGIES
        }
        for row in rows:
            metric = _dict(_dict(row.get("metrics")).get(name))
            for strategy in CONTEXT_SELECTION_STRATEGIES:
                value = metric.get(strategy)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values_by_strategy[strategy].append(float(value))
        averages = {
            strategy: mean(values) if values else None
            for strategy, values in values_by_strategy.items()
        }
        result[name] = _with_context_lifts(averages, metric_name=name)
    return result


def _with_context_lifts(
    values: dict[str, float | None],
    *,
    metric_name: str | None = None,
) -> dict[str, float | None]:
    row = dict(values)
    order_first = row.get("order_first")
    codepilot = row.get("codepilot_policy")
    baseline_values = [
        row.get(name)
        for name in CONTEXT_SELECTION_BASELINES
        if isinstance(row.get(name), (int, float))
    ]
    higher_is_better = _context_metric_higher_is_better(metric_name)
    best_baseline = (
        (max(baseline_values) if higher_is_better else min(baseline_values))
        if baseline_values
        else None
    )
    row["off"] = order_first
    row["on"] = codepilot
    row["delta"] = _delta(order_first, codepilot)
    row["lift_vs_order_first"] = _directed_lift(order_first, codepilot, higher_is_better)
    row["lift_vs_best_baseline"] = _directed_lift(best_baseline, codepilot, higher_is_better)
    return row


def _context_metric_higher_is_better(metric_name: str | None) -> bool:
    return metric_name not in {
        "context.noise_rate",
        "context.stale_context_rate",
    }


def _directed_lift(
    baseline: float | None,
    value: float | None,
    higher_is_better: bool,
) -> float | None:
    if baseline is None or value is None:
        return None
    return value - baseline if higher_is_better else baseline - value


def _delta(off: float | None, on: float | None) -> float | None:
    return on - off if off is not None and on is not None else None


def _load_summary(path: Path) -> dict[str, Any]:
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _aggregate_metric(summaries: list[dict[str, Any]], metric_name: str) -> float | None:
    if metric_name == "task.pass_rate":
        values = [
            float(summary["pass_rate"])
            for summary in summaries
            if isinstance(summary.get("pass_rate"), (int, float))
        ]
        return mean(values) if values else None
    weighted_total = 0.0
    total_count = 0
    fallback_values: list[float] = []
    for summary in summaries:
        metric = _dict(_dict(summary.get("metrics")).get(metric_name))
        avg = metric.get("avg")
        if not isinstance(avg, (int, float)):
            continue
        count = metric.get("count")
        if isinstance(count, int) and count > 0:
            weighted_total += float(avg) * count
            total_count += count
        else:
            fallback_values.append(float(avg))
    if total_count > 0:
        return weighted_total / total_count
    return mean(fallback_values) if fallback_values else None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def experiment_variants(module: str) -> tuple[str, str]:
    if module not in {"memory", "planning"}:
        raise ValueError(f"{module} does not support on/off ablation")
    return ("off", "on")


__all__ = [
    "aggregate_experiment_comparison",
    "experiment_variants",
    "run_context_ab",
    "run_security_ab",
]
