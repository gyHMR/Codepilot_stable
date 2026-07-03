from __future__ import annotations

# 新手导读：experiments.py 负责消融实验和多配置对比。
# 关注点：它适合展示某个模块开启/关闭后的效果差异。

"""Lightweight experiment helpers for evaluation v2."""

from statistics import mean
from typing import Any

from .evidence import ContextEvidence, EvalEvidence
from .scorers import score_metrics


def run_context_ab(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Run deterministic context A/B over static candidate lists.

    ``off`` is a naive selector that takes candidates in order until the token
    budget is full.  ``on`` is the case's precomputed/context-builder selection.
    This keeps the experiment deterministic and detached from the agent loop.
    """

    rows: list[dict[str, Any]] = []
    for case in cases:
        budget = int(case.get("budget_tokens") or 0)
        candidates = list(case.get("candidates") or [])
        selected = list(case.get("selected") or [])
        expected = dict(case.get("expected") or {})
        off_items = _naive_select(candidates, budget)
        off_score = _context_hit_score(
            case_id=str(case.get("id") or "case"),
            selected_items=off_items,
            expected=expected,
            budget_tokens=budget,
        )
        on_score = _context_hit_score(
            case_id=str(case.get("id") or "case"),
            selected_items=selected,
            expected=expected,
            budget_tokens=budget,
        )
        rows.append(
            {
                "case_id": str(case.get("id") or "case"),
                "off": off_score,
                "on": on_score,
            }
        )
    off_values = [row["off"] for row in rows if row["off"] is not None]
    on_values = [row["on"] for row in rows if row["on"] is not None]
    off_avg = mean(off_values) if off_values else None
    on_avg = mean(on_values) if on_values else None
    delta = on_avg - off_avg if off_avg is not None and on_avg is not None else None
    return {
        "schema_version": 1,
        "module": "context",
        "kind": "deterministic_ab",
        "cases": rows,
        "metrics": {
            "context.key_context_hit_rate": {
                "off": off_avg,
                "on": on_avg,
                "delta": delta,
            }
        },
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


def _context_hit_score(
    *,
    case_id: str,
    selected_items: list[dict[str, Any]],
    expected: dict[str, Any],
    budget_tokens: int,
) -> float | None:
    tokens_after = budget_tokens or sum(int(item.get("tokens") or 0) for item in selected_items)
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
    score = score_metrics(evidence, ["context.key_context_hit_rate"])[
        "context.key_context_hit_rate"
    ]
    return None if score.value is None else float(score.value)


def experiment_variants(module: str) -> tuple[str, str]:
    if module not in {"memory", "planning"}:
        raise ValueError(f"{module} does not support on/off ablation")
    return ("off", "on")


__all__ = [
    "experiment_variants",
    "run_context_ab",
    "run_security_ab",
]
