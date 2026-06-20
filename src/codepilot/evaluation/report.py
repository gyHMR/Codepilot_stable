from __future__ import annotations

"""Eval suite aggregation."""

from collections import Counter
from typing import Any

from .types import EvalResult


def build_suite_summary(results: list[EvalResult]) -> dict[str, Any]:
    verdicts = Counter(result.verdict for result in results)
    verifier_statuses = Counter(
        verifier.status
        for result in results
        for verifier in result.verifier_results
    )
    passed = verdicts.get("passed", 0)
    total = len(results)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "verdicts": dict(sorted(verdicts.items())),
        "verifier_statuses": dict(sorted(verifier_statuses.items())),
        "run_count": sum(len(result.run_ids) for result in results),
        "duration_ms": sum(
            result.duration_ms or 0 for result in results
        ),
    }

