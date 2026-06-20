from __future__ import annotations

"""Assertion dispatch and multi-dimensional result aggregation."""

from collections import defaultdict
from typing import Iterable

from .audit_assertions import run_audit_assertion
from .outcome_assertions import (
    compute_workspace_changes,
    run_outcome_assertion,
)
from .types import (
    AssertionResult,
    AssertionSpec,
    DimensionResult,
    EvalDimension,
    EvalEvidence,
)


def run_assertions(
    specs: Iterable[AssertionSpec],
    evidence: EvalEvidence,
) -> list[AssertionResult]:
    evidence.changes = compute_workspace_changes(
        evidence.workspace,
        evidence.baseline,
    )
    results = []
    for spec in specs:
        try:
            if spec.type in {"command", "file", "diff"}:
                result = run_outcome_assertion(spec, evidence)
            else:
                result = run_audit_assertion(spec, evidence)
        except Exception as exc:
            result = AssertionResult(
                name=spec.type,
                dimension=spec.dimension,
                status="error",
                summary=f"Assertion raised {type(exc).__name__}: {exc}",
                expected=spec.options,
                actual={"error_kind": type(exc).__name__},
                required=spec.required,
            )
        results.append(result)
    return results


def build_dimension_results(
    assertion_results: list[AssertionResult],
) -> list[DimensionResult]:
    grouped: dict[EvalDimension, list[AssertionResult]] = defaultdict(list)
    for result in assertion_results:
        grouped[result.dimension].append(result)
    dimensions = []
    for dimension, results in grouped.items():
        statuses = {result.status for result in results}
        if "error" in statuses:
            status = "error"
        elif "failed" in statuses:
            status = "failed"
        elif statuses == {"skipped"}:
            status = "not_applicable"
        else:
            status = "passed"
        passed = sum(result.status == "passed" for result in results)
        dimensions.append(
            DimensionResult(
                dimension=dimension,
                status=status,  # type: ignore[arg-type]
                summary=f"{passed}/{len(results)} assertions passed",
                assertion_results=results,
                metrics={
                    "total": len(results),
                    "passed": passed,
                    "failed": sum(
                        result.status == "failed" for result in results
                    ),
                    "errors": sum(
                        result.status == "error" for result in results
                    ),
                },
            )
        )
    return sorted(dimensions, key=lambda item: item.dimension)


def failure_categories(
    assertion_results: list[AssertionResult],
) -> list[str]:
    return [
        f"{result.dimension}.{result.name}_{result.status}"
        for result in assertion_results
        if result.status in {"failed", "error"}
    ]


def required_assertions_passed(
    assertion_results: list[AssertionResult],
) -> bool:
    required = [result for result in assertion_results if result.required]
    return bool(required) and all(
        result.status == "passed" for result in required
    )


__all__ = [
    "build_dimension_results",
    "failure_categories",
    "required_assertions_passed",
    "run_assertions",
]
