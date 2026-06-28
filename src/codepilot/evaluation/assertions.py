from __future__ import annotations

"""断言分发和多维度结果聚合。"""

from collections import defaultdict
from typing import Any, Iterable

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
    """执行断言列表：按类型分发到 outcome 或 audit 断言引擎。"""
    evidence.changes = compute_workspace_changes(
        evidence.workspace,
        evidence.baseline,
    )
    results = []
    for spec in specs:
        try:
            if spec.type in {"command", "file", "diff"}:
                result = run_outcome_assertion(spec, evidence)
            elif spec.type == "metric":
                result = AssertionResult(
                    name=spec.type,
                    dimension=spec.dimension,
                    status="skipped",
                    summary="Metric assertions run after metrics are calculated",
                    expected=spec.options,
                    required=spec.required,
                    role=spec.role,
                )
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
                role=spec.role,
            )
        results.append(result)
    return results


def run_metric_assertions(
    specs: Iterable[AssertionSpec],
    metrics: dict[str, Any],
) -> list[AssertionResult]:
    """执行指标阈值断言。"""

    results = []
    for spec in specs:
        try:
            results.append(_assert_metric(spec, metrics))
        except Exception as exc:
            results.append(
                AssertionResult(
                    name=spec.type,
                    dimension=spec.dimension,
                    status="error",
                    summary=f"Assertion raised {type(exc).__name__}: {exc}",
                    expected=spec.options,
                    actual={"error_kind": type(exc).__name__},
                    required=spec.required,
                    role=spec.role,
                )
            )
    return results


def build_dimension_results(
    assertion_results: list[AssertionResult],
) -> list[DimensionResult]:
    """将断言结果按维度聚合为 DimensionResult 列表。"""
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
    required = [
        result
        for result in assertion_results
        if result.required and result.role != "diagnostic"
    ]
    return bool(required) and all(
        result.status == "passed" for result in required
    )


def _assert_metric(
    spec: AssertionSpec,
    metrics: dict[str, Any],
) -> AssertionResult:
    metric_name = str(spec.options["metric"])
    operator = str(spec.options.get("op", ">="))
    expected_value = float(spec.options["value"])
    metric = metrics.get(metric_name)
    actual_value, display = _metric_value(metric)
    expected = {
        "metric": metric_name,
        "op": operator,
        "value": expected_value,
    }
    if bool(spec.options.get("allow_na", False)):
        expected["allow_na"] = True
    actual = {
        "metric": metric_name,
        "value": actual_value,
        "display": display,
    }
    if actual_value is None:
        if bool(spec.options.get("allow_na", False)):
            return AssertionResult(
                name=spec.type,
                dimension=spec.dimension,
                status="passed",
                summary=f"Metric is unavailable and allowed: {metric_name}",
                expected=expected,
                actual=actual,
                evidence_refs=[f"metric:{metric_name}"],
                required=spec.required,
                role=spec.role,
            )
        return AssertionResult(
            name=spec.type,
            dimension=spec.dimension,
            status="failed",
            summary=f"Metric is unavailable: {metric_name}",
            expected=expected,
            actual=actual,
            evidence_refs=[f"metric:{metric_name}"],
            required=spec.required,
            role=spec.role,
        )
    passed = _compare(float(actual_value), operator, expected_value)
    summary = (
        f"{metric_name} {operator} {expected_value:g}"
        if passed
        else (
            f"{metric_name} expected {operator} {expected_value:g}, "
            f"got {actual_value:g}"
        )
    )
    return AssertionResult(
        name=spec.type,
        dimension=spec.dimension,
        status="passed" if passed else "failed",
        summary=summary,
        expected=expected,
        actual=actual,
        evidence_refs=[f"metric:{metric_name}"],
        required=spec.required,
        role=spec.role,
    )


def _metric_value(metric: Any) -> tuple[float | int | None, str | None]:
    if isinstance(metric, dict):
        value = metric.get("value")
        display = metric.get("display")
    else:
        value = metric
        display = str(metric) if metric is not None else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, display if isinstance(display, str) else None
    return value, display if isinstance(display, str) else str(value)


def _compare(actual: float, operator: str, expected: float) -> bool:
    if operator == ">=":
        return actual >= expected
    if operator == ">":
        return actual > expected
    if operator == "<=":
        return actual <= expected
    if operator == "<":
        return actual < expected
    if operator == "==":
        return actual == expected
    if operator == "!=":
        return actual != expected
    raise ValueError(f"Unsupported metric operator: {operator}")


__all__ = [
    "build_dimension_results",
    "failure_categories",
    "required_assertions_passed",
    "run_metric_assertions",
    "run_assertions",
]
