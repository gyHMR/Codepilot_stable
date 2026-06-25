from __future__ import annotations

"""评估套件的聚合统计和人类可读报告生成。"""

from collections import Counter, defaultdict
from typing import Any

from .types import EvalResult


def build_suite_summary(results: list[EvalResult]) -> dict[str, Any]:
    """构建评估套件摘要：统计通过率、维度分布、失败分类和指标。"""
    overall = Counter(result.overall for result in results)
    dimensions: dict[str, Counter[str]] = defaultdict(Counter)
    failures: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    metric_values: dict[str, list[float]] = defaultdict(list)
    for result in results:
        failures.update(result.failure_categories)
        for key, value in result.metrics.items():
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] += value
            elif isinstance(value, dict):
                metric_value = value.get("value")
                if isinstance(metric_value, (int, float)) and not isinstance(
                    metric_value,
                    bool,
                ):
                    metric_values[key].append(float(metric_value))
        for dimension in result.dimensions:
            dimensions[dimension.dimension][dimension.status] += 1
    total = len(results)
    passed = overall.get("passed", 0)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "overall": dict(sorted(overall.items())),
        "dimensions": {
            name: dict(sorted(statuses.items()))
            for name, statuses in sorted(dimensions.items())
        },
        "failure_categories": dict(sorted(failures.items())),
        "metrics": dict(sorted(totals.items())),
        "metric_averages": {
            name: sum(values) / len(values)
            for name, values in sorted(metric_values.items())
            if values
        },
        "duration_ms": sum(result.duration_ms or 0 for result in results),
    }


def render_suite_markdown(
    results: list[EvalResult],
    summary: dict[str, Any],
) -> str:
    """将评估结果渲染为 Markdown 格式的报告。"""
    lines = [
        "# Codepilot Evaluation Report",
        "",
        "## Summary",
        "",
        f"- Cases: {summary.get('total', 0)}",
        f"- Passed: {summary.get('passed', 0)}",
        f"- Pass rate: {float(summary.get('pass_rate', 0.0)):.1%}",
        f"- Duration: {summary.get('duration_ms', 0)} ms",
        "",
        "## Dimensions",
        "",
        "| Dimension | Passed | Failed | Error | N/A |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, statuses in summary.get("dimensions", {}).items():
        lines.append(
            "| "
            f"{name} | {statuses.get('passed', 0)} | "
            f"{statuses.get('failed', 0)} | "
            f"{statuses.get('error', 0)} | "
            f"{statuses.get('not_applicable', 0)} |"
        )
    metric_averages = summary.get("metric_averages", {})
    if metric_averages:
        lines.extend(
            [
                "",
                "## Metrics",
                "",
                "| Metric | Average |",
                "|---|---:|",
            ]
        )
        for name, value in metric_averages.items():
            lines.append(
                f"| {_metric_label(name)} | {_metric_display(name, value)} |"
            )
    lines.extend(["", "## Cases", ""])
    for result in results:
        lines.extend(
            [
                f"### {result.case_id}",
                "",
                f"- Overall: `{result.overall}`",
                f"- Runs: {', '.join(result.run_ids) or '(none)'}",
            ]
        )
        if result.failure_categories:
            lines.append(
                "- Failures: "
                + ", ".join(f"`{item}`" for item in result.failure_categories)
            )
        if result.error:
            lines.append(f"- Execution error: `{result.error}`")
        lines.append("")
        for dimension in result.dimensions:
            lines.append(
                f"- {dimension.dimension}: `{dimension.status}` — "
                f"{dimension.summary}"
            )
        lines.append("")
    if summary.get("failure_categories"):
        lines.extend(["## Failure Categories", ""])
        for name, count in summary["failure_categories"].items():
            lines.append(f"- `{name}`: {count}")
        lines.append("")
    return "\n".join(lines)


def _metric_label(name: str) -> str:
    labels = {
        "context.key_context_hit_rate": "Key Context Hit Rate",
        "context.token_efficiency": "Token Efficiency",
        "context.stale_context_rate": "Stale Context Rate",
        "memory.memory_retrieval_hit_rate": "Memory Retrieval Hit Rate",
        "memory.redundant_read_count": "Redundant Read Count",
        "memory.failed_attempt_recurrence_rate": "Failed Attempt Recurrence Rate",
        "planning.step_completion_rate": "Step Completion Rate",
        "planning.evidence_coverage_rate": "Evidence Coverage Rate",
        "planning.false_completion_rate": "False Completion Rate",
        "planning.replan_success_rate": "Replan Success Rate",
        "planning.repair_replan_success_rate": "Repair/Replan Success Rate",
        "planning.invalid_tool_call_count": "Invalid Tool Call Count",
        "planning.invalid_tool_call_rate": "Invalid Tool Call Rate",
        "security.dangerous_tool_block_rate": "Dangerous Tool Block Rate",
        "security.mutation_after_denial_rate": "Mutation After Denial Rate",
        "security.benign_tool_pass_rate": "Benign Tool Pass Rate",
    }
    return labels.get(name, name)


def _metric_display(name: str, value: float) -> str:
    if name.endswith("_count"):
        return f"{value:.2f}"
    return f"{value:.1%}"


__all__ = ["build_suite_summary", "render_suite_markdown"]
