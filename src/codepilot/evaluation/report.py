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
    for result in results:
        failures.update(result.failure_categories)
        for key, value in result.metrics.items():
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] += value
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


__all__ = ["build_suite_summary", "render_suite_markdown"]
