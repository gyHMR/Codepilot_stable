from __future__ import annotations

# 新手导读：reports.py 把评估结果渲染成可读报告。
# 关注点：报告服务于简历/面试展示和回归分析。

"""Human-readable reports for evaluation v2."""

from collections import defaultdict
from statistics import mean
from typing import Any

from .schema import EvalResult


def build_summary(results: list[EvalResult]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    metrics: dict[str, list[float]] = defaultdict(list)
    modules: dict[str, dict[str, Any]] = {}
    for result in results:
        module = result.module
        item = modules.setdefault(module, {"total": 0, "passed": 0, "metrics": {}})
        item["total"] += 1
        item["passed"] += 1 if result.passed else 0
        for name, score in result.metrics.items():
            if score.value is not None:
                metrics[name].append(float(score.value))
                item["metrics"].setdefault(name, []).append(float(score.value))
    metric_summary = {
        name: {
            "avg": mean(values),
            "count": len(values),
        }
        for name, values in sorted(metrics.items())
    }
    for module in modules.values():
        module["pass_rate"] = (
            module["passed"] / module["total"] if module["total"] else None
        )
        module["metrics"] = {
            name: {"avg": mean(values), "count": len(values)}
            for name, values in sorted(module["metrics"].items())
        }
    return {
        "schema_version": 1,
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        "pass_rate": passed / total if total else None,
        "modules": modules,
        "metrics": metric_summary,
    }


def render_markdown(results: list[EvalResult], summary: dict[str, Any]) -> str:
    lines = [
        "# Codepilot Evaluation Report",
        "",
        f"- Cases: {summary['passed_cases']}/{summary['total_cases']} passed",
        f"- Pass rate: {_percent(summary.get('pass_rate'))}",
        "",
        "## Metrics",
        "",
    ]
    if not summary.get("metrics"):
        lines.append("_No numeric metrics._")
    else:
        lines.extend(["| Metric | Average | Cases |", "| --- | ---: | ---: |"])
        for name, info in summary["metrics"].items():
            lines.append(f"| `{name}` | {_percent(info['avg'])} | {info['count']} |")
    lines.extend(["", "## Cases", ""])
    if not results:
        lines.append("_No cases._")
    else:
        lines.extend(["| Case | Module | Result | Main metrics |", "| --- | --- | --- | --- |"])
        for result in results:
            metric_text = ", ".join(
                f"{name}={score.display}"
                for name, score in sorted(result.metrics.items())
            )
            status = "passed" if result.passed else "failed"
            lines.append(
                f"| `{result.case_id}` | {result.module} | {status} | {metric_text} |"
            )
    return "\n".join(lines) + "\n"


def render_comparison_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# Codepilot Experiment Comparison",
        "",
        f"- Module: {comparison.get('module', '')}",
        "",
        "| Metric | Off | On | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, values in comparison.get("metrics", {}).items():
        off = values.get("off")
        on = values.get("on")
        delta = values.get("delta")
        lines.append(
            f"| `{name}` | {_percent(off)} | {_percent(on)} | {_percent(delta)} |"
        )
    return "\n".join(lines) + "\n"


def _percent(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.1%}"


__all__ = ["build_summary", "render_comparison_markdown", "render_markdown"]
