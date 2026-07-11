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
    if comparison.get("kind") == "offline_context_selection_benchmark":
        return _render_context_selection_markdown(comparison)
    if comparison.get("kind") == "offline_memory_retrieval_benchmark":
        return _render_memory_retrieval_markdown(comparison)
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


def _render_memory_retrieval_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# Codepilot Memory Retrieval Benchmark",
        "",
        "- Kind: offline memory retrieval benchmark",
        f"- Cases: {comparison.get('case_count', len(comparison.get('cases', [])))}",
        f"- Corpus size: {comparison.get('corpus_size', 'N/A')}",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Average | Cases | Direction |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, info in comparison.get("metrics", {}).items():
        item = _dict(info)
        direction = "higher is better" if item.get("higher_is_better") else "lower is better"
        lines.append(
            f"| `{name}` | {_percent(item.get('avg'))} | {item.get('count', 0)} | {direction} |"
        )
    lines.extend(["", "## Cases", ""])
    lines.append(
        "| Case | Recall@3 | Precision@3 | MRR | Forbidden | Stale | Retrieved IDs |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in comparison.get("cases", []):
        case = _dict(row)
        metrics = _dict(case.get("metrics"))
        retrieved = ", ".join(
            str(_dict(item).get("id") or "")
            for item in case.get("retrieved", [])
            if isinstance(item, dict)
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{case.get('case_id', '')}`",
                    _percent(metrics.get("memory.recall@3")),
                    _percent(metrics.get("memory.precision@3")),
                    _percent(metrics.get("memory.mrr")),
                    _percent(metrics.get("memory.forbidden_retrieval_rate")),
                    _percent(metrics.get("memory.stale_retrieval_rate")),
                    retrieved or "(none)",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _render_context_selection_markdown(comparison: dict[str, Any]) -> str:
    strategies = [
        str(item)
        for item in comparison.get("strategies", [])
        if str(item).strip()
    ]
    lines = [
        "# Codepilot Context Selection Benchmark",
        "",
        "- Kind: offline context selection benchmark",
        f"- Cases: {len(comparison.get('cases', []))}",
        "",
        "## Aggregate Metrics",
        "",
    ]
    header = [
        "Metric",
        *strategies,
        "Lift vs order_first",
        "Lift vs best baseline",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---", *[":---:" for _ in strategies], "---:", "---:"]) + " |")
    for name, values in comparison.get("metrics", {}).items():
        row = [
            f"`{name}`",
            *[_percent(_dict(values).get(strategy)) for strategy in strategies],
            _percent(_dict(values).get("lift_vs_order_first")),
            _percent(_dict(values).get("lift_vs_best_baseline")),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(["", "## Cases", ""])
    lines.append("| Case | order_first | codepilot_policy | oracle | Codepilot selected |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for row in comparison.get("cases", []):
        row_dict = _dict(row)
        strategies_payload = _dict(row_dict.get("strategies"))
        metric = "context.key_context_hit_rate"
        def hit_rate(strategy: str) -> Any:
            return _dict(_dict(strategies_payload.get(strategy)).get("metrics")).get(metric)

        selected = ", ".join(
            str(item)
            for item in _dict(strategies_payload.get("codepilot_policy")).get("selected_ids", [])
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row_dict.get('case_id', '')}`",
                    _percent(hit_rate("order_first")),
                    _percent(hit_rate("codepilot_policy")),
                    _percent(hit_rate("oracle")),
                    selected or "(none)",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _percent(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.1%}"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


__all__ = ["build_summary", "render_comparison_markdown", "render_markdown"]
