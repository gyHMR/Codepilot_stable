from __future__ import annotations

"""简单的模块 on/off 消融实验。"""

import json
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from .types import EvalRunOptions


_TOGGLES = {
    "context": "context_governance_enabled",
    "memory": "memory_enabled",
    "planning": "task_control_enabled",
}


def experiment_variants(module: str) -> list[tuple[str, dict[str, bool]]]:
    """返回一个模块的 off/on 两个固定变体。"""

    try:
        toggle = _TOGGLES[module]
    except KeyError as exc:
        raise ValueError(
            "Experiment module must be context, memory, or planning"
        ) from exc
    return [
        ("off", {toggle: False}),
        ("on", {toggle: True}),
    ]


def build_experiment_comparison(
    module: str,
    runs: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """对 off/on 多次 Suite Summary 取简单平均并计算差值。"""

    variant_metrics = {
        name: _average_summaries(summaries)
        for name, summaries in runs.items()
    }
    metric_names = sorted(
        {
            metric
            for variant in variant_metrics.values()
            for metric in variant["metrics"]
        }
    )
    metrics = {}
    for metric in metric_names:
        off = variant_metrics.get("off", {}).get("metrics", {}).get(metric)
        on = variant_metrics.get("on", {}).get("metrics", {}).get(metric)
        metrics[metric] = {
            "off": off,
            "on": on,
            "change": (
                on - off
                if isinstance(on, (int, float))
                and isinstance(off, (int, float))
                else None
            ),
        }
    if module == "memory":
        read_counts = metrics.get("memory.redundant_read_count", {})
        off_reads = read_counts.get("off")
        on_reads = read_counts.get("on")
        reduction = (
            (off_reads - on_reads) / off_reads
            if isinstance(off_reads, (int, float))
            and isinstance(on_reads, (int, float))
            and off_reads > 0
            else None
        )
        metrics["memory.redundant_read_reduction_rate"] = {
            "off": 0.0 if reduction is not None else None,
            "on": reduction,
            "change": reduction,
        }
    if module == "planning":
        invalid_counts = metrics.get("planning.invalid_tool_call_count", {})
        off_invalid = invalid_counts.get("off")
        on_invalid = invalid_counts.get("on")
        reduction = (
            off_invalid - on_invalid
            if isinstance(off_invalid, (int, float))
            and isinstance(on_invalid, (int, float))
            else None
        )
        metrics["planning.invalid_tool_call_reduction_count"] = {
            "off": 0.0 if reduction is not None else None,
            "on": reduction,
            "change": reduction,
        }
    off_pass = variant_metrics.get("off", {}).get("pass_rate")
    on_pass = variant_metrics.get("on", {}).get("pass_rate")
    return {
        "module": module,
        "repeat": max((len(items) for items in runs.values()), default=0),
        "metrics": metrics,
        "pass_rate": {
            "off": off_pass,
            "on": on_pass,
            "change": (
                on_pass - off_pass
                if isinstance(on_pass, (int, float))
                and isinstance(off_pass, (int, float))
                else None
            ),
        },
    }


async def run_experiment(
    service: Any,
    suite_path: str | Path,
    options: EvalRunOptions,
    *,
    module: str,
    repeat: int = 3,
) -> dict[str, Any]:
    """运行简单 on/off 实验并保存 JSON/Markdown。"""

    if repeat <= 0:
        raise ValueError("repeat must be positive")
    experiment_id = options.eval_id or f"experiment_{uuid.uuid4().hex[:12]}"
    runs: dict[str, list[dict[str, Any]]] = {"off": [], "on": []}
    run_artifacts: dict[str, list[str]] = {"off": [], "on": []}
    for variant, overrides in experiment_variants(module):
        for index in range(repeat):
            run_options = replace(
                options,
                eval_id=f"{experiment_id}_{variant}_{index + 1}",
                benchmark_name=f"{options.benchmark_name or module}-{variant}",
                runtime_overrides=overrides,
            )
            result = await service.run_suite(Path(suite_path), run_options)
            runs[variant].append(result.summary)
            run_artifacts[variant].append(result.artifact_dir)

    comparison = build_experiment_comparison(module, runs)
    comparison["runs"] = run_artifacts
    root = Path(options.artifact_root) / experiment_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "experiment.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "report.md").write_text(
        render_experiment_markdown(comparison),
        encoding="utf-8",
    )
    comparison["artifact_dir"] = str(root)
    return comparison


def render_experiment_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        f"# {str(comparison.get('module', '')).title()} Experiment",
        "",
        f"- Repeat: {comparison.get('repeat', 0)}",
        "",
        "| Metric | Off | On | Change |",
        "|---|---:|---:|---:|",
    ]
    for name, values in comparison.get("metrics", {}).items():
        lines.append(
            f"| {name} | {_display(values.get('off'))} | "
            f"{_display(values.get('on'))} | "
            f"{_display(values.get('change'), signed=True)} |"
        )
    pass_rate = comparison.get("pass_rate", {})
    lines.append(
        f"| task.pass_rate | {_display(pass_rate.get('off'))} | "
        f"{_display(pass_rate.get('on'))} | "
        f"{_display(pass_rate.get('change'), signed=True)} |"
    )
    lines.append("")
    return "\n".join(lines)


def _average_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = {
        name
        for summary in summaries
        for name in summary.get("metric_averages", {})
    }
    return {
        "pass_rate": _average(
            [
                float(summary["pass_rate"])
                for summary in summaries
                if isinstance(summary.get("pass_rate"), (int, float))
            ]
        ),
        "metrics": {
            name: _average(
                [
                    float(value)
                    for summary in summaries
                    if isinstance(
                        value := summary.get("metric_averages", {}).get(name),
                        (int, float),
                    )
                ]
            )
            for name in metric_names
        },
    }


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _display(value: Any, *, signed: bool = False) -> str:
    if not isinstance(value, (int, float)):
        return "N/A"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


__all__ = [
    "build_experiment_comparison",
    "experiment_variants",
    "render_experiment_markdown",
    "run_experiment",
]
