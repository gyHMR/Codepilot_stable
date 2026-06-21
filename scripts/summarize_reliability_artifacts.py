from __future__ import annotations

"""Summarize reliability evaluation artifacts into resume-ready metrics.

The script reads an evaluation artifact directory produced by
``python -m codepilot.evaluation`` or ``scripts/run_reliability_eval.py`` and
aggregates the eight metrics used for the Codepilot reliability benchmark:

- Context: current request preservation rate, average compression ratio
- Memory: retrieval hit rate, average repeated read count
- Planning: evidence coverage rate, false completion rate
- Security: dangerous tool block rate, mutation-after-denial rate
"""

import argparse
import json
from pathlib import Path
from typing import Any


Metric = dict[str, Any]


def summarize_eval(eval_root: str | Path) -> dict[str, Any]:
    """Build a nested reliability metric summary from an eval artifact root."""

    root = Path(eval_root)
    cases = [_load_case(root, case_dir) for case_dir in sorted((root / "cases").glob("*"))]
    cases = [case for case in cases if case["case_id"]]

    context_reports = _reports_for_domain(cases, "context")
    memory_cases = _cases_for_domain(cases, "memory")
    memory_reports = _reports_for_domain(cases, "memory")
    planning_cases = _cases_for_domain(cases, "planning")
    planning_reports = _reports_for_domain(cases, "planning")
    security_reports = _reports_for_domain(cases, "security")

    context_preserved = [
        _dict(report.get("context")).get("current_request_preserved")
        for report in context_reports
        if _dict(report.get("context")).get("current_request_preserved") is not None
    ]
    compression_ratios = [
        float(value)
        for report in context_reports
        if isinstance(
            value := _dict(report.get("context")).get("average_compression_ratio"),
            (int, float),
        )
        and not isinstance(value, bool)
    ]

    memory_hits = sum(1 for case in memory_cases if _memory_case_hit(case))
    memory_read_calls = [
        int(_dict(report.get("memory")).get("read_calls", 0) or 0)
        for report in memory_reports
    ]

    evidence_refs = sum(
        int(_dict(report.get("task")).get("evidence_ref_count", 0) or 0)
        for report in planning_reports
    )
    completed_steps = sum(
        int(_dict(report.get("task")).get("completed_steps", 0) or 0)
        for report in planning_reports
    )
    false_completion_count = sum(
        1 for case in planning_cases if _has_false_completion_violation(case)
    )

    dangerous_attempts = sum(
        sum(
            int(count)
            for count in _dict(_dict(report.get("security")).get("tool_status_counts")).values()
            if isinstance(count, int) and not isinstance(count, bool)
        )
        for report in security_reports
    )
    blocked_attempts = sum(
        int(_dict(report.get("security")).get("denied_or_approval_count", 0) or 0)
        for report in security_reports
    )
    mutation_after_denial = sum(
        int(_dict(report.get("security")).get("mutation_after_denial_count", 0) or 0)
        for report in security_reports
    )

    metrics = {
        "context": {
            "current_request_preservation_rate": _ratio_metric(
                "Current Request Preservation Rate",
                sum(item is True for item in context_preserved),
                len(context_preserved),
                "Higher is better. Measures whether context preparation preserved the active user request.",
            ),
            "average_compression_ratio": _average_metric(
                "Average Compression Ratio",
                compression_ratios,
                "Higher means more context was removed before prompting, as long as preservation stays high.",
            ),
        },
        "memory": {
            "memory_retrieval_hit_rate": _ratio_metric(
                "Memory Retrieval Hit Rate",
                memory_hits,
                len(memory_cases),
                "Higher is better. Counts memory benchmark cases whose memory assertion passed.",
            ),
            "average_repeated_read_count": _average_from_total_metric(
                "Average Repeated Read Count",
                sum(memory_read_calls),
                len(memory_read_calls),
                "Lower is better. Uses audit memory.read_calls as the repeated-read proxy.",
            ),
        },
        "planning": {
            "evidence_coverage_rate": _ratio_metric(
                "Evidence Coverage Rate",
                evidence_refs,
                completed_steps,
                "Higher is better. Measures evidence references per completed task step.",
            ),
            "false_completion_rate": _ratio_metric(
                "False Completion Rate",
                false_completion_count,
                len(planning_cases),
                "Lower is better. Counts planning cases whose false-completion guard failed.",
            ),
        },
        "security": {
            "dangerous_tool_block_rate": _ratio_metric(
                "Dangerous Tool Block Rate",
                blocked_attempts,
                dangerous_attempts,
                "Higher is better. Counts denied or approval-required tool attempts among security runs.",
            ),
            "mutation_after_denial_rate": _ratio_metric(
                "Mutation After Denial Rate",
                mutation_after_denial,
                blocked_attempts,
                "Lower is better. Measures side effects after denied/approval-required operations.",
            ),
        },
    }

    return {
        "eval_root": str(root),
        "case_count": len(cases),
        "domains": {
            domain: len(_cases_for_domain(cases, domain))
            for domain in ("context", "memory", "planning", "security")
        },
        "metrics": metrics,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a compact Markdown report for screenshots and README snippets."""

    lines = [
        "# Codepilot Reliability Metrics",
        "",
        f"- Artifact root: `{summary['eval_root']}`",
        f"- Cases: {summary['case_count']}",
        "",
        "| Module | Metric | Value | Evidence |",
        "|---|---|---:|---|",
    ]
    for module, metrics in summary["metrics"].items():
        for metric in metrics.values():
            lines.append(
                "| "
                f"{module} | {metric['label']} | {metric['display']} | "
                f"{metric['evidence']} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_summary_files(summary: dict[str, Any], eval_root: str | Path) -> None:
    root = Path(eval_root)
    (root / "reliability-metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "reliability-metrics.md").write_text(
        render_markdown(summary),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize Codepilot reliability eval artifacts.",
    )
    parser.add_argument("eval_root", type=Path)
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the summary only; do not write reliability-metrics files.",
    )
    args = parser.parse_args(argv)

    summary = summarize_eval(args.eval_root)
    if not args.no_write:
        write_summary_files(summary, args.eval_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _load_case(root: Path, case_dir: Path) -> dict[str, Any]:
    definition = _dict(_read_json(case_dir / "definition.json").get("definition"))
    assertion_results = _list_of_dicts(
        _read_json(case_dir / "assertion-results.json").get("results")
    )
    case_id = str(definition.get("id") or case_dir.name)
    return {
        "case_id": case_id,
        "domain": str(definition.get("domain") or _domain_from_case_id(case_id)),
        "definition": definition,
        "assertion_results": assertion_results,
        "reports": _load_run_reports(root, case_id),
    }


def _load_run_reports(root: Path, case_id: str) -> list[dict[str, Any]]:
    runs_dir = root / "workspaces" / case_id / ".codepilot" / "runs"
    if not runs_dir.is_dir():
        return []
    return [
        report
        for report in (
            _read_json(path)
            for path in sorted(runs_dir.glob("*/report.json"))
        )
        if report
    ]


def _memory_case_hit(case: dict[str, Any]) -> bool:
    assertions = [
        item for item in _list_of_dicts(case.get("assertion_results"))
        if item.get("name") == "memory" or item.get("dimension") == "memory"
    ]
    if assertions:
        return any(item.get("status") == "passed" for item in assertions)
    return any(
        int(_dict(report.get("memory")).get("retrieval_count", 0) or 0) > 0
        for report in _list_of_dicts(case.get("reports"))
    )


def _has_false_completion_violation(case: dict[str, Any]) -> bool:
    for result in _list_of_dicts(case.get("assertion_results")):
        expected = _dict(result.get("expected"))
        if expected.get("false_completion") is False and result.get("status") != "passed":
            return True
    return False


def _reports_for_domain(cases: list[dict[str, Any]], domain: str) -> list[dict[str, Any]]:
    return [
        report
        for case in _cases_for_domain(cases, domain)
        for report in _list_of_dicts(case.get("reports"))
    ]


def _cases_for_domain(cases: list[dict[str, Any]], domain: str) -> list[dict[str, Any]]:
    return [case for case in cases if case.get("domain") == domain]


def _ratio_metric(
    label: str,
    numerator: int,
    denominator: int,
    description: str,
) -> Metric:
    value = numerator / denominator if denominator else None
    return {
        "label": label,
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
        "display": _display_percent(value),
        "evidence": f"{numerator}/{denominator}",
        "description": description,
    }


def _average_metric(label: str, values: list[float], description: str) -> Metric:
    total = sum(values)
    return _average_from_total_metric(label, total, len(values), description)


def _average_from_total_metric(
    label: str,
    total: float,
    count: int,
    description: str,
) -> Metric:
    value = total / count if count else None
    return {
        "label": label,
        "value": value,
        "numerator": total,
        "denominator": count,
        "display": _display_number(value),
        "evidence": f"{total:g}/{count}",
        "description": description,
    }


def _display_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def _display_number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3g}"


def _domain_from_case_id(case_id: str) -> str:
    return case_id.split("-", 1)[0] if "-" in case_id else "unknown"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


if __name__ == "__main__":
    raise SystemExit(main())
