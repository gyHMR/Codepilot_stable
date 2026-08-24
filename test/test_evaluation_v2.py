from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from codepilot.evaluation.cli import _load_ab_cases, _run_ab
from codepilot.evaluation.artifacts import EvaluationArtifacts
from codepilot.evaluation.evidence import (
    ContextEvidence,
    EvalEvidence,
    ToolCallEvidence,
)
from codepilot.evaluation.experiments import (
    aggregate_experiment_comparison,
    run_context_ab,
)
from codepilot.evaluation.loader import load_eval_case, load_eval_suite
from codepilot.evaluation.memory_retrieval import (
    load_memory_corpus,
    load_memory_retrieval_cases,
    run_memory_retrieval_benchmark,
)
from codepilot.evaluation.reports import (
    build_summary,
    render_comparison_markdown,
    render_markdown,
)
from codepilot.evaluation.runner import _load_persisted_trace
from codepilot.evaluation.schema import EvalCase, EvalResult, MetricScore
from codepilot.evaluation.scorers import score_metrics


def test_loader_accepts_new_task_schema(tmp_path: Path) -> None:
    case_path = tmp_path / "case.json"
    case_path.write_text(
        json.dumps(
            {
                "id": "planning-smoke",
                "module": "planning",
                "fixture": "mini",
                "type": "task",
                "prompt": "Fix the bug",
                "setup": [{"kind": "copy", "path": "src/app.py"}],
                "checks": [{"kind": "command", "command": "python -m pytest"}],
                "metrics": ["planning.step_completion_rate"],
                "context_profile": {"kind": "compression", "pressure": "tight"},
                "tags": ["smoke"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    case = load_eval_case(case_path)

    assert case.id == "planning-smoke"
    assert case.module == "planning"
    assert case.type == "task"
    assert case.prompt == "Fix the bug"
    assert case.steps == []
    assert case.metrics == ["planning.step_completion_rate"]
    assert case.context_profile["kind"] == "compression"


def test_score_metrics_reads_only_eval_evidence() -> None:
    evidence = EvalEvidence(
        case_id="case-1",
        module="context",
        task_passed=True,
        contexts=[
            ContextEvidence(
                selected_items=[
                    {
                        "id": "file:src/app.py",
                        "path": "src/app.py",
                        "tokens": 100,
                        "freshness": "fresh",
                    },
                    {
                        "id": "file:docs/legacy.md",
                        "path": "docs/legacy.md",
                        "tokens": 50,
                        "freshness": "stale",
                    },
                ],
                tokens_before=500,
                tokens_after=200,
            )
        ],
        tools=[
            ToolCallEvidence(
                tool_call_id="1",
                tool_name="write",
                status="denied",
                error_reason="read_only_permission_mode",
                workspace_changed=False,
            ),
            ToolCallEvidence(
                tool_call_id="2",
                tool_name="read",
                status="success",
                workspace_changed=False,
            ),
        ],
        expected={
            "key_context": ["src/app.py"],
            "dangerous_tools": ["write"],
            "benign_tools": ["read"],
        },
    )

    scores = score_metrics(
        evidence,
        [
            "task.pass_rate",
            "context.key_context_hit_rate",
            "context.token_efficiency",
            "context.compression_rate",
            "context.stale_context_rate",
            "security.dangerous_block_rate",
            "security.benign_pass_rate",
            "security.mutation_after_denial_rate",
        ],
    )

    assert scores["task.pass_rate"].value == 1.0
    assert scores["context.key_context_hit_rate"].value == 1.0
    assert scores["context.token_efficiency"].value == 0.5
    assert scores["context.compression_rate"].value == 0.6
    assert scores["context.stale_context_rate"].value == 0.5
    assert scores["security.dangerous_block_rate"].value == 1.0
    assert scores["security.benign_pass_rate"].value == 1.0
    assert scores["security.mutation_after_denial_rate"].value == 0.0


def test_context_compression_rate_uses_weighted_context_tokens() -> None:
    evidence = EvalEvidence(
        case_id="context-compression",
        module="context",
        task_passed=True,
        contexts=[
            ContextEvidence(tokens_before=1000, tokens_after=800),
            ContextEvidence(tokens_before=9000, tokens_after=4200),
        ],
    )

    scores = score_metrics(evidence, ["context.compression_rate"])

    score = scores["context.compression_rate"]
    assert score.value == 0.5
    assert score.numerator == 5000
    assert score.denominator == 10000


def test_context_compression_rate_prefers_section_reduction_tokens() -> None:
    evidence = EvalEvidence(
        case_id="context-compression",
        module="context",
        task_passed=True,
        contexts=[
            ContextEvidence(
                tokens_before=1000,
                tokens_after=1200,
                sections=[
                    {
                        "name": "working_set",
                        "estimated_tokens_before": 5000,
                        "estimated_tokens_after": 2000,
                    },
                    {
                        "name": "memory",
                        "estimated_tokens_before": 100,
                        "estimated_tokens_after": 120,
                    },
                ],
            )
        ],
    )

    scores = score_metrics(evidence, ["context.compression_rate"])

    score = scores["context.compression_rate"]
    assert score.value == 0.6
    assert score.numerator == 3000
    assert score.denominator == 5000


def test_context_compression_rate_ignores_expanded_fallback_contexts() -> None:
    evidence = EvalEvidence(
        case_id="context-expanded",
        module="context",
        task_passed=True,
        contexts=[
            ContextEvidence(tokens_before=1000, tokens_after=1200),
            ContextEvidence(tokens_before=1000, tokens_after=1000),
        ],
    )

    score = score_metrics(evidence, ["context.compression_rate"])["context.compression_rate"]

    assert score.value is None
    assert score.denominator == 0


def test_context_compression_quality_metrics_use_raw_and_compressed_variants() -> None:
    evidence = EvalEvidence(
        case_id="context-compression",
        module="context",
        task_passed=True,
        variant_passed={"raw": True, "compressed": True},
        variant_context_tokens={
            "raw": {"tokens_before": 10000, "tokens_after": 10000},
            "compressed": {"tokens_before": 10000, "tokens_after": 3500},
        },
        contexts=[
            ContextEvidence(
                tokens_before=10000,
                tokens_after=9000,
                sections=[
                    {
                        "name": "working_set",
                        "estimated_tokens_before": 1000,
                        "estimated_tokens_after": 900,
                    }
                ],
            )
        ],
    )

    scores = score_metrics(
        evidence,
        [
            "context.compression_rate",
            "context.raw_pass_rate",
            "context.compressed_pass_rate",
            "context.quality_retention_rate",
        ],
    )

    assert scores["context.compression_rate"].value == 0.65
    assert scores["context.raw_pass_rate"].value == 1.0
    assert scores["context.compressed_pass_rate"].value == 1.0
    assert scores["context.quality_retention_rate"].value == 1.0


def test_context_compression_benchmarks_have_pressure_profiles() -> None:
    cases = _context_compression_cases()

    assert len(cases) == 10
    assert sum(case.context_profile.get("pressure") == "tight" for case in cases) == 5
    assert sum(case.context_profile.get("pressure") == "critical" for case in cases) == 5
    for case in cases:
        assert "context.compression_rate" in case.metrics
        assert "context.quality_retention_rate" in case.metrics
        assert int(case.context_profile.get("history_groups") or 0) > 0
        assert int(case.context_profile.get("chars_per_group") or 0) > 0


def test_load_persisted_trace_prefers_workspace_run_artifact(tmp_path: Path) -> None:
    trace_dir = tmp_path / ".codepilot" / "runs" / "run-1"
    trace_dir.mkdir(parents=True)
    (trace_dir / "trace.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "session_id": "session-1",
                "status": "waiting_approval",
                "stop_reason": "approval_required",
                "contexts": [
                    {
                        "context_id": "ctx-1",
                        "tokens_before": 3000,
                        "tokens_after": 1200,
                        "sections": [
                            {
                                "name": "working_set",
                                "estimated_tokens_before": 2000,
                                "estimated_tokens_after": 800,
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    trace = _load_persisted_trace(tmp_path, "run-1")

    assert trace is not None
    assert trace.run_id == "run-1"
    assert trace.contexts[0].tokens_before == 3000
    assert trace.contexts[0].sections[0]["estimated_tokens_after"] == 800


def test_score_metrics_supports_recovery_and_failed_attempt_paths() -> None:
    evidence = EvalEvidence(
        case_id="case-recovery",
        module="planning",
        task_passed=True,
        run_ids=["run-1", "run-2"],
        tools=[
            ToolCallEvidence(
                tool_call_id="read-current",
                tool_name="read",
                status="success",
                affected_paths=["docs/api-contract-v2.md"],
            )
        ],
        expected={
            "recovery_after_abort": True,
            "failed_attempt_paths": ["docs/api-contract-v1.md"],
        },
    )

    scores = score_metrics(
        evidence,
        [
            "planning.abort_recovery_rate",
            "memory.failed_attempt_recurrence_rate",
        ],
    )

    assert scores["planning.abort_recovery_rate"].value == 1.0
    assert scores["memory.failed_attempt_recurrence_rate"].value == 0.0


def test_artifacts_and_report_use_fixed_v2_layout(tmp_path: Path) -> None:
    artifacts = EvaluationArtifacts(tmp_path, "eval-smoke")
    case = EvalCase(
        id="security-smoke",
        module="security",
        fixture="mini",
        type="task",
        prompt="Read then deny write",
        metrics=["security.dangerous_block_rate"],
    )
    evidence = EvalEvidence(case_id=case.id, module=case.module, task_passed=True)
    result = EvalResult(
        case_id=case.id,
        module=case.module,
        passed=True,
        metrics={
            "security.dangerous_block_rate": MetricScore(
                name="security.dangerous_block_rate",
                value=1.0,
                numerator=1,
                denominator=1,
            )
        },
    )

    artifacts.initialize("security", case_count=1)
    artifacts.write_case(case, result, evidence, workspace_diff="M src/app.py\n")
    summary = build_summary([result])
    artifacts.write_summary(summary, render_markdown([result], summary))

    root = tmp_path / "eval-smoke"
    assert (root / "manifest.json").is_file()
    assert (root / "summary.json").is_file()
    assert (root / "report.md").is_file()
    assert (root / "metrics.csv").is_file()
    assert (root / "cases.csv").is_file()
    assert (root / "cases/security-smoke/case.json").is_file()
    assert (root / "cases/security-smoke/evidence.json").is_file()
    assert (root / "cases/security-smoke/scores.json").is_file()

    report = (root / "report.md").read_text(encoding="utf-8")
    assert "security.dangerous_block_rate" in report
    assert "100.0%" in report


def test_context_ab_compares_naive_and_builder() -> None:
    comparison = run_context_ab(
        cases=[
            {
                "id": "ctx",
                "query": "Fix app API contract regression",
                "expected": {"gold_evidence": ["src/app.py"]},
                "candidates": [
                    {
                        "id": "docs/legacy-api.md",
                        "path": "docs/legacy-api.md",
                        "tokens": 100,
                        "freshness": "stale",
                    },
                    {
                        "id": "src/app.py",
                        "path": "src/app.py",
                        "tokens": 50,
                        "freshness": "fresh",
                        "trust": "observed",
                    },
                ],
                "oracle_selected": [
                    {"id": "src/app.py", "path": "src/app.py", "tokens": 50}
                ],
                "budget_tokens": 100,
            }
        ]
    )

    assert comparison["module"] == "context"
    assert comparison["kind"] == "offline_context_selection_benchmark"
    assert comparison["variants"]["off"] == "order_first"
    assert comparison["variants"]["on"] == "codepilot_policy"
    assert comparison["metrics"]["context.key_context_hit_rate"]["off"] == 0.0
    assert comparison["metrics"]["context.key_context_hit_rate"]["on"] == 1.0
    assert comparison["metrics"]["context.key_context_hit_rate"]["keyword_overlap"] == 1.0
    assert comparison["metrics"]["context.key_context_hit_rate"]["oracle"] == 1.0
    assert comparison["metrics"]["context.key_context_hit_rate"]["lift_vs_order_first"] == 1.0
    assert "context.stale_context_rate" in comparison["metrics"]
    case = comparison["cases"][0]
    assert case["strategies"]["codepilot_policy"]["selected_ids"] == ["src/app.py"]


def test_context_ab_defaults_to_benchmark_case_file() -> None:
    cases = _load_ab_cases(
        Namespace(module="context", cases=None)
    )

    assert len(cases) == 10
    assert cases[0]["id"].startswith("ab-")
    assert "oracle_selected" in cases[0]
    assert "on_selected" not in cases[0]


def test_memory_retrieval_benchmark_scores_ranked_corpus() -> None:
    cases = load_memory_retrieval_cases()
    corpus = load_memory_corpus()

    comparison = run_memory_retrieval_benchmark(cases[:3], corpus)

    assert comparison["module"] == "memory"
    assert comparison["kind"] == "offline_memory_retrieval_benchmark"
    assert comparison["case_count"] == 3
    assert comparison["corpus_size"] >= 20
    assert "memory.recall@3" in comparison["metrics"]
    assert "memory.precision@3" in comparison["metrics"]
    assert "memory.mrr" in comparison["metrics"]
    assert comparison["metrics"]["memory.recall@3"]["count"] == 3
    assert comparison["cases"][0]["retrieved"]
    report = render_comparison_markdown(comparison)
    assert "Memory Retrieval Benchmark" in report
    assert "memory.recall@3" in report


def test_memory_retrieval_benchmark_handles_no_relevant_cases() -> None:
    cases = [
        case
        for case in load_memory_retrieval_cases()
        if case.get("expected", {}).get("no_relevant")
    ]
    corpus = load_memory_corpus()

    comparison = run_memory_retrieval_benchmark(cases, corpus)

    assert len(cases) == 2
    assert comparison["metrics"]["memory.no_relevant_rejection_rate"]["count"] == 2
    for row in comparison["cases"]:
        assert row["metrics"]["memory.recall@3"] is None
        assert row["metrics"]["memory.no_relevant_rejection_rate"] in {0.0, 1.0}


def test_cli_ab_memory_writes_offline_retrieval_artifacts(tmp_path: Path) -> None:
    code = _run_ab(
        Namespace(
            module="memory",
            cases=None,
            corpus=Path("benchmarks/evaluation_memory/corpus/issue_tracker_memory.jsonl"),
            artifact_root=tmp_path,
            eval_id="memory-ranking",
        )
    )

    assert code == 0
    comparison_path = tmp_path / "memory-ranking" / "comparison.json"
    report_path = tmp_path / "memory-ranking" / "report.md"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert comparison["kind"] == "offline_memory_retrieval_benchmark"
    assert report_path.is_file()


def test_experiment_comparison_aggregates_repeat_summaries(tmp_path: Path) -> None:
    off_dir = tmp_path / "variants" / "off" / "repeat-1"
    on_dir = tmp_path / "variants" / "on" / "repeat-1"
    off_dir.mkdir(parents=True)
    on_dir.mkdir(parents=True)
    (off_dir / "summary.json").write_text(
        json.dumps(
            {
                "pass_rate": 0.5,
                "metrics": {
                    "memory.retrieval_hit_rate": {"avg": 0.25, "count": 2},
                    "tool.success_rate": {"avg": 0.75, "count": 2},
                },
            }
        ),
        encoding="utf-8",
    )
    (on_dir / "summary.json").write_text(
        json.dumps(
            {
                "pass_rate": 1.0,
                "metrics": {
                    "memory.retrieval_hit_rate": {"avg": 0.75, "count": 2},
                    "tool.success_rate": {"avg": 1.0, "count": 2},
                },
            }
        ),
        encoding="utf-8",
    )

    comparison = aggregate_experiment_comparison(
        module="memory",
        variants=("off", "on"),
        variant_dirs={
            "off": [off_dir],
            "on": [on_dir],
        },
    )

    assert comparison["metrics"]["task.pass_rate"]["off"] == 0.5
    assert comparison["metrics"]["task.pass_rate"]["on"] == 1.0
    assert comparison["metrics"]["task.pass_rate"]["delta"] == 0.5
    assert comparison["metrics"]["memory.retrieval_hit_rate"]["delta"] == 0.5


def _context_compression_cases() -> list[EvalCase]:
    return [
        case
        for case in load_eval_suite(Path("benchmarks/evaluation_v2/context"))
        if case.context_profile.get("kind") == "compression"
    ]
