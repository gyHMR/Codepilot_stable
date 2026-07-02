from __future__ import annotations

import json
from pathlib import Path

from codepilot.evaluation.artifacts import EvaluationArtifacts
from codepilot.evaluation.evidence import (
    ContextEvidence,
    EvalEvidence,
    ToolCallEvidence,
)
from codepilot.evaluation.experiments import run_context_ab
from codepilot.evaluation.loader import load_eval_case
from codepilot.evaluation.reports import build_summary, render_markdown
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
                tokens_after=200,
            )
        ],
        tools=[
            ToolCallEvidence(
                tool_call_id="1",
                tool_name="write",
                status="denied",
                error_reason="read_only_mode",
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
            "context.stale_context_rate",
            "security.dangerous_block_rate",
            "security.benign_pass_rate",
            "security.mutation_after_denial_rate",
        ],
    )

    assert scores["task.pass_rate"].value == 1.0
    assert scores["context.key_context_hit_rate"].value == 1.0
    assert scores["context.token_efficiency"].value == 0.5
    assert scores["context.stale_context_rate"].value == 0.5
    assert scores["security.dangerous_block_rate"].value == 1.0
    assert scores["security.benign_pass_rate"].value == 1.0
    assert scores["security.mutation_after_denial_rate"].value == 0.0


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
                "expected": {"key_context": ["src/app.py"]},
                "candidates": [
                    {"id": "docs/noise.md", "path": "docs/noise.md", "tokens": 100},
                    {"id": "src/app.py", "path": "src/app.py", "tokens": 50},
                ],
                "selected": [
                    {"id": "src/app.py", "path": "src/app.py", "tokens": 50}
                ],
                "budget_tokens": 100,
            }
        ]
    )

    assert comparison["module"] == "context"
    assert comparison["metrics"]["context.key_context_hit_rate"]["off"] == 0.0
    assert comparison["metrics"]["context.key_context_hit_rate"]["on"] == 1.0
