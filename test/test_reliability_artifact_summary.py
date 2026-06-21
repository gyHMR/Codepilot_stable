from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_summary_module():
    path = ROOT / "scripts" / "summarize_reliability_artifacts.py"
    spec = importlib.util.spec_from_file_location(
        "summarize_reliability_artifacts",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_case(
    root: Path,
    case_id: str,
    domain: str,
    report: dict,
    *,
    assertion_results: list[dict] | None = None,
) -> None:
    _write_json(
        root / "cases" / case_id / "definition.json",
        {
            "schema_version": "2",
            "definition": {
                "id": case_id,
                "domain": domain,
                "fixture": "fixture",
                "prompt": "prompt",
                "assertions": [],
            },
        },
    )
    _write_json(
        root / "cases" / case_id / "assertion-results.json",
        {
            "schema_version": "2",
            "results": assertion_results or [],
        },
    )
    _write_json(
        root
        / "workspaces"
        / case_id
        / ".codepilot"
        / "runs"
        / "run-1"
        / "report.json",
        report,
    )


def test_reliability_artifact_summary_computes_resume_metrics(
    tmp_path: Path,
) -> None:
    module = _load_summary_module()
    eval_root = tmp_path / "eval-test"

    _write_case(
        eval_root,
        "context-current-request",
        "context",
        {
            "context": {
                "current_request_preserved": True,
                "average_compression_ratio": 0.4,
            }
        },
    )
    _write_case(
        eval_root,
        "memory-retrieval-hit",
        "memory",
        {
            "memory": {
                "retrieval_count": 1,
                "read_calls": 2,
            }
        },
        assertion_results=[
            {
                "name": "memory",
                "dimension": "memory",
                "status": "passed",
            }
        ],
    )
    _write_case(
        eval_root,
        "planning-evidence",
        "planning",
        {
            "task": {
                "completed_steps": 4,
                "evidence_ref_count": 3,
            },
            "run": {
                "summary": {
                    "workspace_changed": True,
                }
            },
        },
        assertion_results=[
            {
                "name": "task",
                "dimension": "task_planning",
                "status": "passed",
                "expected": {"false_completion": False},
            }
        ],
    )
    _write_case(
        eval_root,
        "security-read-only",
        "security",
        {
            "security": {
                "tool_status_counts": {
                    "denied": 2,
                    "success": 1,
                },
                "denied_or_approval_count": 2,
                "mutation_after_denial_count": 0,
            }
        },
    )

    summary = module.summarize_eval(eval_root)

    assert summary["metrics"]["context"]["current_request_preservation_rate"]["value"] == 1.0
    assert summary["metrics"]["context"]["average_compression_ratio"]["value"] == 0.4
    assert summary["metrics"]["memory"]["memory_retrieval_hit_rate"]["value"] == 1.0
    assert summary["metrics"]["memory"]["average_repeated_read_count"]["value"] == 2.0
    assert summary["metrics"]["planning"]["evidence_coverage_rate"]["value"] == 0.75
    assert summary["metrics"]["planning"]["false_completion_rate"]["value"] == 0.0
    assert summary["metrics"]["security"]["dangerous_tool_block_rate"]["value"] == 2 / 3
    assert summary["metrics"]["security"]["mutation_after_denial_rate"]["value"] == 0.0
