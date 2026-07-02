from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from codepilot.evaluation.__main__ import build_parser
from codepilot.evaluation.experiment import (
    build_experiment_comparison,
    experiment_variants,
    run_experiment,
)
from codepilot.evaluation.types import EvalRunOptions
from codepilot.runtime.types import CreateAgentSessionOptions


def test_cli_exposes_check_run_experiment_and_report_commands() -> None:
    parser = build_parser()

    assert parser.parse_args(["check"]).command == "check"
    assert parser.parse_args(["run", "context"]).module == "context"
    filtered = parser.parse_args(
        ["run", "all", "--include-tag", "suite:graded", "--include-tag", "difficulty:hard"]
    )
    assert filtered.include_tags == ["suite:graded", "difficulty:hard"]
    experiment = parser.parse_args(
        ["experiment", "memory", "--repeat", "3", "--include-tag", "suite:graded"]
    )
    assert experiment.module == "memory"
    assert experiment.repeat == 3
    assert experiment.include_tags == ["suite:graded"]
    with pytest.raises(SystemExit):
        parser.parse_args(["experiment", "context"])
    assert parser.parse_args(["report", ".codepilot/evals/eval_1"]).command == "report"


def test_experiment_variants_map_module_to_one_runtime_toggle() -> None:
    assert experiment_variants("memory") == [
        ("off", {"memory_enabled": False}),
        ("on", {"memory_enabled": True}),
    ]
    assert experiment_variants("planning") == [
        ("off", {"task_control_enabled": False}),
        ("on", {"task_control_enabled": True}),
    ]
    with pytest.raises(ValueError):
        experiment_variants("context")


def test_experiment_comparison_reports_on_off_averages_and_change() -> None:
    comparison = build_experiment_comparison(
        "memory",
        {
            "off": [
                {
                    "metric_averages": {
                        "memory.memory_retrieval_hit_rate": 0.0,
                        "memory.redundant_read_count": 2.0,
                    },
                    "pass_rate": 0.5,
                }
            ],
            "on": [
                {
                    "metric_averages": {
                        "memory.memory_retrieval_hit_rate": 1.0,
                        "memory.redundant_read_count": 1.0,
                    },
                    "pass_rate": 1.0,
                }
            ],
        },
    )

    assert comparison["metrics"]["memory.memory_retrieval_hit_rate"] == {
        "off": 0.0,
        "on": 1.0,
        "change": 1.0,
    }
    assert comparison["metrics"]["memory.redundant_read_count"] == {
        "off": 2.0,
        "on": 1.0,
        "change": -1.0,
    }
    assert comparison["metrics"]["memory.redundant_read_reduction_rate"] == {
        "off": 0.0,
        "on": 0.5,
        "change": 0.5,
    }
    assert comparison["pass_rate"] == {
        "off": 0.5,
        "on": 1.0,
        "change": 0.5,
    }


def test_experiment_comparison_prefers_primary_metric_averages() -> None:
    comparison = build_experiment_comparison(
        "memory",
        {
            "off": [
                {
                    "primary_metric_averages": {
                        "memory.memory_retrieval_hit_rate": 0.25,
                    },
                    "metric_averages": {
                        "memory.memory_retrieval_hit_rate": 0.0,
                    },
                    "pass_rate": 0.5,
                }
            ],
            "on": [
                {
                    "primary_metric_averages": {
                        "memory.memory_retrieval_hit_rate": 0.75,
                    },
                    "metric_averages": {
                        "memory.memory_retrieval_hit_rate": 1.0,
                    },
                    "pass_rate": 1.0,
                }
            ],
        },
    )

    assert comparison["metrics"]["memory.memory_retrieval_hit_rate"] == {
        "off": 0.25,
        "on": 0.75,
        "change": 0.5,
    }


def test_planning_experiment_reports_invalid_tool_call_reduction() -> None:
    comparison = build_experiment_comparison(
        "planning",
        {
            "off": [
                {
                    "metric_averages": {
                        "planning.invalid_tool_call_count": 3.0,
                    },
                    "pass_rate": 0.5,
                }
            ],
            "on": [
                {
                    "metric_averages": {
                        "planning.invalid_tool_call_count": 1.0,
                    },
                    "pass_rate": 1.0,
                }
            ],
        },
    )

    assert comparison["metrics"]["planning.invalid_tool_call_count"] == {
        "off": 3.0,
        "on": 1.0,
        "change": -2.0,
    }
    assert comparison["metrics"]["planning.invalid_tool_call_reduction_count"] == {
        "off": 0.0,
        "on": 2.0,
        "change": 2.0,
    }


def test_experiment_run_artifacts_are_nested_under_experiment_root(
    tmp_path: Path,
) -> None:
    class FakeService:
        def __init__(self) -> None:
            self.options = []

        async def run_suite(self, suite_path: Path, options: EvalRunOptions):
            self.options.append(options)
            return SimpleNamespace(
                summary={"primary_metric_averages": {}, "pass_rate": 1.0},
                artifact_dir=str(Path(options.artifact_root) / str(options.eval_id)),
            )

    service = FakeService()
    options = EvalRunOptions(
        fixtures_root=tmp_path / "fixtures",
        artifact_root=tmp_path / "evals",
        eval_id="exp-memory-r1",
        benchmark_name="memory",
        session_options=CreateAgentSessionOptions(workspace_dir=tmp_path),
    )

    result = asyncio.run(
        run_experiment(
            service,
            tmp_path / "suite",
            options,
            module="memory",
            repeat=1,
        )
    )

    expected_runs_root = tmp_path / "evals" / "exp-memory-r1" / "runs"
    assert [item.eval_id for item in service.options] == ["off_1", "on_1"]
    assert all(item.artifact_root == expected_runs_root for item in service.options)
    assert result["artifact_dir"] == str(tmp_path / "evals" / "exp-memory-r1")
    assert result["runs"] == {
        "off": [str(expected_runs_root / "off_1")],
        "on": [str(expected_runs_root / "on_1")],
    }
