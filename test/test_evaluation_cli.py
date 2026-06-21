from __future__ import annotations

from codepilot.evaluation.__main__ import build_parser
from codepilot.evaluation.experiment import (
    build_experiment_comparison,
    experiment_variants,
)


def test_cli_exposes_check_run_experiment_and_report_commands() -> None:
    parser = build_parser()

    assert parser.parse_args(["check"]).command == "check"
    assert parser.parse_args(["run", "context"]).module == "context"
    experiment = parser.parse_args(
        ["experiment", "memory", "--repeat", "3"]
    )
    assert experiment.module == "memory"
    assert experiment.repeat == 3
    assert parser.parse_args(["report", ".codepilot/evals/eval_1"]).command == "report"


def test_experiment_variants_map_module_to_one_runtime_toggle() -> None:
    assert experiment_variants("context") == [
        ("off", {"context_governance_enabled": False}),
        ("on", {"context_governance_enabled": True}),
    ]
    assert experiment_variants("memory") == [
        ("off", {"memory_enabled": False}),
        ("on", {"memory_enabled": True}),
    ]
    assert experiment_variants("planning") == [
        ("off", {"task_control_enabled": False}),
        ("on", {"task_control_enabled": True}),
    ]


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
