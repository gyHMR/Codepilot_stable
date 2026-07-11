from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


def _load_script_module():
    script_path = Path("scripts/run_evaluation_v2.py")
    spec = importlib.util.spec_from_file_location("run_evaluation_v2", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_evaluation_script_builds_four_metric_steps(tmp_path: Path) -> None:
    module = _load_script_module()
    args = Namespace(
        suite_root=Path("benchmarks/evaluation_v2"),
        fixtures_root=Path("benchmarks/fixtures"),
        workspace_policy="failed",
        provider=None,
        model_id=None,
    )

    steps = module._build_steps(args, tmp_path / "group")
    names = [step[0] for step in steps]
    commands = [" ".join(step[1]) for step in steps]

    assert names == [
        "tool-dangerous-block-rate",
        "memory-retrieval-ranking",
        "context-key-hit-ab",
        "context-compression-rate",
    ]
    assert "run security" in commands[0]
    assert "ab memory" in commands[1]
    assert "ab context" in commands[2]
    assert "run context" in commands[3]
    assert "--include-tag suite:context-compression" in commands[3]
    assert all("experiment" not in command for command in commands)
    assert all("planning" not in command for command in commands)


def test_run_evaluation_script_collects_target_metrics(tmp_path: Path) -> None:
    module = _load_script_module()
    group = tmp_path / "group"
    (group / "tool-dangerous-block-rate").mkdir(parents=True)
    (group / "memory-retrieval-ranking").mkdir()
    (group / "context-key-hit-ab").mkdir()
    (group / "context-compression-rate").mkdir()
    (group / "tool-dangerous-block-rate" / "summary.json").write_text(
        json.dumps({"metrics": {"security.dangerous_block_rate": {"avg": 1.0, "count": 10}}}),
        encoding="utf-8",
    )
    (group / "memory-retrieval-ranking" / "comparison.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "memory.recall@3": {"avg": 0.8, "count": 10},
                    "memory.precision@3": {"avg": 0.4, "count": 10},
                    "memory.mrr": {"avg": 0.9, "count": 10},
                    "memory.forbidden_retrieval_rate": {"avg": 0.1, "count": 10},
                }
            }
        ),
        encoding="utf-8",
    )
    (group / "context-key-hit-ab" / "comparison.json").write_text(
        json.dumps({"metrics": {"context.key_context_hit_rate": {"on": 1.0, "delta": 0.95}}}),
        encoding="utf-8",
    )
    (group / "context-compression-rate" / "summary.json").write_text(
        json.dumps({"metrics": {"context.compression_rate": {"avg": 0.42, "count": 10}}}),
        encoding="utf-8",
    )

    metrics = module._collect_target_metrics(group)

    assert metrics["security.dangerous_block_rate"]["value"] == 1.0
    assert metrics["memory.recall@3"]["value"] == 0.8
    assert metrics["memory.precision@3"]["value"] == 0.4
    assert metrics["memory.mrr"]["value"] == 0.9
    assert metrics["memory.forbidden_retrieval_rate"]["value"] == 0.1
    assert metrics["context.key_context_hit_rate.ab_on"]["value"] == 1.0
    assert metrics["context.key_context_hit_rate.ab_delta"]["value"] == 0.95
    assert metrics["context.compression_rate"]["value"] == 0.42
