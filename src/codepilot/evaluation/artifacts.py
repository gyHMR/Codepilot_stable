from __future__ import annotations

# 新手导读：artifacts.py 管理评估产物路径和输出文件。
# 关注点：它让 benchmark 结果可复查，而不是只停留在终端输出。

"""Persistent artifact writer for evaluation v2."""

import csv
import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from codepilot.observability.redact import redact_artifact

from .evidence import EvalEvidence
from .schema import EvalCase, EvalResult


class EvaluationArtifacts:
    """Write the fixed ``.codepilot/evals/<eval_id>`` layout."""

    def __init__(self, artifact_root: Path | str, eval_id: str) -> None:
        self.root = Path(artifact_root) / eval_id
        self.cases_root = self.root / "cases"
        self._case_rows: list[dict[str, Any]] = []
        self._metric_rows: list[dict[str, Any]] = []

    def initialize(self, benchmark_name: str, *, case_count: int) -> None:
        self.cases_root.mkdir(parents=True, exist_ok=True)
        self.write_json(
            self.root / "manifest.json",
            {
                "schema_version": 1,
                "eval_id": self.root.name,
                "benchmark_name": benchmark_name,
                "case_count": case_count,
                "created_at_ms": int(time.time() * 1000),
            },
        )

    def case_dir(self, case_id: str) -> Path:
        safe_id = "".join(
            char if char.isalnum() or char in {"-", "_", "."} else "_"
            for char in case_id
        )
        return self.cases_root / safe_id

    def write_case(
        self,
        case: EvalCase,
        result: EvalResult,
        evidence: EvalEvidence,
        *,
        workspace_diff: str = "",
    ) -> None:
        case_dir = self.case_dir(case.id)
        case_dir.mkdir(parents=True, exist_ok=True)
        self.write_json(case_dir / "case.json", _to_json(case))
        self.write_json(case_dir / "result.json", _to_json(result))
        self.write_json(case_dir / "evidence.json", _to_json(evidence))
        self.write_json(
            case_dir / "scores.json",
            {name: _to_json(score) for name, score in result.metrics.items()},
        )
        self.write_json(
            case_dir / "steps.json",
            [_to_json(step) for step in evidence.steps],
        )
        (case_dir / "workspace.diff").write_text(workspace_diff, encoding="utf-8")
        self.write_json(case_dir / "runs.json", result.run_ids)
        self._remember_case_row(result)

    def write_summary(self, summary: dict[str, Any], markdown: str) -> None:
        self.write_json(self.root / "summary.json", summary)
        (self.root / "report.md").write_text(markdown, encoding="utf-8")
        self._write_csv(self.root / "metrics.csv", self._metric_rows)
        self._write_csv(self.root / "cases.csv", self._case_rows)

    def write_comparison(self, comparison: dict[str, Any], markdown: str) -> None:
        self.write_json(self.root / "comparison.json", comparison)
        (self.root / "report.md").write_text(markdown, encoding="utf-8")

    @staticmethod
    def write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        safe_payload = redact_artifact(_to_json(payload))
        path.write_text(
            json.dumps(safe_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _remember_case_row(self, result: EvalResult) -> None:
        self._case_rows.append(
            {
                "case_id": result.case_id,
                "module": result.module,
                "passed": result.passed,
                "run_ids": " ".join(result.run_ids),
                "error": result.error or "",
            }
        )
        for score in result.metrics.values():
            self._metric_rows.append(
                {
                    "case_id": result.case_id,
                    "module": result.module,
                    "metric": score.name,
                    "value": "" if score.value is None else score.value,
                    "display": score.display,
                    "numerator": score.numerator,
                    "denominator": score.denominator,
                }
            )

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row})
        with path.open("w", encoding="utf-8", newline="") as handle:
            if not fieldnames:
                handle.write("")
                return
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _to_json(value: Any) -> Any:
    if is_dataclass(value):
        return _to_json(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = ["EvaluationArtifacts"]
