from __future__ import annotations

"""Persistence for Eval artifacts."""

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import EvalResult, EvaluationEvidence
from .verifier import format_workspace_diff


EVAL_SCHEMA_VERSION = "1"


class EvalArtifactStore:
    def __init__(self, root: str | Path, eval_id: str) -> None:
        self.root = Path(root) / eval_id
        self.cases_root = self.root / "cases"
        self.workspaces_root = self.root / "workspaces"

    def initialize_manifest(
        self,
        *,
        benchmark_name: str,
        model_id: str | None,
        provider: str | None,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_json(
            self.root / "manifest.json",
            {
                "schema_version": EVAL_SCHEMA_VERSION,
                "eval_id": self.root.name,
                "benchmark_name": benchmark_name,
                "created_at": _utc_now(),
                "codepilot_version": _codepilot_version(),
                "git_commit": _git_commit(),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "model_id": model_id,
                "provider": provider,
            },
        )

    def case_dir(self, case_id: str) -> Path:
        path = self.cases_root / case_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def workspace_dir(self, case_id: str) -> Path:
        path = self.workspaces_root / case_id
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_case_definition(self, case_id: str, definition: object) -> None:
        self._write_json(
            self.case_dir(case_id) / "case.json",
            {
                "schema_version": EVAL_SCHEMA_VERSION,
                "definition": definition,
            },
        )

    def write_case_result(
        self,
        result: EvalResult,
        evidence: EvaluationEvidence,
    ) -> None:
        self.write_result(result)
        case_dir = self.case_dir(result.case_id)
        (case_dir / "workspace.diff").write_text(
            format_workspace_diff(evidence.changes),
            encoding="utf-8",
        )

    def write_result(self, result: EvalResult) -> None:
        """Write result metadata even when no workspace could be prepared."""

        case_dir = self.case_dir(result.case_id)
        self._write_json(
            case_dir / "result.json",
            {
                "schema_version": EVAL_SCHEMA_VERSION,
                "result": result,
            },
        )
        self._write_json(
            case_dir / "verifier-results.json",
            {
                "schema_version": EVAL_SCHEMA_VERSION,
                "results": result.verifier_results,
            },
        )
        (case_dir / "workspace.diff").write_text("", encoding="utf-8")
        self._write_json(
            case_dir / "run-refs.json",
            {
                "schema_version": EVAL_SCHEMA_VERSION,
                "session_id": result.session_id,
                "runs": [
                    {
                        "run_id": run_id,
                        "artifact_path": (
                            f".codepilot/runs/{run_id}"
                        ),
                    }
                    for run_id in result.run_ids
                ],
            },
        )

    def write_summary(self, summary: dict[str, Any]) -> None:
        self._write_json(
            self.root / "summary.json",
            {
                "schema_version": EVAL_SCHEMA_VERSION,
                "summary": summary,
            },
        )

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                _json_value(value),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _codepilot_version() -> str:
    try:
        from codepilot import __version__

        return str(__version__)
    except (ImportError, AttributeError):
        return "unknown"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2,
            check=False,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None
