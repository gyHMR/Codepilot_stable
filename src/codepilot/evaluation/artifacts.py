from __future__ import annotations

"""Persistence and reproducibility metadata for Eval artifacts."""

import hashlib
import json
import locale
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from codepilot.observability import redact_artifact

from .outcome_assertions import format_workspace_diff
from .types import EvalEvidence, EvalResult


EVAL_SCHEMA_VERSION = "2"


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
        benchmark_snapshot: str,
        fixture_snapshot: str,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_json(
            self.root / "manifest.json",
            {
                "schema_version": EVAL_SCHEMA_VERSION,
                "eval_id": self.root.name,
                "benchmark_name": benchmark_name,
                "benchmark_snapshot": benchmark_snapshot,
                "fixture_snapshot": fixture_snapshot,
                "started_at": _utc_now(),
                "finished_at": None,
                "codepilot_version": _codepilot_version(),
                "git_commit": _git_value("rev-parse", "HEAD"),
                "git_branch": _git_value(
                    "rev-parse",
                    "--abbrev-ref",
                    "HEAD",
                ),
                "git_dirty": _git_dirty(),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "timezone": str(datetime.now().astimezone().tzinfo),
                "locale": locale.getlocale(),
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

    def write_case_definition(
        self,
        case_id: str,
        definition: object,
    ) -> None:
        self._write_json(
            self.case_dir(case_id) / "definition.json",
            {
                "schema_version": EVAL_SCHEMA_VERSION,
                "definition": definition,
            },
        )

    def write_case_result(
        self,
        result: EvalResult,
        evidence: EvalEvidence,
    ) -> None:
        self.write_result(result)
        (self.case_dir(result.case_id) / "workspace.diff").write_text(
            format_workspace_diff(evidence.changes),
            encoding="utf-8",
        )

    def write_result(self, result: EvalResult) -> None:
        case_dir = self.case_dir(result.case_id)
        assertions = [
            assertion
            for dimension in result.dimensions
            for assertion in dimension.assertion_results
        ]
        self._write_json(
            case_dir / "result.json",
            {
                "schema_version": EVAL_SCHEMA_VERSION,
                "result": result,
            },
        )
        self._write_json(
            case_dir / "assertion-results.json",
            {
                "schema_version": EVAL_SCHEMA_VERSION,
                "results": assertions,
            },
        )
        self._write_json(
            case_dir / "metrics.json",
            {
                "schema_version": EVAL_SCHEMA_VERSION,
                "metrics": result.metrics,
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
                        "artifact_path": f".codepilot/runs/{run_id}",
                    }
                    for run_id in result.run_ids
                ],
            },
        )

    def write_summary(
        self,
        summary: dict[str, Any],
        *,
        markdown: str,
    ) -> None:
        self._write_json(
            self.root / "summary.json",
            {
                "schema_version": EVAL_SCHEMA_VERSION,
                "summary": summary,
            },
        )
        (self.root / "report.md").write_text(markdown, encoding="utf-8")
        manifest = self._read_json(self.root / "manifest.json")
        manifest["finished_at"] = _utc_now()
        self._write_json(self.root / "manifest.json", manifest)

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                redact_artifact(_json_value(value)),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}


def stable_hash(value: object) -> str:
    encoded = json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_tree(root: str | Path) -> str:
    base = Path(root)
    digest = hashlib.sha256()
    if not base.is_dir():
        return digest.hexdigest()
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        relative = path.relative_to(base)
        if any(
            part in {".git", ".codepilot", "__pycache__", ".pytest_cache"}
            for part in relative.parts
        ):
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
        }
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


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
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
    return result.stdout.strip()


def _git_dirty() -> bool | None:
    value = _git_value("status", "--porcelain")
    return None if value is None else bool(value)


__all__ = [
    "EVAL_SCHEMA_VERSION",
    "EvalArtifactStore",
    "hash_tree",
    "stable_hash",
]
