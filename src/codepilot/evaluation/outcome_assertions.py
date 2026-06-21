from __future__ import annotations

"""对隔离工作区的结果断言（命令执行、文件检查、diff 检查）。"""

import fnmatch
import hashlib
import subprocess
from pathlib import Path
from typing import Any

from .types import AssertionResult, AssertionSpec, EvalEvidence, WorkspaceChange


_IGNORED_ROOTS = {".codepilot", ".git", "__pycache__", ".pytest_cache"}


def capture_workspace_baseline(workspace: Path) -> dict[str, str]:
    """捕获工作区基线快照（路径 -> SHA256）。"""
    return {
        _relative_path(workspace, path): _sha256_file(path)
        for path in _workspace_files(workspace)
    }


def compute_workspace_changes(
    workspace: Path,
    baseline: dict[str, str],
) -> list[WorkspaceChange]:
    """对比基线和当前工作区，计算变更列表。"""
    current = capture_workspace_baseline(workspace)
    changes: list[WorkspaceChange] = []
    for path in sorted(set(baseline) | set(current)):
        if path not in baseline:
            status = "added"
        elif path not in current:
            status = "deleted"
        elif baseline[path] != current[path]:
            status = "modified"
        else:
            continue
        changes.append(WorkspaceChange(path=path, status=status))
    return changes


def format_workspace_diff(changes: list[WorkspaceChange]) -> str:
    prefix = {"added": "A", "modified": "M", "deleted": "D"}
    return "\n".join(f"{prefix[item.status]} {item.path}" for item in changes)


def run_outcome_assertion(
    spec: AssertionSpec,
    evidence: EvalEvidence,
) -> AssertionResult:
    if spec.type == "command":
        return _assert_command(spec, evidence)
    if spec.type == "file":
        return _assert_file(spec, evidence)
    if spec.type == "diff":
        return _assert_diff(spec, evidence)
    raise ValueError(f"Unsupported outcome assertion: {spec.type}")


def _assert_command(
    spec: AssertionSpec,
    evidence: EvalEvidence,
) -> AssertionResult:
    command = str(spec.options["command"])
    expected_exit = int(spec.options.get("expect_exit_code", 0))
    timeout = int(spec.options.get("timeout_seconds", 120))
    try:
        completed = subprocess.run(
            command,
            cwd=evidence.workspace,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return _result(
            spec,
            "failed",
            f"Command timed out after {timeout}s",
            {"exit_code": expected_exit, "timeout_seconds": timeout},
            {"timed_out": True},
            [f"command:{command}"],
        )
    passed = completed.returncode == expected_exit
    summary = (
        f"Command exited with {completed.returncode}"
        if passed
        else f"Expected exit code {expected_exit}, got {completed.returncode}"
    )
    return _result(
        spec,
        "passed" if passed else "failed",
        summary,
        {"exit_code": expected_exit},
        {
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        },
        [f"command:{command}"],
    )


def _assert_file(
    spec: AssertionSpec,
    evidence: EvalEvidence,
) -> AssertionResult:
    relative = str(spec.options["path"])
    path = _safe_workspace_path(evidence.workspace, relative)
    expected_exists = bool(spec.options.get("exists", True))
    actual_exists = path.is_file()
    failures: list[str] = []
    expected: dict[str, Any] = {"exists": expected_exists}
    actual: dict[str, Any] = {"exists": actual_exists}
    if actual_exists != expected_exists:
        failures.append(
            f"expected exists={expected_exists}, got {actual_exists}"
        )
    if actual_exists:
        content = path.read_text(encoding="utf-8")
        actual["sha256"] = _sha256_file(path)
        if "contains" in spec.options:
            raw = spec.options["contains"]
            needles = [raw] if isinstance(raw, str) else raw
            if not isinstance(needles, list) or not all(
                isinstance(item, str) for item in needles
            ):
                raise ValueError("file.contains must be a string or string array")
            missing = [item for item in needles if item not in content]
            expected["contains"] = needles
            actual["missing"] = missing
            if missing:
                failures.append(f"missing expected content: {missing}")
        if "equals" in spec.options:
            wanted = str(spec.options["equals"])
            expected["equals"] = wanted
            actual["content"] = content
            if content != wanted:
                failures.append("file content did not match exactly")
        if "sha256" in spec.options:
            wanted_hash = str(spec.options["sha256"]).lower()
            expected["sha256"] = wanted_hash
            if actual["sha256"] != wanted_hash:
                failures.append("file hash did not match")
    return _result(
        spec,
        "failed" if failures else "passed",
        "; ".join(failures) if failures else f"File verified: {relative}",
        expected,
        actual,
        [f"file:{relative}"],
    )


def _assert_diff(
    spec: AssertionSpec,
    evidence: EvalEvidence,
) -> AssertionResult:
    allowed = [str(item) for item in spec.options.get("allowed_paths", [])]
    forbidden = [str(item) for item in spec.options.get("forbidden_paths", [])]
    changed_paths = [item.path for item in evidence.changes]
    outside_allowed = (
        [
            path
            for path in changed_paths
            if not any(_path_matches(path, pattern) for pattern in allowed)
        ]
        if allowed
        else []
    )
    forbidden_changes = [
        path
        for path in changed_paths
        if any(_path_matches(path, pattern) for pattern in forbidden)
    ]
    failures = []
    if outside_allowed:
        failures.append(f"changes outside allowed paths: {outside_allowed}")
    if forbidden_changes:
        failures.append(f"forbidden paths changed: {forbidden_changes}")
    return _result(
        spec,
        "failed" if failures else "passed",
        "; ".join(failures) if failures else "Workspace diff is allowed",
        {"allowed_paths": allowed, "forbidden_paths": forbidden},
        {"changed_paths": changed_paths},
        [f"file:{path}" for path in changed_paths],
    )


def _result(
    spec: AssertionSpec,
    status: str,
    summary: str,
    expected: object,
    actual: object,
    refs: list[str],
) -> AssertionResult:
    return AssertionResult(
        name=spec.type,
        dimension=spec.dimension,
        status=status,  # type: ignore[arg-type]
        summary=summary,
        expected=expected,
        actual=actual,
        evidence_refs=refs,
        required=spec.required,
    )


def _workspace_files(workspace: Path):
    if not workspace.exists():
        return
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(workspace).parts
        if any(part in _IGNORED_ROOTS for part in relative_parts):
            continue
        yield path


def _safe_workspace_path(workspace: Path, relative: str) -> Path:
    target = (workspace / relative).resolve()
    root = workspace.resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Path escapes workspace: {relative}")
    return target


def _relative_path(workspace: Path, path: Path) -> str:
    return path.relative_to(workspace).as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_matches(path: str, pattern: str) -> bool:
    normalized = pattern.replace("\\", "/").rstrip("/")
    return (
        path == normalized
        or path.startswith(normalized + "/")
        or fnmatch.fnmatchcase(path, normalized)
    )


__all__ = [
    "capture_workspace_baseline",
    "compute_workspace_changes",
    "format_workspace_diff",
    "run_outcome_assertion",
]
