from __future__ import annotations

"""Outcome and Runtime-contract verifiers."""

import fnmatch
import hashlib
import subprocess
from pathlib import Path
from typing import Any

from codepilot.observability import build_run_report

from .types import (
    EvaluationEvidence,
    VerifierResult,
    VerifierSpec,
    WorkspaceChange,
)


_IGNORED_ROOTS = {".codepilot", ".git", "__pycache__"}


def capture_workspace_baseline(workspace: Path) -> dict[str, str]:
    """Capture content hashes without depending on Git."""

    baseline: dict[str, str] = {}
    for path in _workspace_files(workspace):
        baseline[_relative_path(workspace, path)] = _sha256_file(path)
    return baseline


def compute_workspace_changes(
    workspace: Path,
    baseline: dict[str, str],
) -> list[WorkspaceChange]:
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


def run_verifiers(
    specs: list[VerifierSpec],
    evidence: EvaluationEvidence,
) -> list[VerifierResult]:
    evidence.changes = compute_workspace_changes(
        evidence.workspace,
        evidence.baseline,
    )
    results: list[VerifierResult] = []
    for spec in specs:
        try:
            results.append(_run_verifier(spec, evidence))
        except Exception as exc:
            results.append(
                VerifierResult(
                    name=spec.type,
                    status="error",
                    summary=f"Verifier raised {type(exc).__name__}: {exc}",
                    expected=spec.options,
                    evidence={"error_kind": type(exc).__name__},
                )
            )
    return results


def _run_verifier(
    spec: VerifierSpec,
    evidence: EvaluationEvidence,
) -> VerifierResult:
    if spec.type == "command":
        return _verify_command(spec, evidence)
    if spec.type == "file":
        return _verify_file(spec, evidence)
    if spec.type == "diff":
        return _verify_diff(spec, evidence)
    if spec.type == "run":
        return _verify_run(spec, evidence)
    if spec.type == "trace":
        return _verify_trace(spec, evidence)
    raise ValueError(f"Unsupported verifier: {spec.type}")


def _verify_command(
    spec: VerifierSpec,
    evidence: EvaluationEvidence,
) -> VerifierResult:
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
        return VerifierResult(
            name="command",
            status="failed",
            summary=f"Command timed out after {timeout}s",
            expected={"exit_code": expected_exit, "timeout_seconds": timeout},
            actual={"timed_out": True},
            evidence={
                "command": command,
                "stdout": _text(exc.stdout),
                "stderr": _text(exc.stderr),
            },
        )

    passed = completed.returncode == expected_exit
    return VerifierResult(
        name="command",
        status="passed" if passed else "failed",
        summary=(
            f"Command exited with {completed.returncode}"
            if passed
            else f"Expected exit code {expected_exit}, got {completed.returncode}"
        ),
        expected={"exit_code": expected_exit},
        actual={"exit_code": completed.returncode},
        evidence={
            "command": command,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        },
    )


def _verify_file(
    spec: VerifierSpec,
    evidence: EvaluationEvidence,
) -> VerifierResult:
    relative = str(spec.options["path"])
    path = _safe_workspace_path(evidence.workspace, relative)
    expected_exists = bool(spec.options.get("exists", True))
    actual_exists = path.is_file()
    failures: list[str] = []
    actual: dict[str, Any] = {"exists": actual_exists}
    expected: dict[str, Any] = {"exists": expected_exists}

    if actual_exists != expected_exists:
        failures.append(
            f"expected exists={expected_exists}, got {actual_exists}"
        )
    if actual_exists:
        content = path.read_text(encoding="utf-8")
        actual["sha256"] = _sha256_file(path)
        if "contains" in spec.options:
            needles = spec.options["contains"]
            if isinstance(needles, str):
                needles = [needles]
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
            actual["equals"] = content
            if content != wanted:
                failures.append("file content did not match exactly")
        if "sha256" in spec.options:
            wanted_hash = str(spec.options["sha256"]).lower()
            expected["sha256"] = wanted_hash
            if actual["sha256"] != wanted_hash:
                failures.append("file hash did not match")

    return VerifierResult(
        name="file",
        status="failed" if failures else "passed",
        summary="; ".join(failures) if failures else f"File verified: {relative}",
        expected=expected,
        actual=actual,
        evidence={"path": relative},
    )


def _verify_diff(
    spec: VerifierSpec,
    evidence: EvaluationEvidence,
) -> VerifierResult:
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
    failures: list[str] = []
    if outside_allowed:
        failures.append(f"changes outside allowed paths: {outside_allowed}")
    if forbidden_changes:
        failures.append(f"forbidden paths changed: {forbidden_changes}")
    return VerifierResult(
        name="diff",
        status="failed" if failures else "passed",
        summary="; ".join(failures) if failures else "Workspace diff is allowed",
        expected={
            "allowed_paths": allowed,
            "forbidden_paths": forbidden,
        },
        actual={"changed_paths": changed_paths},
        evidence={
            "changes": [
                {"path": item.path, "status": item.status}
                for item in evidence.changes
            ]
        },
    )


def _verify_run(
    spec: VerifierSpec,
    evidence: EvaluationEvidence,
) -> VerifierResult:
    run_id = _select_run_id(spec, evidence)
    if run_id is None:
        return VerifierResult(
            name="run",
            status="skipped",
            summary="No Run Artifact is available",
            expected=spec.options,
            actual=None,
        )
    result = evidence.run_results.get(run_id)
    if result is None:
        return VerifierResult(
            name="run",
            status="error",
            summary=f"Run result is missing: {run_id}",
            expected=spec.options,
            evidence={"run_id": run_id},
        )
    events = evidence.run_events.get(run_id, [])
    report = build_run_report(result, events=events)
    summary = report["summary"]
    failures: list[str] = []
    comparisons = {
        "expect_status": "status",
        "expect_stop_reason": "stop_reason",
        "expect_tool_calls": "tool_calls",
        "expect_model_attempts": "model_attempts",
        "expect_workspace_changed": "workspace_changed",
    }
    expected: dict[str, Any] = {}
    actual: dict[str, Any] = {}
    for option, field in comparisons.items():
        if option not in spec.options:
            continue
        expected[field] = spec.options[option]
        actual[field] = summary.get(field)
        if expected[field] != actual[field]:
            failures.append(
                f"{field}: expected {expected[field]!r}, got {actual[field]!r}"
            )
    if "expect_freshness" in spec.options:
        expected["freshness"] = spec.options["expect_freshness"]
        freshness_statuses = [
            item.get("status")
            for item in evidence.freshness_history
            if isinstance(item.get("status"), str)
        ]
        if not freshness_statuses and evidence.freshness:
            freshness_statuses.append(evidence.freshness.get("status"))
        actual["freshness"] = freshness_statuses
        if expected["freshness"] not in freshness_statuses:
            failures.append(
                "freshness: expected "
                f"{expected['freshness']!r}, observed {freshness_statuses!r}"
            )
    return VerifierResult(
        name="run",
        status="failed" if failures else "passed",
        summary="; ".join(failures) if failures else f"Run verified: {run_id}",
        expected=expected,
        actual=actual,
        evidence={"run_id": run_id, "report": report},
    )


def _verify_trace(
    spec: VerifierSpec,
    evidence: EvaluationEvidence,
) -> VerifierResult:
    run_id = _select_run_id(spec, evidence)
    if run_id is None:
        return VerifierResult(
            name="trace",
            status="skipped",
            summary="No Run trace is available",
            expected=spec.options,
        )
    events = evidence.run_events.get(run_id, [])
    result = evidence.run_results.get(run_id, {})
    starts = [
        event for event in events
        if event.get("type") == "tool_execution_start"
    ]
    ends = [
        event for event in events
        if event.get("type") == "tool_execution_end"
    ]
    start_ids = [_tool_call_id(event) for event in starts]
    end_ids = [_tool_call_id(event) for event in ends]
    failures: list[str] = []

    if spec.options.get("require_lifecycle", True):
        agent_starts = sum(
            event.get("type") == "agent_start" for event in events
        )
        agent_ends = sum(event.get("type") == "agent_end" for event in events)
        if agent_starts != 1 or agent_ends != 1:
            failures.append(
                f"lifecycle expected 1 start/end, got {agent_starts}/{agent_ends}"
            )
    if spec.options.get("require_tool_pairing", True):
        if sorted(start_ids) != sorted(end_ids):
            failures.append(
                f"tool calls are not paired: starts={start_ids}, ends={end_ids}"
            )
    if spec.options.get("check_counters", True) and result:
        counters = result.get("counters", {})
        expected_calls = counters.get("tool_calls", 0)
        if expected_calls != len(starts):
            failures.append(
                f"tool_calls counter={expected_calls}, trace starts={len(starts)}"
            )

    return VerifierResult(
        name="trace",
        status="failed" if failures else "passed",
        summary="; ".join(failures) if failures else f"Trace verified: {run_id}",
        expected={
            "require_lifecycle": spec.options.get("require_lifecycle", True),
            "require_tool_pairing": spec.options.get(
                "require_tool_pairing",
                True,
            ),
            "check_counters": spec.options.get("check_counters", True),
        },
        actual={
            "event_count": len(events),
            "tool_start_ids": start_ids,
            "tool_end_ids": end_ids,
        },
        evidence={"run_id": run_id},
    )


def _select_run_id(
    spec: VerifierSpec,
    evidence: EvaluationEvidence,
) -> str | None:
    requested = spec.options.get("run_id", "latest")
    if requested == "latest":
        return evidence.run_ids[-1] if evidence.run_ids else None
    if requested == "first":
        return evidence.run_ids[0] if evidence.run_ids else None
    return str(requested) if requested else None


def _workspace_files(workspace: Path):
    if not workspace.exists():
        return
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(workspace).parts
        if relative_parts and relative_parts[0] in _IGNORED_ROOTS:
            continue
        if any(part == "__pycache__" for part in relative_parts):
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


def _tool_call_id(event: dict[str, Any]) -> str:
    value = event.get("toolCallId") or event.get("tool_call_id")
    return str(value or "")


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
