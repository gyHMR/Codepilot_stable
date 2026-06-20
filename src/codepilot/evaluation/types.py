from __future__ import annotations

"""Evaluation domain types.

The evaluation layer describes expected behavior and verdicts. It intentionally
stores references to Runtime artifacts instead of copying full event streams.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from codepilot.runtime import CreateAgentSessionOptions


EvalCategory = Literal["harness", "coding"]
EvalVerdict = Literal[
    "passed",
    "task_failed",
    "harness_failed",
    "recovery_failed",
    "invalid_case",
]
VerifierStatus = Literal["passed", "failed", "error", "skipped"]
VerifierType = Literal["command", "file", "diff", "run", "trace"]
ScenarioStepType = Literal[
    "prompt",
    "cancel",
    "modify_file",
    "restart",
    "continue",
    "verify",
]


@dataclass(frozen=True)
class VerifierSpec:
    """A verifier declaration loaded from benchmark JSON."""

    type: VerifierType
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioStep:
    """One recovery-scenario action."""

    type: ScenarioStepType
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalCase:
    id: str
    category: EvalCategory
    fixture: str
    prompt: str
    timeout_seconds: int = 120
    verifiers: list[VerifierSpec] = field(default_factory=list)


@dataclass(frozen=True)
class EvalScenario:
    id: str
    fixture: str
    steps: list[ScenarioStep]
    verifiers: list[VerifierSpec] = field(default_factory=list)
    timeout_seconds: int = 120


@dataclass(frozen=True)
class VerifierResult:
    name: str
    status: VerifierStatus
    summary: str
    expected: object | None = None
    actual: object | None = None
    evidence: dict[str, object] = field(default_factory=dict)


@dataclass
class EvalResult:
    case_id: str
    verdict: EvalVerdict
    session_id: str | None
    run_ids: list[str]
    verifier_results: list[VerifierResult]
    artifact_dir: str
    error: str | None = None
    duration_ms: int | None = None


@dataclass
class EvalSuiteResult:
    eval_id: str
    results: list[EvalResult]
    artifact_dir: str
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalRunOptions:
    """Execution configuration supplied by the caller, not benchmark JSON."""

    fixtures_root: str | Path
    session_options: CreateAgentSessionOptions
    artifact_root: str | Path = ".codepilot/evals"
    eval_id: str | None = None
    benchmark_name: str = ""
    keep_workspace: bool = True


@dataclass(frozen=True)
class WorkspaceChange:
    path: str
    status: Literal["added", "modified", "deleted"]


@dataclass
class EvaluationEvidence:
    """Evidence shared by verifiers for one case or scenario."""

    workspace: Path
    baseline: dict[str, str]
    session_id: str | None = None
    run_ids: list[str] = field(default_factory=list)
    run_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    run_events: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    freshness: dict[str, Any] | None = None
    freshness_history: list[dict[str, Any]] = field(default_factory=list)
    changes: list[WorkspaceChange] = field(default_factory=list)
    step_results: list[dict[str, Any]] = field(default_factory=list)
