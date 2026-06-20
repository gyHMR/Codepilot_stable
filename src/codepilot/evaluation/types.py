from __future__ import annotations

"""Stable domain types for multi-dimensional Agent evaluation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from codepilot.observability import AuditBundle
from codepilot.runtime import CreateAgentSessionOptions


EvalDomain = Literal[
    "runtime",
    "coding",
    "context",
    "memory",
    "security",
    "planning",
    "recovery",
]
EvalDimension = Literal[
    "coding_outcome",
    "runtime_contract",
    "context_governance",
    "memory",
    "tool_security",
    "task_planning",
    "recovery",
    "efficiency",
]
EvalOverall = Literal["passed", "failed", "invalid_case", "execution_error"]
DimensionStatus = Literal["passed", "failed", "error", "not_applicable"]
AssertionStatus = Literal["passed", "failed", "error", "skipped"]
AssertionType = Literal[
    "command",
    "file",
    "diff",
    "run",
    "trace",
    "context",
    "memory",
    "security",
    "task",
]
ScenarioStepType = Literal[
    "prompt",
    "cancel",
    "modify_file",
    "restart",
    "continue",
    "verify",
    "inspect",
]


@dataclass(frozen=True)
class AssertionSpec:
    type: AssertionType
    dimension: EvalDimension
    options: dict[str, Any] = field(default_factory=dict)
    required: bool = True


@dataclass(frozen=True)
class ScenarioStep:
    type: ScenarioStepType
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalBudgets:
    max_model_attempts: int | None = None
    max_tool_calls: int | None = None
    max_replans: int | None = None
    timeout_seconds: int = 120


@dataclass(frozen=True)
class EvalRuntimeProfile:
    context_governance_enabled: bool = True
    memory_enabled: bool = True
    task_control_enabled: bool = True
    permission_mode: str = "workspace-write"
    scripted_stream: str | None = None


@dataclass(frozen=True)
class EvalCase:
    id: str
    domain: EvalDomain
    fixture: str
    prompt: str
    assertions: list[AssertionSpec]
    budgets: EvalBudgets = EvalBudgets()
    runtime: EvalRuntimeProfile = EvalRuntimeProfile()
    tags: list[str] = field(default_factory=list)

    @property
    def timeout_seconds(self) -> int:
        return self.budgets.timeout_seconds


@dataclass(frozen=True)
class EvalScenario:
    id: str
    domain: EvalDomain
    fixture: str
    steps: list[ScenarioStep]
    assertions: list[AssertionSpec]
    budgets: EvalBudgets = EvalBudgets()
    runtime: EvalRuntimeProfile = EvalRuntimeProfile()
    tags: list[str] = field(default_factory=list)

    @property
    def timeout_seconds(self) -> int:
        return self.budgets.timeout_seconds


@dataclass(frozen=True)
class AssertionResult:
    name: str
    dimension: EvalDimension
    status: AssertionStatus
    summary: str
    expected: object | None = None
    actual: object | None = None
    evidence_refs: list[str] = field(default_factory=list)
    required: bool = True


@dataclass(frozen=True)
class DimensionResult:
    dimension: EvalDimension
    status: DimensionStatus
    summary: str
    assertion_results: list[AssertionResult]
    metrics: dict[str, float | int | bool] = field(default_factory=dict)


@dataclass
class EvalResult:
    case_id: str
    overall: EvalOverall
    session_id: str | None
    run_ids: list[str]
    dimensions: list[DimensionResult]
    failure_categories: list[str]
    metrics: dict[str, Any]
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
class EvalEvidence:
    workspace: Path
    baseline: dict[str, str]
    session_id: str | None = None
    audit_bundles: list[AuditBundle] = field(default_factory=list)
    freshness_history: list[dict[str, Any]] = field(default_factory=list)
    changes: list[WorkspaceChange] = field(default_factory=list)
    step_results: list[dict[str, Any]] = field(default_factory=list)

    @property
    def run_ids(self) -> list[str]:
        return [bundle.run_id for bundle in self.audit_bundles]

    def select_bundle(self, requested: object = "latest") -> AuditBundle | None:
        if not self.audit_bundles:
            return None
        if requested == "first":
            return self.audit_bundles[0]
        if requested == "latest" or requested is None:
            return self.audit_bundles[-1]
        return next(
            (
                bundle
                for bundle in self.audit_bundles
                if bundle.run_id == str(requested)
            ),
            None,
        )


EvalDefinition = EvalCase | EvalScenario
