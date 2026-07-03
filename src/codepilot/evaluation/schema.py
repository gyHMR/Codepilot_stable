from __future__ import annotations

# 新手导读：schema.py 定义 benchmark、断言、实验配置等评估数据结构。
# 关注点：它是 evaluation 层的协议中心。

"""Evaluation v2 schema."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from codepilot.runtime.contracts import CreateAgentSessionOptions


EvalModule = Literal["planning", "context", "memory", "security", "tool"]
EvalCaseType = Literal["task", "scenario"]
EvalStepKind = Literal["prompt", "restart", "modify_file", "verify", "inspect"]
WorkspacePolicy = Literal["failed", "all", "none"]


@dataclass(frozen=True)
class EvalStep:
    kind: EvalStepKind
    text: str = ""
    path: str = ""
    source: str = ""
    content: str | None = None
    check: dict[str, Any] | None = None


@dataclass(frozen=True)
class EvalCheck:
    kind: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalCase:
    id: str
    module: EvalModule
    fixture: str
    type: EvalCaseType
    prompt: str = ""
    steps: list[EvalStep] = field(default_factory=list)
    setup: list[EvalStep] = field(default_factory=list)
    checks: list[EvalCheck] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    expected: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    timeout_seconds: int = 120


@dataclass(frozen=True)
class MetricScore:
    name: str
    value: float | int | None
    numerator: float | int = 0
    denominator: float | int = 0
    display: str = ""

    def __post_init__(self) -> None:
        if not self.display:
            object.__setattr__(self, "display", _display(self.value, self.name))


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    summary: str
    expected: Any = None
    actual: Any = None


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    module: str
    passed: bool
    metrics: dict[str, MetricScore] = field(default_factory=dict)
    checks: list[CheckResult] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)
    error: str | None = None
    duration_ms: int | None = None


@dataclass(frozen=True)
class EvalSuiteResult:
    eval_id: str
    results: list[EvalResult]
    artifact_dir: str
    summary: dict[str, Any]


@dataclass(frozen=True)
class EvalRunOptions:
    fixtures_root: Path | str
    session_options: CreateAgentSessionOptions
    artifact_root: Path | str = ".codepilot/evals"
    eval_id: str | None = None
    include_tags: list[str] = field(default_factory=list)
    workspace_policy: WorkspacePolicy = "failed"
    runtime_overrides: dict[str, bool] = field(default_factory=dict)


def _display(value: float | int | None, name: str) -> str:
    if value is None:
        return "N/A"
    if name.endswith("_count") or name.endswith("_calls"):
        return f"{value:.2f}" if isinstance(value, float) else str(value)
    return f"{float(value):.1%}"


__all__ = [
    "CheckResult",
    "EvalCase",
    "EvalCaseType",
    "EvalCheck",
    "EvalModule",
    "EvalResult",
    "EvalRunOptions",
    "EvalSuiteResult",
    "EvalStep",
    "EvalStepKind",
    "MetricScore",
    "WorkspacePolicy",
]
