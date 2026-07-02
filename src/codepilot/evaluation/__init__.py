"""Codepilot evaluation v2."""

from .artifacts import EvaluationArtifacts
from .evidence import (
    ContextEvidence,
    EvalEvidence,
    TaskStepEvidence,
    ToolCallEvidence,
    evidence_from_traces,
)
from .experiments import experiment_variants, run_context_ab, run_security_ab
from .loader import EvalCaseValidationError, load_eval_case, load_eval_suite
from .reports import build_summary, render_markdown
from .runner import EvaluationRunner
from .schema import (
    CheckResult,
    EvalCase,
    EvalCheck,
    EvalResult,
    EvalRunOptions,
    EvalStep,
    EvalSuiteResult,
    MetricScore,
)
from .scorers import score_metrics
from .service import EvaluationService

__all__ = [
    "CheckResult",
    "ContextEvidence",
    "EvalCase",
    "EvalCaseValidationError",
    "EvalCheck",
    "EvalEvidence",
    "EvalResult",
    "EvalRunOptions",
    "EvalStep",
    "EvalSuiteResult",
    "EvaluationArtifacts",
    "EvaluationRunner",
    "EvaluationService",
    "MetricScore",
    "TaskStepEvidence",
    "ToolCallEvidence",
    "build_summary",
    "evidence_from_traces",
    "experiment_variants",
    "load_eval_case",
    "load_eval_suite",
    "render_markdown",
    "run_context_ab",
    "run_security_ab",
    "score_metrics",
]
