"""Codepilot evaluation framework."""

from .loader import (
    EvalCaseValidationError,
    load_eval_definition,
    load_eval_suite,
    parse_eval_definition,
)
from .report import build_suite_summary
from .service import EvaluationService
from .types import (
    EvalCase,
    EvalResult,
    EvalRunOptions,
    EvalScenario,
    EvalSuiteResult,
    ScenarioStep,
    VerifierResult,
    VerifierSpec,
)

__all__ = [
    "EvalCase",
    "EvalCaseValidationError",
    "EvalResult",
    "EvalRunOptions",
    "EvalScenario",
    "EvalSuiteResult",
    "EvaluationService",
    "ScenarioStep",
    "VerifierResult",
    "VerifierSpec",
    "build_suite_summary",
    "load_eval_definition",
    "load_eval_suite",
    "parse_eval_definition",
]
