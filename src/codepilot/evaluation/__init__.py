"""Codepilot 评估与轻量级实验框架。"""

from .assertions import (
    build_dimension_results,
    failure_categories,
    run_assertions,
)
from .loader import (
    EvalCaseValidationError,
    load_eval_definition,
    load_eval_suite,
    parse_eval_definition,
)
from .report import build_suite_summary, render_suite_markdown
from .service import EvaluationService
from .types import (
    AssertionResult,
    AssertionSpec,
    DimensionResult,
    EvalBudgets,
    EvalCase,
    EvalResult,
    EvalRunOptions,
    EvalRuntimeProfile,
    EvalScenario,
    EvalSuiteResult,
    ScenarioStep,
)

__all__ = [
    "AssertionResult",
    "AssertionSpec",
    "DimensionResult",
    "EvalBudgets",
    "EvalCase",
    "EvalCaseValidationError",
    "EvalResult",
    "EvalRunOptions",
    "EvalRuntimeProfile",
    "EvalScenario",
    "EvalSuiteResult",
    "EvaluationService",
    "ScenarioStep",
    "build_dimension_results",
    "build_suite_summary",
    "failure_categories",
    "load_eval_definition",
    "load_eval_suite",
    "parse_eval_definition",
    "render_suite_markdown",
    "run_assertions",
]
