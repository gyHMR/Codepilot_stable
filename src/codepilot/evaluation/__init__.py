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
from .experiment import (
    build_experiment_comparison,
    experiment_variants,
    run_experiment,
)
from .metrics import calculate_case_metrics
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
    "build_experiment_comparison",
    "build_suite_summary",
    "calculate_case_metrics",
    "experiment_variants",
    "failure_categories",
    "load_eval_definition",
    "load_eval_suite",
    "parse_eval_definition",
    "render_suite_markdown",
    "run_assertions",
    "run_experiment",
]
