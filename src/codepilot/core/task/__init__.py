from __future__ import annotations

"""Task helpers exposed to runtime and sessions."""

from .contracts import (
    PlanSource,
    PlanningBudget,
    PlanningBudgetProfile,
    PlanningBudgetUsage,
    PlanningDiscoveryReport,
    PlanningPhase,
    PlanningStatus,
    TaskMode,
    TaskPlanningState,
    budget_for_profile,
    ensure_plan_source,
    ensure_planning_budget_profile,
    ensure_planning_phase,
    ensure_planning_status,
    ensure_task_mode,
    planning_budget_from_mapping,
    planning_budget_usage_from_mapping,
    planning_discovery_report_from_mapping,
    task_planning_state_from_mapping,
)
_LAZY_EXPORTS = {
    "TaskController": (".controller", "TaskController"),
    "build_task_state_from_payload": (".controller", "build_task_state_from_payload"),
    "TaskModePolicy": (".modes", "TaskModePolicy"),
    "policy_for_mode": (".modes", "policy_for_mode"),
    "PlannedTaskStep": (".planner", "PlannedTaskStep"),
    "TaskPlanDraft": (".planner", "TaskPlanDraft"),
    "TaskPlanner": (".planner", "TaskPlanner"),
    "CompletionCheck": (".state", "CompletionCheck"),
    "ExecutionDecision": (".state", "ExecutionDecision"),
    "TaskState": (".state", "TaskState"),
    "TaskStep": (".state", "TaskStep"),
    "COMPLETE_TASK_STEP_TOOL": (".tools", "COMPLETE_TASK_STEP_TOOL"),
    "TASK_UPDATE_TOOL": (".tools", "TASK_UPDATE_TOOL"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "PlanSource",
    "PlanningBudget",
    "PlanningBudgetProfile",
    "PlanningBudgetUsage",
    "PlanningDiscoveryReport",
    "PlanningPhase",
    "PlanningStatus",
    "TaskMode",
    "TaskPlanningState",
    "budget_for_profile",
    "ensure_plan_source",
    "ensure_planning_budget_profile",
    "ensure_planning_phase",
    "ensure_planning_status",
    "ensure_task_mode",
    "planning_budget_from_mapping",
    "planning_budget_usage_from_mapping",
    "planning_discovery_report_from_mapping",
    "task_planning_state_from_mapping",
    *_LAZY_EXPORTS,
]
