from __future__ import annotations

# 新手导读：包门面文件：集中导出本层最常用的类型和入口，降低学习时的导入成本。
# 关注点：core 层负责 Agent 推理循环和工具调度，是“模型想做什么”到“工具怎么跑”的中枢。

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
    "build_task_state_from_recovery_projection": (
        ".controller",
        "build_task_state_from_recovery_projection",
    ),
    "PlanningBootstrap": (".bootstrap", "PlanningBootstrap"),
    "PlanningBootstrapResult": (".bootstrap", "PlanningBootstrapResult"),
    "TaskModePolicy": (".modes", "TaskModePolicy"),
    "policy_for_mode": (".modes", "policy_for_mode"),
    "PlannedTaskStep": (".planner", "PlannedTaskStep"),
    "TaskPlanDraft": (".planner", "TaskPlanDraft"),
    "TaskPlanner": (".planner", "TaskPlanner"),
    "AttemptRecord": (".state", "AttemptRecord"),
    "ChangeSet": (".state", "ChangeSet"),
    "CompletionCheck": (".state", "CompletionCheck"),
    "ExecutionDecision": (".state", "ExecutionDecision"),
    "TaskState": (".state", "TaskState"),
    "TaskStep": (".state", "TaskStep"),
    "COMPLETE_TASK_STEP_TOOL": (".tools", "COMPLETE_TASK_STEP_TOOL"),
    "complete_task_step_tool": (".tools", "complete_task_step_tool"),
    "has_complete_task_step_tool": (".tools", "has_complete_task_step_tool"),
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
