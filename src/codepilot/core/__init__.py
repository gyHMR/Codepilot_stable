from __future__ import annotations

# 新手导读：core 包门面只放 V2 agent loop 主入口和稳定的核心构件。
# 关注点：一次 run 从 AgentLoopInput 进入，通过 AgentLoopPorts 访问模型/工具/上下文，最终得到 AgentLoopOutcome。

"""
Codepilot core layer.

The default public path is the V2 agent-loop spine:

    AgentLoopInput + AgentLoopPorts -> run_agent_loop() -> AgentLoopOutcome

Old core loop entry points have been removed so newcomers start from the V2
execution contract.
"""

from .contracts import (
    AgentLoopInput,
    AgentLoopLimits,
    AgentLoopOutcome,
    AgentLoopPorts,
    AgentLoopStatus,
    AgentResumeInput,
    ContextPort,
    RunCorrelation,
    WorkspaceEffects,
)
from .events import AgentEventEmitter
from .loop import resume_agent_loop, run_agent_loop
from .message_conversion import convert_to_llm
from .run_state import RunState, new_run_id
from .task_control import (
    COMPLETE_TASK_STEP_TOOL,
    AttemptRecord,
    ChangeSet,
    CompletionCheck,
    ExecutionDecision,
    PlanSource,
    PlannedTaskStep,
    PlanningBudget,
    PlanningBudgetProfile,
    PlanningBudgetUsage,
    PlanningDiscoveryReport,
    PlanningPhase,
    PlanningStatus,
    TaskController,
    TaskMode,
    TaskModePolicy,
    TaskPlanDraft,
    TaskPlanner,
    TaskPlanningState,
    TaskState,
    TaskStep,
    build_task_state_from_recovery_projection,
    budget_for_profile,
    ensure_plan_source,
    ensure_planning_budget_profile,
    ensure_task_mode,
    policy_for_mode,
)
from .types import (
    AgentContext,
    AgentMessage,
    ContextPreparationRequest,
    PreparedAgentContext,
    PrepareContextFn,
    ToolExecutionMode,
)

__all__ = [
    "AgentLoopInput",
    "AgentLoopLimits",
    "AgentLoopOutcome",
    "AgentLoopPorts",
    "AgentLoopStatus",
    "AgentResumeInput",
    "ContextPort",
    "RunCorrelation",
    "WorkspaceEffects",
    "run_agent_loop",
    "resume_agent_loop",
    "AgentEventEmitter",
    "convert_to_llm",
    "RunState",
    "new_run_id",
    "TaskController",
    "PlanSource",
    "TaskMode",
    "TaskModePolicy",
    "PlanningBudget",
    "PlanningBudgetProfile",
    "PlanningBudgetUsage",
    "PlanningDiscoveryReport",
    "PlanningPhase",
    "PlanningStatus",
    "TaskPlanningState",
    "build_task_state_from_recovery_projection",
    "budget_for_profile",
    "ensure_plan_source",
    "ensure_planning_budget_profile",
    "ensure_task_mode",
    "policy_for_mode",
    "PlannedTaskStep",
    "TaskPlanDraft",
    "TaskPlanner",
    "COMPLETE_TASK_STEP_TOOL",
    "AttemptRecord",
    "ChangeSet",
    "CompletionCheck",
    "ExecutionDecision",
    "TaskState",
    "TaskStep",
    "AgentContext",
    "AgentMessage",
    "ContextPreparationRequest",
    "PreparedAgentContext",
    "PrepareContextFn",
    "ToolExecutionMode",
]
