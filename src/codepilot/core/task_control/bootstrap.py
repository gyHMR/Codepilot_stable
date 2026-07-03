from __future__ import annotations

# 新手导读：本文件把用户输入、规划器输出和恢复投影整理成 TaskController 可用的初始状态。
# 关注点：关注它如何在新任务和恢复任务之间选择初始化路径。

"""Plan-mode bootstrap: discovery followed by plan synthesis."""

from dataclasses import dataclass
from typing import Any, Mapping

from ..events import AgentEventEmitter, maybe_await
from ..llm_runner import StreamFn
from ..types import AgentContext, AgentLoopConfig
from .contracts import (
    PlanningBudget,
    TaskPlanningState,
    budget_for_profile,
    planning_discovery_report_from_mapping,
    task_planning_state_from_mapping,
)
from .discovery import PlanningDiscovery
from .planner import TaskPlanDraft, TaskPlanner


@dataclass(frozen=True)
class PlanningBootstrapResult:
    plan: TaskPlanDraft | None
    planning: TaskPlanningState


class PlanningBootstrap:
    """Build the initial plan-mode state without leaking scratch messages."""

    async def run(
        self,
        context: AgentContext,
        *,
        config: AgentLoopConfig,
        emitter: AgentEventEmitter,
        stream_fn: StreamFn | None,
        fallback_goal: str,
        signal: Any | None = None,
    ) -> PlanningBootstrapResult:
        budget = budget_for_profile(config.planning_budget_profile)
        projection = (
            context.task_recovery_projection
            if isinstance(context.task_recovery_projection, Mapping)
            else None
        )
        if _has_task_progress(projection):
            return PlanningBootstrapResult(
                plan=None,
                planning=TaskPlanningState(phase="recovered", source="recovered"),
            )

        discovery_report = _recovered_discovery(projection)
        if discovery_report is None:
            discovery_report = await PlanningDiscovery().run(
                context,
                config=config,
                budget=budget,
                emitter=emitter,
                stream_fn=stream_fn,
                signal=signal,
            )

        await emitter.emit(
            {
                "type": "planning_synthesis_started",
                "mode": "plan",
                "planning": {
                    "phase": "synthesis",
                    "source": "default",
                    "budget": budget.to_signal(),
                    "discovery": discovery_report.to_signal(),
                },
            }
        )
        api_key = (
            await maybe_await(config.get_api_key(config.model.provider))
            if config.get_api_key is not None
            else None
        )
        plan = await TaskPlanner().generate(
            model=config.model,
            messages=context.messages,
            convert_to_llm=config.convert_to_llm,
            fallback_goal=fallback_goal,
            stream_fn=stream_fn,
            api_key=api_key,
            session_id=config.session_id,
            discovery_report=discovery_report,
        )
        planning = TaskPlanningState(
            phase="execution",
            source=plan.source,
            budget=budget,
            discovery=discovery_report,
            fallback_reason=plan.fallback_reason,
        )
        await emitter.emit(
            {
                "type": "planning_synthesis_completed",
                "mode": "plan",
                "planning": planning.to_signal(),
            }
        )
        return PlanningBootstrapResult(plan=plan, planning=planning)


def _has_task_progress(projection: Mapping[str, object] | None) -> bool:
    return isinstance(projection, Mapping) and isinstance(
        projection.get("task_progress"),
        Mapping,
    )


def _recovered_discovery(projection: Mapping[str, object] | None):
    if not isinstance(projection, Mapping):
        return None
    planning = projection.get("planning")
    if not isinstance(planning, Mapping):
        return None
    recovered = task_planning_state_from_mapping(planning)
    return recovered.discovery


__all__ = ["PlanningBootstrap", "PlanningBootstrapResult"]
