from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from codepilot.protocols import Message, TaskSummary, ToolCall, ToolResultMessage, UserMessage

from .run_state import RunState
from .task_control import (
    CompletionCheck,
    ExecutionDecision,
    TaskController,
    TaskPlanningState,
    budget_for_profile,
    policy_for_mode,
)
from .task_control.state import TaskState


@dataclass
class AgentTaskRuntime:
    """V2 core adapter for task-control semantics."""

    controller: TaskController
    task: TaskState
    run_state: RunState

    @classmethod
    def from_strategy(
        cls,
        *,
        strategy: dict[str, Any],
        messages: list[Message],
        run_id: str,
        session_id: str | None,
    ) -> "AgentTaskRuntime | None":
        if not bool(strategy.get("enabled", False)):
            return None
        mode = str(strategy.get("mode") or "edit")
        policy = policy_for_mode(mode)
        planning = strategy.get("planning")
        if planning is None and policy.planner_required:
            planning = TaskPlanningState(
                phase="execution",
                source="default",
                budget=budget_for_profile(
                    str(strategy.get("planning_budget_profile") or "balanced")
                ),
            )
        controller = TaskController()
        task = controller.initialize(
            messages,
            goal=_optional_text(strategy.get("goal")),
            proposed_steps=strategy.get("steps"),
            mode=policy.mode,
            planning=planning,
            max_replans_per_run=_optional_int(
                strategy.get("max_replans_per_run")
                or strategy.get("max_task_replans_per_run")
            ),
            task_recovery_projection=_optional_mapping(
                strategy.get("recovery_projection")
            ),
        )
        return cls(
            controller=controller,
            task=task,
            run_state=RunState(run_id=run_id, session_id=session_id),
        )

    def context_text(self) -> str:
        return self.controller.render_context(self.task)

    def event_payload(self) -> dict[str, object]:
        return self.controller.event_payload(self.task)

    def after_tool_results(
        self,
        results: list[ToolResultMessage],
    ) -> ExecutionDecision:
        self.run_state.collect_tool_results(results)
        return self.controller.after_tool_results(self.task, self.run_state, results)

    def snapshot(self) -> TaskSummary:
        return self.controller.summarize(self.task)

    def complete(self) -> tuple[TaskSummary, CompletionCheck]:
        check = self.controller.check_completion(self.task, self.run_state)
        return self.controller.summarize(self.task), check

    def completion_steering(self, check: CompletionCheck) -> UserMessage:
        return self.controller.completion_steering(check)

    def needs_final_verification_grace(self, tool_calls: list[ToolCall]) -> bool:
        return (
            self.run_state.workspace_changed
            and not self.run_state.fresh_verification_passed
            and any(_looks_like_verification_call(call) for call in tool_calls)
        )

    def merge_observation_summary(
        self,
        *,
        affected_paths: tuple[str, ...],
        workspace_changed: bool,
        verification: list[Any],
    ) -> None:
        self.run_state.affected_paths.update(affected_paths)
        self.run_state.workspace_changed = self.run_state.workspace_changed or workspace_changed
        self.run_state.verification = list(verification)
        self.run_state.fresh_verification_passed = any(
            getattr(item, "status", None) == "passed" for item in verification
        )


def with_task_context(
    context: dict[str, Any],
    runtime: AgentTaskRuntime | None,
) -> dict[str, Any]:
    if runtime is None:
        return context
    return {
        **context,
        "current_task": runtime.context_text(),
        "task_control_signal": runtime.controller.control_signal(runtime.task),
    }


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def _optional_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    return dict(value)


def _looks_like_verification_call(call: ToolCall) -> bool:
    name = call.name.lower()
    if any(marker in name for marker in ("test", "verify", "check", "pytest")):
        return True
    command = call.arguments.get("command") or call.arguments.get("cmd")
    if not isinstance(command, str):
        return False
    return any(marker in command.lower() for marker in ("pytest", "test", "compile", "lint"))


__all__ = ["AgentTaskRuntime", "with_task_context"]
