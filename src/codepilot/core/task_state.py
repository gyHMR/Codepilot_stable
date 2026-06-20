from __future__ import annotations

"""Lightweight task planning state for one Agent Run."""

from dataclasses import dataclass, field
from typing import Literal


TaskStepStatus = Literal["pending", "in_progress", "completed", "blocked"]
TaskPhase = Literal["understanding", "acting", "verifying", "waiting", "finished"]
ExecutionAction = Literal[
    "continue",
    "repair",
    "replan",
    "wait_approval",
    "finish",
    "stop",
]


@dataclass
class TaskStep:
    id: str
    title: str
    status: TaskStepStatus = "pending"
    evidence_refs: list[str] = field(default_factory=list)
    failure_count: int = 0
    note: str | None = None


@dataclass
class TaskState:
    task_id: str
    goal: str
    constraints: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    steps: list[TaskStep] = field(default_factory=list)
    current_step_id: str | None = None
    phase: TaskPhase = "understanding"
    next_action: str | None = None
    replan_count: int = 0
    max_replans_per_run: int = 2
    completion_satisfied: bool = False
    completion_reason: str = ""
    completion_prompt_count: int = 0


@dataclass(frozen=True)
class ExecutionDecision:
    action: ExecutionAction
    reason: str
    next_action: str | None = None


@dataclass(frozen=True)
class CompletionCheck:
    satisfied: bool
    reason: str
    missing: list[str] = field(default_factory=list)
    can_continue: bool = False
    unverified: bool = False


__all__ = [
    "CompletionCheck",
    "ExecutionAction",
    "ExecutionDecision",
    "TaskPhase",
    "TaskState",
    "TaskStep",
    "TaskStepStatus",
]
