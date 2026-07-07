from __future__ import annotations

"""Small task state used by the agent loop.

The task module is intentionally plain: a task has a goal, a short list of
steps, one current step, and a completion check.  Session memory, rollback
history, and long-term context governance live outside core.
"""

from dataclasses import dataclass, field
from typing import Literal, cast

from .contracts import TaskMode, TaskPlanningState, ensure_task_mode


TaskStepStatus = Literal["pending", "in_progress", "completed", "blocked"]
TaskStepKind = Literal["read", "plan", "edit", "verify", "summarize", "other"]
TaskPhase = Literal["acting", "verifying", "waiting", "finished"]
ExecutionAction = Literal["continue", "wait_approval", "stop"]
CompletionReason = Literal[
    "all_steps_completed",
    "blocked_steps",
    "modified_without_fresh_verification",
    "incomplete_steps",
]

TASK_STEP_KINDS = frozenset({"read", "plan", "edit", "verify", "summarize", "other"})
TASK_STEP_STATUSES = frozenset({"pending", "in_progress", "completed", "blocked"})
TASK_PHASES = frozenset({"acting", "verifying", "waiting", "finished"})
EXECUTION_ACTIONS = frozenset({"continue", "wait_approval", "stop"})
COMPLETION_REASONS = frozenset(
    {
        "all_steps_completed",
        "blocked_steps",
        "modified_without_fresh_verification",
        "incomplete_steps",
    }
)


@dataclass
class TaskStep:
    id: str
    title: str
    kind: TaskStepKind = "other"
    status: TaskStepStatus = "pending"
    acceptance: str | None = None
    verification_hint: str | None = None
    summary: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    failure_count: int = 0
    note: str | None = None

    def __post_init__(self) -> None:
        self.id = _required_text(self.id, "step id", limit=80)
        self.title = _required_text(self.title, "step title", limit=120)
        self.kind = ensure_task_step_kind(self.kind)
        self.status = ensure_task_step_status(self.status)
        self.acceptance = _optional_text(self.acceptance, limit=240)
        self.verification_hint = _optional_text(self.verification_hint, limit=240)
        self.summary = _optional_text(self.summary, limit=240)
        self.note = _optional_text(self.note, limit=240)
        self.failure_count = max(0, int(self.failure_count or 0))
        self.evidence_refs = _unique_texts(self.evidence_refs, limit=160)

    def start(self) -> None:
        if self.status == "pending":
            self.status = "in_progress"

    def add_evidence(self, refs: list[str]) -> None:
        for ref in _unique_texts(refs, limit=160):
            if ref not in self.evidence_refs:
                self.evidence_refs.append(ref)

    def complete(
        self,
        *,
        summary: str | None = None,
        evidence_refs: list[str] | None = None,
    ) -> None:
        self.status = "completed"
        self.note = None
        self.summary = _optional_text(summary, limit=240) or self.summary
        self.add_evidence(evidence_refs or [])

    def block(
        self,
        note: str,
        *,
        evidence_refs: list[str] | None = None,
    ) -> None:
        self.status = "blocked"
        self.note = _optional_text(note, limit=240)
        self.add_evidence(evidence_refs or [])

    def record_failure(
        self,
        note: str,
        *,
        evidence_refs: list[str] | None = None,
    ) -> None:
        self.failure_count += 1
        self.status = "in_progress"
        self.note = _optional_text(note, limit=240)
        self.add_evidence(evidence_refs or [])


@dataclass
class TaskState:
    task_id: str
    goal: str
    steps: list[TaskStep] = field(default_factory=list)
    mode: TaskMode = "build"
    planning: TaskPlanningState = field(default_factory=TaskPlanningState)
    current_step_id: str | None = None
    phase: TaskPhase = "acting"
    next_action: str | None = None
    completion_satisfied: bool = False
    completion_reason: str = ""
    completion_prompt_count: int = 0
    recent_error_code: str | None = None
    last_decision: str | None = None

    def __post_init__(self) -> None:
        self.task_id = _required_text(self.task_id, "task id", limit=120)
        self.goal = _required_text(self.goal, "task goal", limit=1200)
        self.mode = ensure_task_mode(self.mode)
        if not isinstance(self.planning, TaskPlanningState):
            self.planning = TaskPlanningState()
        self.phase = ensure_task_phase(self.phase)
        self.next_action = _optional_text(self.next_action, limit=240)
        self.recent_error_code = _optional_text(self.recent_error_code, limit=120)
        self.completion_reason = _clean_text(self.completion_reason, limit=120)
        self.steps = [step if isinstance(step, TaskStep) else TaskStep(**step) for step in self.steps]
        if self.steps and self.current_step_id is None:
            self.advance()

    def current_step(self) -> TaskStep | None:
        if self.current_step_id is None:
            return None
        return next((step for step in self.steps if step.id == self.current_step_id), None)

    def open_steps(self) -> list[TaskStep]:
        return [step for step in self.steps if step.status in {"pending", "in_progress"}]

    def blocked_steps(self) -> list[TaskStep]:
        return [step for step in self.steps if step.status == "blocked"]

    def advance(self) -> TaskStep | None:
        step = next((item for item in self.open_steps()), None)
        if step is None:
            self.current_step_id = None
            self.next_action = None
            self.phase = "finished"
            return None
        step.start()
        self.current_step_id = step.id
        self.next_action = step.title
        if self.phase == "finished":
            self.phase = "acting"
        return step

    def mark_finished(self) -> None:
        self.current_step_id = None
        self.next_action = None
        self.phase = "finished"

    def completed_step_titles(self) -> list[str]:
        return [step.title for step in self.steps if step.status == "completed"]

    def pending_step_titles(self) -> list[str]:
        return [step.title for step in self.open_steps()]

    def blocked_step_titles(self) -> list[str]:
        return [step.title for step in self.blocked_steps()]


@dataclass(frozen=True)
class ExecutionDecision:
    action: ExecutionAction
    reason: str
    next_action: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", ensure_execution_action(self.action))
        object.__setattr__(self, "reason", _clean_text(self.reason, limit=120))
        object.__setattr__(self, "next_action", _optional_text(self.next_action, limit=240))


@dataclass(frozen=True)
class CompletionCheck:
    satisfied: bool
    reason: CompletionReason
    missing: list[str] = field(default_factory=list)
    can_continue: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "satisfied", bool(self.satisfied))
        object.__setattr__(self, "reason", ensure_completion_reason(self.reason))
        object.__setattr__(self, "missing", _unique_texts(self.missing, limit=160))
        object.__setattr__(self, "can_continue", bool(self.can_continue))


def ensure_task_step_status(value: object) -> TaskStepStatus:
    if value not in TASK_STEP_STATUSES:
        raise ValueError(f"Unknown task step status: {value}")
    return cast(TaskStepStatus, value)


def ensure_task_step_kind(value: object) -> TaskStepKind:
    if value not in TASK_STEP_KINDS:
        raise ValueError(f"Unknown task step kind: {value}")
    return cast(TaskStepKind, value)


def ensure_task_phase(value: object) -> TaskPhase:
    if value not in TASK_PHASES:
        raise ValueError(f"Unknown task phase: {value}")
    return cast(TaskPhase, value)


def ensure_execution_action(value: object) -> ExecutionAction:
    if value not in EXECUTION_ACTIONS:
        raise ValueError(f"Unknown execution action: {value}")
    return cast(ExecutionAction, value)


def ensure_completion_reason(value: object) -> CompletionReason:
    if value not in COMPLETION_REASONS:
        raise ValueError(f"Unknown completion reason: {value}")
    return cast(CompletionReason, value)


def _required_text(value: object, field_name: str, *, limit: int) -> str:
    text = _clean_text(value, limit=limit)
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text


def _optional_text(value: object, *, limit: int) -> str | None:
    return _clean_text(value, limit=limit) or None


def _clean_text(value: object, *, limit: int) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())[:limit]


def _unique_texts(items: object, *, limit: int) -> list[str]:
    if not isinstance(items, list | tuple):
        return []
    seen: set[str] = set()
    values: list[str] = []
    for item in items:
        text = _clean_text(item, limit=limit)
        if text and text not in seen:
            seen.add(text)
            values.append(text)
    return values


__all__ = [
    "CompletionCheck",
    "CompletionReason",
    "ExecutionAction",
    "ExecutionDecision",
    "TASK_STEP_KINDS",
    "TaskPhase",
    "TaskState",
    "TaskStep",
    "TaskStepKind",
    "TaskStepStatus",
    "ensure_completion_reason",
    "ensure_execution_action",
    "ensure_task_phase",
    "ensure_task_step_kind",
    "ensure_task_step_status",
]
