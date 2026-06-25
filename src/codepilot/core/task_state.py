from __future__ import annotations

"""一次 Agent Run 的轻量级任务规划状态。"""

from dataclasses import dataclass, field
from typing import Literal, cast


# 任务步骤状态：待处理 / 进行中 / 已完成 / 已阻塞
TaskStepStatus = Literal["pending", "in_progress", "completed", "blocked"]
# 任务步骤类型：帮助上下文和报告解释这一步为什么存在
TaskStepKind = Literal["investigate", "edit", "verify", "summarize", "other"]
# 任务步骤内的轻量进展状态：避免把“工具执行成功”直接等同于完成
TaskProgressState = Literal[
    "none",
    "evidence_collected",
    "changed",
    "verified",
]
# 任务阶段：理解中 / 执行中 / 验证中 / 等待中 / 已完成
TaskPhase = Literal["understanding", "acting", "verifying", "waiting", "finished"]
# 一次工具尝试的状态：运行中 / 成功 / 失败
AttemptStatus = Literal["running", "succeeded", "failed"]
# 文件变更证据集合的验证状态
ChangeSetStatus = Literal["pending", "verified", "failed", "revert_required"]
# 执行决策动作：继续 / 修复 / 重新规划 / 等待审批 / 完成 / 停止
ExecutionAction = Literal[
    "continue",
    "repair",
    "replan",
    "propose_revert",
    "wait_approval",
    "finish",
    "stop",
]
# 完成门禁原因：解释本次任务是否满足完成条件
CompletionReason = Literal[
    "blocked_steps",
    "modified_without_fresh_verification",
    "incomplete_steps",
    "all_steps_completed",
]
TASK_STEP_KINDS = frozenset({"investigate", "edit", "verify", "summarize", "other"})
_TASK_STEP_STATUSES = frozenset({"pending", "in_progress", "completed", "blocked"})
_TASK_PROGRESS_STATES = frozenset(
    {"none", "evidence_collected", "changed", "verified"}
)
_TASK_PHASES = frozenset({"understanding", "acting", "verifying", "waiting", "finished"})
_ATTEMPT_STATUSES = frozenset({"running", "succeeded", "failed"})
_CHANGE_SET_STATUSES = frozenset({"pending", "verified", "failed", "revert_required"})
_EXECUTION_ACTIONS = frozenset(
    {"continue", "repair", "replan", "propose_revert", "wait_approval", "finish", "stop"}
)
_COMPLETION_REASONS = frozenset(
    {
        "blocked_steps",
        "modified_without_fresh_verification",
        "incomplete_steps",
        "all_steps_completed",
    }
)


@dataclass
class TaskStep:
    """单个任务步骤：包含标题、状态、证据引用和失败计数。

    ``TaskController`` 负责决定“为什么”进入某个状态；
    ``TaskStep`` 负责保证状态迁移本身一致，例如证据去重、失败计数递增、
    完成后清理阻塞备注等。
    """
    id: str                          # 步骤唯一标识
    title: str                       # 步骤标题/描述
    status: TaskStepStatus = "pending"  # 当前状态
    kind: TaskStepKind = "other"     # 步骤类型：investigate/edit/verify/summarize/other
    acceptance: str | None = None    # 当前步骤的完成标准
    verification_hint: str | None = None  # 建议验证方式
    summary: str | None = None       # 步骤完成摘要
    evidence_refs: list[str] = field(default_factory=list)  # 证据引用（工具调用ID、文件路径等）
    failure_count: int = 0           # 失败次数
    note: str | None = None          # 备注信息
    progress_state: TaskProgressState = "none"  # 步骤内证据/变更/验证状态

    def __post_init__(self) -> None:
        _ensure_task_step_status(self.status)
        _ensure_task_step_kind(self.kind)
        _ensure_task_progress_state(self.progress_state)

    def add_evidence_refs(self, refs: list[str]) -> None:
        """追加证据引用，并保持插入顺序去重。"""

        for ref in refs:
            if ref and ref not in self.evidence_refs:
                self.evidence_refs.append(ref)

    def mark_in_progress(self) -> None:
        """将步骤标记为正在执行。"""

        self.status = "in_progress"

    def block(
        self,
        note: str,
        *,
        evidence_refs: list[str] | None = None,
    ) -> None:
        """将步骤标记为阻塞，并保留阻塞原因和证据。"""

        self.status = "blocked"
        self.note = note
        self.add_evidence_refs(evidence_refs or [])

    def record_failure(
        self,
        note: str,
        *,
        evidence_refs: list[str] | None = None,
    ) -> None:
        """记录一次失败，但保持步骤可继续修复。"""

        self.failure_count += 1
        self.status = "in_progress"
        self.note = note
        self.add_evidence_refs(evidence_refs or [])

    def complete(
        self,
        *,
        summary: str | None = None,
        evidence_refs: list[str] | None = None,
        progress_state: TaskProgressState | None = None,
    ) -> None:
        """完成步骤，并清理只对未完成状态有意义的备注。"""

        self.status = "completed"
        self.note = None
        if summary:
            self.summary = summary
        if progress_state is not None:
            self.progress_state = _ensure_task_progress_state(progress_state)
        self.add_evidence_refs(evidence_refs or [])


@dataclass
class AttemptRecord:
    """一次工具行动尝试，用于把工具结果归因到任务步骤。"""
    attempt_id: str
    step_id: str | None
    action_intent: str
    tool_call_ids: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    status: AttemptStatus = "running"
    failure_type: str | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        _ensure_attempt_status(self.status)


@dataclass
class ChangeSet:
    """一次文件变更证据集合（第一版只记录 hash 和路径，不做自动回滚）。"""
    change_id: str
    attempt_id: str
    step_id: str | None
    affected_paths: list[str] = field(default_factory=list)
    before_hashes: dict[str, str] = field(default_factory=dict)
    after_hashes: dict[str, str] = field(default_factory=dict)
    tool_call_ids: list[str] = field(default_factory=list)
    diff_summary: str | None = None
    status: ChangeSetStatus = "pending"
    verification_refs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _ensure_change_set_status(self.status)


@dataclass
class ReplanRecord:
    """一次基于证据的重新规划记录。"""
    replan_id: str
    trigger: str
    failed_attempt_id: str | None = None
    abandoned_strategy: str | None = None
    new_strategy: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    requires_revert: bool = False
    rollback_targets: list[str] = field(default_factory=list)


@dataclass
class TaskState:
    """任务状态：跟踪一个 Agent Run 的任务规划与执行进度。"""
    task_id: str                     # 任务唯一标识
    goal: str                        # 任务目标
    constraints: list[str] = field(default_factory=list)        # 约束条件
    acceptance_criteria: list[str] = field(default_factory=list) # 验收标准
    steps: list[TaskStep] = field(default_factory=list)         # 任务步骤列表
    current_step_id: str | None = None  # 当前正在执行的步骤 ID
    phase: TaskPhase = "understanding"  # 当前任务阶段
    next_action: str | None = None      # 下一步动作描述
    replan_count: int = 0               # 已重新规划次数
    max_replans_per_run: int = 2        # 单次运行最大重新规划次数
    completion_satisfied: bool = False  # 任务是否满足完成条件
    completion_reason: str = ""         # 完成/未完成原因
    completion_prompt_count: int = 0    # 完成提示次数（用于控制重复提示）
    action_intent: str | None = None
    recent_error_code: str | None = None
    recent_failure_type: str | None = None
    last_decision: str | None = None
    rollback_required: bool = False
    rollback_targets: list[str] = field(default_factory=list)
    attempts: list[AttemptRecord] = field(default_factory=list)
    change_sets: list[ChangeSet] = field(default_factory=list)
    replans: list[ReplanRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        _ensure_task_phase(self.phase)

    def current_step(self) -> TaskStep | None:
        """Return the step identified by ``current_step_id`` if it still exists."""

        if self.current_step_id is None:
            return None
        return next(
            (step for step in self.steps if step.id == self.current_step_id),
            None,
        )

    def first_open_step(self) -> TaskStep | None:
        """Return the first step that can still be worked on."""

        return next(
            (step for step in self.steps if step.status in {"pending", "in_progress"}),
            None,
        )

    def advance_to_next_open_step(self) -> TaskStep | None:
        """Move the task pointer to the next pending or in-progress step."""

        next_step = self.first_open_step()
        if next_step is None:
            self.current_step_id = None
            self.next_action = None
            return None
        next_step.mark_in_progress()
        self.current_step_id = next_step.id
        self.next_action = next_step.title
        return next_step

    def completed_step_titles(self) -> list[str]:
        """Return completed step titles in plan order."""

        return [step.title for step in self.steps if step.status == "completed"]

    def pending_step_titles(self) -> list[str]:
        """Return titles for steps that still require work."""

        return [
            step.title
            for step in self.steps
            if step.status in {"pending", "in_progress"}
        ]

    def blocked_step_titles(self) -> list[str]:
        """Return blocked step titles in plan order."""

        return [step.title for step in self.steps if step.status == "blocked"]


@dataclass(frozen=True)
class ExecutionDecision:
    """执行决策：由 TaskController 在工具执行后返回，指导循环下一步动作。"""
    action: ExecutionAction       # 决策动作
    reason: str                   # 决策原因
    next_action: str | None = None  # 下一步动作描述

    def __post_init__(self) -> None:
        _ensure_execution_action(self.action)


@dataclass(frozen=True)
class CompletionCheck:
    """完成检查结果：判断任务是否已完成，以及缺失的条件。"""
    satisfied: bool               # 是否满足完成条件
    reason: CompletionReason      # 检查结果原因
    missing: list[str] = field(default_factory=list)  # 缺失的完成条件列表
    can_continue: bool = False    # 是否可以继续尝试完成
    unverified: bool = False      # 是否存在未验证的变更

    def __post_init__(self) -> None:
        _ensure_completion_reason(self.reason)


def _ensure_task_step_status(value: object) -> TaskStepStatus:
    if value not in _TASK_STEP_STATUSES:
        raise ValueError(f"Unknown task step status: {value}")
    return cast(TaskStepStatus, value)


def _ensure_task_step_kind(value: object) -> TaskStepKind:
    if value not in TASK_STEP_KINDS:
        raise ValueError(f"Unknown task step kind: {value}")
    return cast(TaskStepKind, value)


def _ensure_task_progress_state(value: object) -> TaskProgressState:
    if value not in _TASK_PROGRESS_STATES:
        raise ValueError(f"Unknown task progress state: {value}")
    return cast(TaskProgressState, value)


def _ensure_task_phase(value: object) -> TaskPhase:
    if value not in _TASK_PHASES:
        raise ValueError(f"Unknown task phase: {value}")
    return cast(TaskPhase, value)


def _ensure_attempt_status(value: object) -> AttemptStatus:
    if value not in _ATTEMPT_STATUSES:
        raise ValueError(f"Unknown attempt status: {value}")
    return cast(AttemptStatus, value)


def _ensure_change_set_status(value: object) -> ChangeSetStatus:
    if value not in _CHANGE_SET_STATUSES:
        raise ValueError(f"Unknown change set status: {value}")
    return cast(ChangeSetStatus, value)


def _ensure_execution_action(value: object) -> ExecutionAction:
    if value not in _EXECUTION_ACTIONS:
        raise ValueError(f"Unknown execution action: {value}")
    return cast(ExecutionAction, value)


def _ensure_completion_reason(value: object) -> CompletionReason:
    if value not in _COMPLETION_REASONS:
        raise ValueError(f"Unknown completion reason: {value}")
    return cast(CompletionReason, value)


__all__ = [
    "CompletionCheck",
    "CompletionReason",
    "AttemptStatus",
    "ChangeSetStatus",
    "ExecutionAction",
    "ExecutionDecision",
    "AttemptRecord",
    "ChangeSet",
    "ReplanRecord",
    "TASK_STEP_KINDS",
    "TaskStepKind",
    "TaskProgressState",
    "TaskPhase",
    "TaskState",
    "TaskStep",
    "TaskStepStatus",
]
