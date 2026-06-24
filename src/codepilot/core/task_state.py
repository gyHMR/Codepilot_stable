from __future__ import annotations

"""一次 Agent Run 的轻量级任务规划状态。"""

from dataclasses import dataclass, field
from typing import Literal


# 任务步骤状态：待处理 / 进行中 / 已完成 / 已阻塞
TaskStepStatus = Literal["pending", "in_progress", "completed", "blocked"]
# 任务步骤内的轻量进展状态：避免把“工具执行成功”直接等同于完成
TaskProgressState = Literal[
    "none",
    "evidence_collected",
    "changed",
    "verified",
]
# 任务阶段：理解中 / 执行中 / 验证中 / 等待中 / 已完成
TaskPhase = Literal["understanding", "acting", "verifying", "waiting", "finished"]
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


@dataclass
class TaskStep:
    """单个任务步骤：包含标题、状态、证据引用和失败计数。"""
    id: str                          # 步骤唯一标识
    title: str                       # 步骤标题/描述
    status: TaskStepStatus = "pending"  # 当前状态
    kind: str = "other"              # 步骤类型：investigate/edit/verify/summarize/other
    acceptance: str | None = None    # 当前步骤的完成标准
    verification_hint: str | None = None  # 建议验证方式
    summary: str | None = None       # 步骤完成摘要
    evidence_refs: list[str] = field(default_factory=list)  # 证据引用（工具调用ID、文件路径等）
    failure_count: int = 0           # 失败次数
    note: str | None = None          # 备注信息
    progress_state: TaskProgressState = "none"  # 步骤内证据/变更/验证状态


@dataclass
class AttemptRecord:
    """一次工具行动尝试，用于把工具结果归因到任务步骤。"""
    attempt_id: str
    step_id: str | None
    action_intent: str
    tool_call_ids: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    status: str = "running"
    failure_type: str | None = None
    failure_reason: str | None = None


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
    status: str = "pending"
    verification_refs: list[str] = field(default_factory=list)


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


@dataclass(frozen=True)
class ExecutionDecision:
    """执行决策：由 TaskController 在工具执行后返回，指导循环下一步动作。"""
    action: ExecutionAction       # 决策动作
    reason: str                   # 决策原因
    next_action: str | None = None  # 下一步动作描述


@dataclass(frozen=True)
class CompletionCheck:
    """完成检查结果：判断任务是否已完成，以及缺失的条件。"""
    satisfied: bool               # 是否满足完成条件
    reason: str                   # 检查结果原因
    missing: list[str] = field(default_factory=list)  # 缺失的完成条件列表
    can_continue: bool = False    # 是否可以继续尝试完成
    unverified: bool = False      # 是否存在未验证的变更


__all__ = [
    "CompletionCheck",
    "ExecutionAction",
    "ExecutionDecision",
    "AttemptRecord",
    "ChangeSet",
    "ReplanRecord",
    "TaskProgressState",
    "TaskPhase",
    "TaskState",
    "TaskStep",
    "TaskStepStatus",
]
