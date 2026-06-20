from __future__ import annotations

"""一次 Agent Run 的轻量级任务规划状态。"""

from dataclasses import dataclass, field
from typing import Literal


# 任务步骤状态：待处理 / 进行中 / 已完成 / 已阻塞
TaskStepStatus = Literal["pending", "in_progress", "completed", "blocked"]
# 任务阶段：理解中 / 执行中 / 验证中 / 等待中 / 已完成
TaskPhase = Literal["understanding", "acting", "verifying", "waiting", "finished"]
# 执行决策动作：继续 / 修复 / 重新规划 / 等待审批 / 完成 / 停止
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
    """单个任务步骤：包含标题、状态、证据引用和失败计数。"""
    id: str                          # 步骤唯一标识
    title: str                       # 步骤标题/描述
    status: TaskStepStatus = "pending"  # 当前状态
    evidence_refs: list[str] = field(default_factory=list)  # 证据引用（工具调用ID、文件路径等）
    failure_count: int = 0           # 失败次数
    note: str | None = None          # 备注信息


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
    "TaskPhase",
    "TaskState",
    "TaskStep",
    "TaskStepStatus",
]
