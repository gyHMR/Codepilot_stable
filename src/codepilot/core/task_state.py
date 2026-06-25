from __future__ import annotations

"""
一次 Agent Run 的轻量级任务规划状态模块。

本模块定义了任务控制系统的所有数据类型，用于跟踪 Agent 在一次运行中的
任务规划与执行进度。

核心设计：
    - TaskState: 任务的全局状态，包含目标、步骤列表、阶段等
    - TaskStep: 单个任务步骤，包含状态、证据引用、失败计数等
    - AttemptRecord: 工具执行尝试记录，用于归因和审计
    - ChangeSet: 文件变更证据集合，记录变更前后的 hash
    - ExecutionDecision: 任务控制器返回的执行决策
    - CompletionCheck: 任务完成度检查结果

状态流转：
    TaskPhase: understanding → acting → verifying → waiting → finished
    TaskStepStatus: pending → in_progress → completed / blocked
    TaskProgressState: none → evidence_collected → changed → verified
"""

from dataclasses import dataclass, field
from typing import Literal, cast


# ── 类型别名定义 ────────────────────────────────────────────────
# 这些 Literal 类型定义了任务控制系统中所有状态的合法值

# 任务步骤状态：待处理 / 进行中 / 已完成 / 已阻塞
TaskStepStatus = Literal["pending", "in_progress", "completed", "blocked"]

# 任务步骤类型：帮助上下文和报告解释这一步为什么存在
# - investigate: 调查/探索（如读取代码、搜索文件）
# - edit: 编辑/修改（如写入文件、应用补丁）
# - verify: 验证/测试（如运行测试、检查结果）
# - summarize: 总结/汇报（如生成报告、整理结论）
# - other: 其他类型
TaskStepKind = Literal["investigate", "edit", "verify", "summarize", "other"]

# 任务步骤内的轻量进展状态：避免把"工具执行成功"直接等同于完成
# - none: 无进展
# - evidence_collected: 已收集证据（如读取了相关文件）
# - changed: 已产生变更（如修改了文件）
# - verified: 已验证（如测试通过）
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
# - pending: 待验证
# - verified: 已验证通过
# - failed: 验证失败
# - revert_required: 需要回滚
ChangeSetStatus = Literal["pending", "verified", "failed", "revert_required"]

# 执行决策动作：继续 / 修复 / 重新规划 / 建议回滚 / 等待审批 / 完成 / 停止
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
    "blocked_steps",                        # 存在阻塞的步骤
    "modified_without_fresh_verification",  # 工作区已修改但未通过最新验证
    "incomplete_steps",                     # 存在未完成的步骤
    "all_steps_completed",                  # 所有步骤已完成
]

# ── 合法值集合（用于校验） ──────────────────────────────────────
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

    TaskController 负责决定"为什么"进入某个状态；
    TaskStep 负责保证状态迁移本身一致，例如证据去重、失败计数递增、
    完成后清理阻塞备注等。

    状态流转：
        pending → in_progress → completed（正常完成）
        pending → in_progress → blocked（遇到阻塞）
        in_progress → in_progress（记录失败后继续修复）

    Attributes:
        id: 步骤唯一标识（如 "step_1"、"step_2"）。
        title: 步骤标题/描述（如 "读取配置文件"、"修复 bug"）。
        status: 当前状态（pending/in_progress/completed/blocked）。
        kind: 步骤类型，帮助理解这一步为什么存在。
        acceptance: 当前步骤的完成标准（如 "所有测试通过"）。
        verification_hint: 建议验证方式（如 "运行 pytest"）。
        summary: 步骤完成摘要（如 "已修复登录逻辑"）。
        evidence_refs: 证据引用列表（工具调用ID、文件路径等）。
        failure_count: 失败次数，用于判断是否需要重新规划。
        note: 备注信息（如失败原因、阻塞说明）。
        progress_state: 步骤内进展状态（none/evidence_collected/changed/verified）。
    """
    id: str                                                                    # 步骤唯一标识
    title: str                                                                 # 步骤标题
    status: TaskStepStatus = "pending"                                         # 当前状态
    kind: TaskStepKind = "other"                                               # 步骤类型
    acceptance: str | None = None                                              # 完成标准
    verification_hint: str | None = None                                       # 验证方式提示
    summary: str | None = None                                                 # 完成摘要
    evidence_refs: list[str] = field(default_factory=list)                     # 证据引用列表
    failure_count: int = 0                                                     # 失败次数
    note: str | None = None                                                    # 备注
    progress_state: TaskProgressState = "none"                                 # 进展状态

    def __post_init__(self) -> None:
        """初始化后校验：确保状态、类型、进展状态的值合法。"""
        _ensure_task_step_status(self.status)
        _ensure_task_step_kind(self.kind)
        _ensure_task_progress_state(self.progress_state)
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
    """一次工具执行尝试记录：用于将工具结果归因到任务步骤。

    每次工具执行后，TaskController 会创建一个 AttemptRecord，
    记录这次尝试的意图、关联的工具调用、执行结果等信息。
    这些记录用于审计、调试和任务摘要生成。

    Attributes:
        attempt_id: 尝试唯一标识（如 "attempt_abc123"）。
        step_id: 关联的任务步骤 ID（可选）。
        action_intent: 操作意图（如 "edit_file"、"read_context"、"run_verification"）。
        tool_call_ids: 关联的工具调用 ID 列表。
        evidence_refs: 证据引用列表。
        status: 尝试状态（running/succeeded/failed）。
        failure_type: 失败类型（如 "tool_error"、"verification_failed"）。
        failure_reason: 失败原因描述。
    """
    attempt_id: str                                                            # 尝试唯一标识
    step_id: str | None                                                        # 关联步骤 ID
    action_intent: str                                                         # 操作意图
    tool_call_ids: list[str] = field(default_factory=list)                     # 工具调用 ID 列表
    evidence_refs: list[str] = field(default_factory=list)                     # 证据引用列表
    status: AttemptStatus = "running"                                          # 尝试状态
    failure_type: str | None = None                                            # 失败类型
    failure_reason: str | None = None                                          # 失败原因

    def __post_init__(self) -> None:
        """初始化后校验：确保尝试状态的值合法。"""
        _ensure_attempt_status(self.status)


@dataclass
class ChangeSet:
    """文件变更证据集合：记录一次工具执行产生的文件变更。

    第一版只记录 hash 和路径，不做自动回滚。
    用于跟踪哪些文件被修改、修改前后的状态，以及验证结果。

    Attributes:
        change_id: 变更集合唯一标识（如 "change_abc123"）。
        attempt_id: 关联的尝试 ID。
        step_id: 关联的任务步骤 ID。
        affected_paths: 受影响的文件路径列表。
        before_hashes: 变更前的文件 hash（路径 → hash）。
        after_hashes: 变更后的文件 hash（路径 → hash）。
        tool_call_ids: 关联的工具调用 ID 列表。
        diff_summary: 变更摘要（如 diff 输出）。
        status: 验证状态（pending/verified/failed/revert_required）。
        verification_refs: 验证引用列表。
    """
    change_id: str                                                             # 变更集合唯一标识
    attempt_id: str                                                            # 关联尝试 ID
    step_id: str | None                                                        # 关联步骤 ID
    affected_paths: list[str] = field(default_factory=list)                    # 受影响路径
    before_hashes: dict[str, str] = field(default_factory=dict)                # 变更前 hash
    after_hashes: dict[str, str] = field(default_factory=dict)                 # 变更后 hash
    tool_call_ids: list[str] = field(default_factory=list)                     # 工具调用 ID
    diff_summary: str | None = None                                            # 变更摘要
    status: ChangeSetStatus = "pending"                                        # 验证状态
    verification_refs: list[str] = field(default_factory=list)                 # 验证引用

    def __post_init__(self) -> None:
        """初始化后校验：确保变更状态的值合法。"""
        _ensure_change_set_status(self.status)


@dataclass
class ReplanRecord:
    """重新规划记录：记录一次基于证据的任务重新规划。

    当任务步骤连续失败时，TaskController 可能触发重新规划，
    保留已完成的步骤，替换当前和后续步骤。

    Attributes:
        replan_id: 重新规划唯一标识（如 "replan_abc123"）。
        trigger: 触发原因（如 "verification_failed"）。
        failed_attempt_id: 关联的失败尝试 ID。
        abandoned_strategy: 被放弃的策略描述。
        new_strategy: 新策略描述。
        evidence_refs: 触发重新规划的证据引用。
        requires_revert: 是否需要回滚之前的变更。
        rollback_targets: 需要回滚的文件路径列表。
    """
    replan_id: str                                                             # 重新规划唯一标识
    trigger: str                                                               # 触发原因
    failed_attempt_id: str | None = None                                       # 失败尝试 ID
    abandoned_strategy: str | None = None                                      # 被放弃的策略
    new_strategy: str = ""                                                     # 新策略
    evidence_refs: list[str] = field(default_factory=list)                     # 证据引用
    requires_revert: bool = False                                              # 是否需要回滚
    rollback_targets: list[str] = field(default_factory=list)                  # 回滚目标路径


@dataclass
class TaskState:
    """任务状态：跟踪一个 Agent Run 的任务规划与执行进度。

    这是任务控制系统的核心数据结构，由 TaskController 创建和维护。
    它记录了任务的目标、步骤列表、当前阶段、执行历史等完整信息。

    生命周期：
        1. TaskController.initialize() 创建 TaskState
        2. Agent 循环中通过 after_tool_results() 更新状态
        3. 通过 check_completion() 检查完成度
        4. 通过 summarize() 生成任务摘要

    Attributes:
        task_id: 任务唯一标识（如 "task_abc123"）。
        goal: 任务目标描述（如 "修复登录页面的 bug"）。
        constraints: 约束条件列表（如 "不能修改数据库 schema"）。
        acceptance_criteria: 验收标准列表（如 "所有测试通过"）。
        steps: 任务步骤列表，按顺序执行。
        current_step_id: 当前正在执行的步骤 ID。
        phase: 当前任务阶段（understanding/acting/verifying/waiting/finished）。
        next_action: 下一步动作描述（如 "读取配置文件"）。
        replan_count: 已重新规划次数。
        max_replans_per_run: 单次运行最大重新规划次数（防止无限重新规划）。
        completion_satisfied: 任务是否满足完成条件。
        completion_reason: 完成/未完成原因。
        completion_prompt_count: 完成提示次数（控制重复提示）。
        action_intent: 当前操作意图（如 "edit_file"、"read_context"）。
        recent_error_code: 最近的错误代码。
        recent_failure_type: 最近的失败类型。
        last_decision: 上一次执行决策。
        rollback_required: 是否需要回滚之前的变更。
        rollback_targets: 需要回滚的文件路径列表。
        attempts: 工具执行尝试记录列表。
        change_sets: 文件变更证据集合列表。
        replans: 重新规划记录列表。
    """
    task_id: str                                                                 # 任务唯一标识
    goal: str                                                                    # 任务目标
    constraints: list[str] = field(default_factory=list)                         # 约束条件
    acceptance_criteria: list[str] = field(default_factory=list)                 # 验收标准
    steps: list[TaskStep] = field(default_factory=list)                          # 步骤列表
    current_step_id: str | None = None                                           # 当前步骤 ID
    phase: TaskPhase = "understanding"                                           # 当前阶段
    next_action: str | None = None                                               # 下一步动作
    replan_count: int = 0                                                        # 重新规划次数
    max_replans_per_run: int = 2                                                 # 最大重新规划次数
    completion_satisfied: bool = False                                           # 是否满足完成条件
    completion_reason: str = ""                                                  # 完成原因
    completion_prompt_count: int = 0                                             # 完成提示次数
    action_intent: str | None = None                                             # 操作意图
    recent_error_code: str | None = None                                         # 最近错误代码
    recent_failure_type: str | None = None                                       # 最近失败类型
    last_decision: str | None = None                                             # 上次决策
    rollback_required: bool = False                                              # 是否需要回滚
    rollback_targets: list[str] = field(default_factory=list)                    # 回滚目标
    attempts: list[AttemptRecord] = field(default_factory=list)                  # 尝试记录
    change_sets: list[ChangeSet] = field(default_factory=list)                   # 变更集合
    replans: list[ReplanRecord] = field(default_factory=list)                    # 重新规划记录

    def __post_init__(self) -> None:
        _ensure_task_phase(self.phase)

    def current_step(self) -> TaskStep | None:
        """返回当前正在执行的步骤（通过 current_step_id 查找）。

        Returns:
            TaskStep | None: 当前步骤，如果 current_step_id 为 None 或步骤不存在则返回 None。
        """
        if self.current_step_id is None:
            return None
        return next(
            (step for step in self.steps if step.id == self.current_step_id),
            None,
        )

    def first_open_step(self) -> TaskStep | None:
        """返回第一个可继续执行的步骤（pending 或 in_progress 状态）。

        Returns:
            TaskStep | None: 第一个可执行的步骤，如果没有则返回 None。
        """
        return next(
            (step for step in self.steps if step.status in {"pending", "in_progress"}),
            None,
        )

    def advance_to_next_open_step(self) -> TaskStep | None:
        """将任务指针移动到下一个待执行的步骤。

        如果所有步骤都已完成，将 current_step_id 设为 None。
        否则，将下一个步骤标记为 in_progress 并更新 next_action。

        Returns:
            TaskStep | None: 下一个步骤，如果没有则返回 None。
        """
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
        """返回已完成步骤的标题列表（按计划顺序）。"""
        return [step.title for step in self.steps if step.status == "completed"]

    def pending_step_titles(self) -> list[str]:
        """返回待处理步骤的标题列表（pending 或 in_progress 状态）。"""
        return [
            step.title
            for step in self.steps
            if step.status in {"pending", "in_progress"}
        ]

    def blocked_step_titles(self) -> list[str]:
        """返回被阻塞步骤的标题列表。"""
        return [step.title for step in self.steps if step.status == "blocked"]


@dataclass(frozen=True)
class ExecutionDecision:
    """执行决策：由 TaskController 在工具执行后返回，指导 Agent 循环下一步动作。

    这是任务控制器与 Agent 循环之间的接口，告诉循环应该继续、修复、
    重新规划还是停止。

    Attributes:
        action: 决策动作（continue/repair/replan/propose_revert/wait_approval/finish/stop）。
        reason: 决策原因（如 "tool_error"、"verification_failed"、"all_steps_completed"）。
        next_action: 下一步动作描述（如 "修复失败的测试"）。
    """
    action: ExecutionAction                                                  # 决策动作
    reason: str                                                              # 决策原因
    next_action: str | None = None                                           # 下一步动作

    def __post_init__(self) -> None:
        """初始化后校验：确保决策动作的值合法。"""
        _ensure_execution_action(self.action)


@dataclass(frozen=True)
class CompletionCheck:
    """完成检查结果：判断任务是否已完成，以及缺失的条件。

    由 TaskController.check_completion() 返回，用于决定：
    - 任务已完成 → 正常结束
    - 任务未完成但可继续 → 注入引导消息继续尝试
    - 任务未完成且无法继续 → 停止等待用户

    Attributes:
        satisfied: 是否满足完成条件。
        reason: 检查结果原因（blocked_steps/modified_without_fresh_verification/incomplete_steps/all_steps_completed）。
        missing: 缺失的完成条件列表（如 ["unblocked_steps", "fresh_verification"]）。
        can_continue: 是否可以继续尝试完成（如需要运行验证）。
        unverified: 是否存在未验证的变更。
    """
    satisfied: bool                                                          # 是否满足完成条件
    reason: CompletionReason                                                 # 检查结果原因
    missing: list[str] = field(default_factory=list)                         # 缺失条件列表
    can_continue: bool = False                                               # 是否可继续
    unverified: bool = False                                                 # 是否有未验证变更

    def __post_init__(self) -> None:
        """初始化后校验：确保检查结果原因的值合法。"""
        _ensure_completion_reason(self.reason)


# ── 类型校验辅助函数 ─────────────────────────────────────────────
# 这些函数用于 dataclass 的 __post_init__ 中，确保字段值的类型正确。
# 采用防御式编程：在赋值时就捕获类型错误，而不是等到使用时才发现。

def _ensure_task_step_status(value: object) -> TaskStepStatus:
    """校验任务步骤状态：必须是 pending/in_progress/completed/blocked 之一。"""
    if value not in _TASK_STEP_STATUSES:
        raise ValueError(f"Unknown task step status: {value}")
    return cast(TaskStepStatus, value)


def _ensure_task_step_kind(value: object) -> TaskStepKind:
    """校验任务步骤类型：必须是 investigate/edit/verify/summarize/other 之一。"""
    if value not in TASK_STEP_KINDS:
        raise ValueError(f"Unknown task step kind: {value}")
    return cast(TaskStepKind, value)


def _ensure_task_progress_state(value: object) -> TaskProgressState:
    """校验任务进展状态：必须是 none/evidence_collected/changed/verified 之一。"""
    if value not in _TASK_PROGRESS_STATES:
        raise ValueError(f"Unknown task progress state: {value}")
    return cast(TaskProgressState, value)


def _ensure_task_phase(value: object) -> TaskPhase:
    """校验任务阶段：必须是 understanding/acting/verifying/waiting/finished 之一。"""
    if value not in _TASK_PHASES:
        raise ValueError(f"Unknown task phase: {value}")
    return cast(TaskPhase, value)


def _ensure_attempt_status(value: object) -> AttemptStatus:
    """校验尝试状态：必须是 running/succeeded/failed 之一。"""
    if value not in _ATTEMPT_STATUSES:
        raise ValueError(f"Unknown attempt status: {value}")
    return cast(AttemptStatus, value)


def _ensure_change_set_status(value: object) -> ChangeSetStatus:
    """校验变更集合状态：必须是 pending/verified/failed/revert_required 之一。"""
    if value not in _CHANGE_SET_STATUSES:
        raise ValueError(f"Unknown change set status: {value}")
    return cast(ChangeSetStatus, value)


def _ensure_execution_action(value: object) -> ExecutionAction:
    """校验执行决策动作：必须是 continue/repair/replan/propose_revert/wait_approval/finish/stop 之一。"""
    if value not in _EXECUTION_ACTIONS:
        raise ValueError(f"Unknown execution action: {value}")
    return cast(ExecutionAction, value)


def _ensure_completion_reason(value: object) -> CompletionReason:
    """校验完成检查原因：必须是 blocked_steps/modified_without_fresh_verification/incomplete_steps/all_steps_completed 之一。"""
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
