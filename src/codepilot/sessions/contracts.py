"""会话层核心契约 —— 定义 Session、Run、Message、Checkpoint 等数据模型。

本文件是 sessions 层的数据模型层，定义了所有核心类型：

1. 会话状态（SessionState）
   - 会话的元信息（ID、工作区、模型、模式等）
   - 支持主会话和子代理会话（subagent）

2. 运行状态（RunState）
   - 单次 Agent 运行的生命周期（created → running → completed/failed/cancelled）
   - 运行阶段（phase）：received → model → tools → finalizing → finished
   - 恢复点（resume_point）：标记运行中断的位置

3. 检查点（RunCheckpoint、ComponentCheckpoint、WorkspaceCheckpoint）
   - 运行中断时的状态快照
   - 组件级检查点（tools、context）
   - 工作区检查点（Git HEAD、文件哈希）

4. 消息记录（MessageRecord、MessageCursor）
   - 消息的持久化格式
   - 消息链遍历（通过 parent_id 链接）

5. 恢复相关（RecoveryRequest、RecoveryBundle、RecoveryResult）
   - 会话恢复的完整数据包
   - 工作区状态验证（unchanged/changed/missing）

6. 意图类型（SessionRunIntent、SessionResumeIntent 等）
   - 会话层接收的外部请求
   - 驱动 SessionStateService 的操作

7. 验证函数（validate_run_state、validate_run_transition）
   - 运行状态的不变量检查
   - 状态转换的合法性检查
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Mapping, Optional

from codepilot.core.contracts import (
    AgentMessage,
    AgentLoopInput,
    AgentLoopOutcome,
    ContextPort,
    PrepareContextFn,
    RunStatePort,
)
from codepilot.core.plan import PlanningBudgetProfile, RunMode
from codepilot.llm.provider_types import ProviderSimpleStreamFn
from codepilot.protocols import AgentEvent, AgentRunStatus, Message
from codepilot.protocols import Model
from codepilot.protocols.commands import LifecycleHook, RegisteredCommand


# ── Schema 版本号 ──────────────────────────────────────────────────────────────

SESSION_STATE_SCHEMA_VERSION = 1
RUN_STATE_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
MESSAGE_RECORD_SCHEMA_VERSION = 1


# ── 类型别名 ──────────────────────────────────────────────────────────────────

# SessionKind: 会话类型
#   - primary: 主会话（用户直接交互的会话）
#   - subagent: 子代理会话（由主会话派生的子会话）
SessionKind = Literal["primary", "subagent"]

# RunStatus: 运行状态（生命周期）
#   - created:   已创建（尚未开始执行）
#   - running:   执行中
#   - waiting:   等待中（等待审批/用户输入）
#   - completed: 已完成
#   - failed:    失败
#   - cancelled: 已取消
RunStatus = Literal[
    "created",
    "running",
    "waiting",
    "completed",
    "failed",
    "cancelled",
]

# RunPhase: 运行阶段（执行过程中的位置）
#   - received:   已接收输入
#   - model:      模型推理中
#   - tools:      工具执行中
#   - finalizing: 最终处理中
#   - finished:   已完成
RunPhase = Literal["received", "model", "tools", "finalizing", "finished"]
RunCommitKind = Literal["progress", "waiting", "terminal"]

# ResumePoint: 恢复点（中断后恢复的入口位置）
#   - before_model:      模型推理前
#   - after_model:       模型推理后（工具调用前）
#   - before_tools:      工具执行前
#   - after_tools:       工具执行后
#   - before_finalization: 最终处理前
ResumePoint = Literal[
    "before_model",
    "after_model",
    "before_tools",
    "after_tools",
    "before_finalization",
]

# WaitingKind: 等待类型（为什么运行被挂起）
#   - tool_approval:    等待工具审批
#   - user_input:       等待用户输入
#   - plan_confirmation: 等待计划确认
WaitingKind = Literal["tool_approval", "user_input", "plan_confirmation"]

# CheckpointOwner: 检查点所有者
#   - tools:   工具子系统
#   - context: 上下文子系统
CheckpointOwner = Literal["tools", "context"]

# RecoveryStatus: 恢复状态
#   - ready:             准备就绪，可以直接恢复
#   - needs_validation:  需要验证（组件检查点或工作区有变化）
#   - blocked:           无法恢复
#   - not_found:         未找到会话/运行
RecoveryStatus = Literal["ready", "needs_validation", "blocked", "not_found"]

# WorkspaceRecoveryStatus: 工作区恢复状态
#   - unchanged: 工作区未变化
#   - changed:   工作区有变化
#   - missing:   工作区文件缺失
#   - unknown:   未知
WorkspaceRecoveryStatus = Literal["unchanged", "changed", "missing", "unknown"]

# 合法值集合
_RUN_STATUSES = frozenset({"created", "running", "waiting", "completed", "failed", "cancelled"})
_TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "cancelled"})
_RUN_PHASES = frozenset({"received", "model", "tools", "finalizing", "finished"})
_RESUME_POINTS = frozenset(
    {"before_model", "after_model", "before_tools", "after_tools", "before_finalization"}
)
_WAITING_KINDS = frozenset({"tool_approval", "user_input", "plan_confirmation"})
_CHECKPOINT_OWNERS = frozenset({"tools", "context"})
_RECOVERY_STATUSES = frozenset({"ready", "needs_validation", "blocked", "not_found"})
_WORKSPACE_RECOVERY_STATUSES = frozenset({"unchanged", "changed", "missing", "unknown"})

# 阶段到可用恢复点的映射
_PHASE_RESUME_POINTS = {
    "received": frozenset({"before_model"}),
    "model": frozenset({"before_model", "after_model"}),
    "tools": frozenset({"before_tools", "after_tools"}),
    "finalizing": frozenset({"before_finalization"}),
    "finished": frozenset(),
}

# 运行状态转换规则
_RUN_TRANSITIONS = {
    "created": frozenset({"running", "failed", "cancelled"}),
    "running": frozenset({"running", "waiting", "completed", "failed", "cancelled"}),
    "waiting": frozenset({"running", "failed", "cancelled"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}


class SessionStateValidationError(ValueError):
    """会话状态验证错误 —— 当状态对象违反不变量时抛出。"""


class SessionStateConflictError(RuntimeError):
    """会话状态冲突 —— 当持久化状态在加载后被修改时抛出。"""


# ── 核心数据模型 ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelRef:
    """模型引用 —— 标识会话使用的 LLM 提供商和模型 ID。

    参数:
        provider: LLM 提供商名称（如 "anthropic"、"openai"）
        model: 模型 ID（如 "claude-sonnet-4-20250514"）
    """
    provider: str
    model: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _require_text(self.provider, "model provider"))
        object.__setattr__(self, "model", _require_text(self.model, "model id"))


@dataclass(frozen=True)
class WorkspaceEffectsSnapshot:
    """工作区效果快照 —— 记录运行对工作区的变更。

    参数:
        changed: 工作区是否发生了变更
        affected_paths: 受影响的工作区文件路径列表
        baseline_ref: 基线引用（如 Git 提交哈希）
        final_fingerprint: 最终状态指纹
    """
    changed: bool = False
    affected_paths: tuple[str, ...] = ()
    baseline_ref: str | None = None
    final_fingerprint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "affected_paths", _text_tuple(self.affected_paths, "affected_paths"))
        object.__setattr__(self, "baseline_ref", _optional_text(self.baseline_ref))
        object.__setattr__(self, "final_fingerprint", _optional_text(self.final_fingerprint))


@dataclass(frozen=True)
class MessageRecord:
    """消息记录 —— 持久化到文件系统的会话消息。

    消息通过 parent_id 链接形成消息链（支持分支/分叉）。
    每条消息属于一个会话和一个运行。

    参数:
        message_id: 消息的唯一 ID
        session_id: 所属会话的 ID
        message: 协议层消息（UserMessage / AssistantMessage / ToolResultMessage）
        run_id: 所属运行的 ID（可选）
        parent_id: 父消息的 ID（用于构建消息链，可选）
        created_at: 创建时间（ISO 格式）
        schema_version: Schema 版本号
    """
    message_id: str
    session_id: str
    message: Message
    run_id: str | None = None
    parent_id: str | None = None
    created_at: str = ""
    schema_version: int = MESSAGE_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version, MESSAGE_RECORD_SCHEMA_VERSION, "message record")
        object.__setattr__(self, "message_id", _require_text(self.message_id, "message_id"))
        object.__setattr__(self, "session_id", _require_text(self.session_id, "session_id"))
        object.__setattr__(self, "run_id", _optional_text(self.run_id))
        object.__setattr__(self, "parent_id", _optional_text(self.parent_id))
        object.__setattr__(self, "created_at", _require_text(self.created_at, "created_at"))


@dataclass(frozen=True)
class MessageCursor:
    """消息游标 —— 指向消息链中的当前叶子节点。

    参数:
        leaf_message_id: 当前叶子消息的 ID
    """
    leaf_message_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "leaf_message_id",
            _require_text(self.leaf_message_id, "leaf_message_id"),
        )


@dataclass(frozen=True)
class WaitingState:
    """等待状态 —— 运行被挂起时的状态信息。

    参数:
        kind: 等待类型（工具审批 / 用户输入 / 计划确认）
        request_id: 对应的请求 ID（审批 ID 或交互 ID）
        payload: 额外数据载荷
    """
    kind: WaitingKind
    request_id: str
    payload: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in _WAITING_KINDS:
            raise SessionStateValidationError(f"Unknown waiting kind: {self.kind}")
        object.__setattr__(self, "request_id", _require_text(self.request_id, "request_id"))
        object.__setattr__(self, "payload", _serializable_mapping(self.payload, "waiting payload"))


@dataclass(frozen=True)
class ComponentCheckpoint:
    """组件检查点 —— 子系统（tools/context）在检查点时的状态快照。

    参数:
        owner: 组件所有者（"tools" 或 "context"）
        schema_version: 组件状态的 Schema 版本
        state: 组件状态数据（JSON 可序列化）
    """
    owner: CheckpointOwner
    schema_version: int
    state: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.owner not in _CHECKPOINT_OWNERS:
            raise SessionStateValidationError(f"Unknown checkpoint owner: {self.owner}")
        _require_positive_int(self.schema_version, "component schema_version")
        object.__setattr__(self, "state", _serializable_mapping(self.state, "component state"))


@dataclass(frozen=True)
class WorkspaceCheckpoint:
    """工作区检查点 —— 运行开始时的文件系统状态快照。

    用于恢复时检测工作区是否发生了变化（文件修改、Git HEAD 变更等）。

    参数:
        root: 工作区根目录路径
        git_head: Git HEAD 的提交哈希（可选）
        dirty_paths: 未提交的变更路径列表
        tracked_path_hashes: 跟踪文件的 SHA256 哈希映射（路径 → 哈希）
    """
    root: str
    git_head: str | None = None
    dirty_paths: tuple[str, ...] = ()
    tracked_path_hashes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", _require_text(self.root, "workspace root"))
        object.__setattr__(self, "git_head", _optional_text(self.git_head))
        object.__setattr__(self, "dirty_paths", _text_tuple(self.dirty_paths, "dirty_paths"))
        hashes = {
            _require_text(path, "tracked path"): _require_text(value, "tracked path hash")
            for path, value in self.tracked_path_hashes.items()
        }
        object.__setattr__(self, "tracked_path_hashes", hashes)


@dataclass(frozen=True)
class RunCheckpoint:
    """运行检查点 —— 运行中断时的完整状态快照。

    包含恢复所需的所有信息：
    - 恢复点（从哪里继续）
    - 消息游标（当前消息位置）
    - 等待状态（如果是因为审批/交互挂起）
    - 组件检查点（tools/context 的状态）
    - 工作区检查点（文件系统状态）

    参数:
        checkpoint_id: 检查点的唯一 ID
        resume_point: 恢复点位置
        message_cursor: 消息游标（当前叶子消息）
        waiting: 等待状态（如果运行被挂起）
        components: 组件检查点列表
        workspace: 工作区检查点
        created_at: 创建时间（ISO 格式）
        schema_version: Schema 版本号
    """
    checkpoint_id: str
    resume_point: ResumePoint
    message_cursor: MessageCursor
    waiting: WaitingState | None = None
    components: tuple[ComponentCheckpoint, ...] = ()
    workspace: WorkspaceCheckpoint | None = None
    created_at: str = ""
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version, CHECKPOINT_SCHEMA_VERSION, "run checkpoint")
        object.__setattr__(self, "checkpoint_id", _require_text(self.checkpoint_id, "checkpoint_id"))
        if self.resume_point not in _RESUME_POINTS:
            raise SessionStateValidationError(f"Unknown resume point: {self.resume_point}")
        object.__setattr__(self, "components", tuple(self.components))
        owners = [component.owner for component in self.components]
        if len(owners) != len(set(owners)):
            raise SessionStateValidationError("Checkpoint component owners must be unique")
        object.__setattr__(self, "created_at", _require_text(self.created_at, "created_at"))


@dataclass(frozen=True)
class SessionState:
    """会话状态 —— 一个会话的完整元信息。

    会话是用户与 Agent 交互的上下文容器。一个会话包含多次运行（Run），
    每次运行包含多轮模型推理 + 工具执行。

    参数:
        session_id: 会话的唯一 ID
        workspace_root: 工作区根目录路径
        model: 使用的 LLM 模型引用
        current_run_id: 当前活动的运行 ID（如果有）
        last_run_id: 最近一次运行的 ID
        leaf_message_id: 消息链的叶子节点 ID
        parent_session_id: 父会话 ID（仅 subagent 会话有）
        parent_run_id: 父运行 ID（仅 subagent 会话有）
        session_kind: 会话类型（primary / subagent）
        current_mode: 当前会话模式（build / plan / read）
        system_prompt_hash: 系统提示词的哈希值
        created_at: 创建时间
        updated_at: 最后更新时间
        revision: 乐观锁版本号（每次更新递增）
        schema_version: Schema 版本号
    """
    session_id: str
    workspace_root: str
    model: ModelRef
    current_run_id: str | None = None
    last_run_id: str | None = None
    leaf_message_id: str | None = None
    parent_session_id: str | None = None
    parent_run_id: str | None = None
    session_kind: SessionKind = "primary"
    current_mode: str = "build"
    system_prompt_hash: str = ""
    created_at: str = ""
    updated_at: str = ""
    revision: int = 1
    schema_version: int = SESSION_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version, SESSION_STATE_SCHEMA_VERSION, "session state")
        object.__setattr__(self, "session_id", _require_text(self.session_id, "session_id"))
        object.__setattr__(self, "workspace_root", _require_text(self.workspace_root, "workspace_root"))
        object.__setattr__(self, "current_run_id", _optional_text(self.current_run_id))
        object.__setattr__(self, "last_run_id", _optional_text(self.last_run_id))
        object.__setattr__(self, "leaf_message_id", _optional_text(self.leaf_message_id))
        object.__setattr__(self, "parent_session_id", _optional_text(self.parent_session_id))
        object.__setattr__(self, "parent_run_id", _optional_text(self.parent_run_id))
        if self.session_kind not in {"primary", "subagent"}:
            raise SessionStateValidationError(f"Unknown session kind: {self.session_kind}")
        if self.session_kind == "primary" and self.parent_run_id is not None:
            raise SessionStateValidationError("Primary sessions cannot have parent_run_id")
        if self.session_kind == "subagent" and self.parent_session_id is None:
            raise SessionStateValidationError("Subagent sessions require parent_session_id")
        object.__setattr__(self, "current_mode", _require_text(self.current_mode, "current_mode"))
        object.__setattr__(
            self,
            "system_prompt_hash",
            _require_text(self.system_prompt_hash, "system_prompt_hash"),
        )
        object.__setattr__(self, "created_at", _require_text(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _require_text(self.updated_at, "updated_at"))
        _require_positive_int(self.revision, "session revision")


@dataclass(frozen=True)
class RunState:
    """运行状态 —— 一次 Agent 运行的完整生命周期数据。

    运行是 Agent 执行的基本单位。一次运行从接收用户输入开始，
    经过模型推理、工具执行，最终完成或失败。

    运行的生命周期:
    created → running → waiting(可选) → running → completed/failed/cancelled

    阶段流转:
    received → model → tools → finalizing → finished

    参数:
        run_id: 运行的唯一 ID
        session_id: 所属会话的 ID
        status: 运行状态
        phase: 运行阶段
        input_message_id: 触发本次运行的用户消息 ID
        core_state: 核心状态（会话协调器的状态）
        workspace_effects: 工作区效果快照
        stop_reason: 停止原因
        latest_message_id: 最新消息的 ID
        checkpoint: 当前检查点（非终止状态必须有）
        result_ref: 结果引用路径（artifact 路径）
        resume_count: 恢复次数（运行被resume了多少次）
        created_at: 创建时间
        updated_at: 最后更新时间
        started_at: 开始执行时间
        ended_at: 结束执行时间
        revision: 乐观锁版本号
        schema_version: Schema 版本号
    """
    run_id: str
    session_id: str
    status: RunStatus
    phase: RunPhase
    input_message_id: str
    core_state: dict[str, object]
    workspace_effects: WorkspaceEffectsSnapshot = field(default_factory=WorkspaceEffectsSnapshot)
    stop_reason: str | None = None
    latest_message_id: str | None = None
    checkpoint: RunCheckpoint | None = None
    result_ref: str | None = None
    last_commit_id: str | None = None
    last_commit_kind: RunCommitKind | None = None
    last_commit_digest: str | None = None
    last_commit_session_revision: int | None = None
    last_commit_message_ids: tuple[str, ...] = ()
    resume_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    started_at: str | None = None
    ended_at: str | None = None
    revision: int = 1
    schema_version: int = RUN_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version, RUN_STATE_SCHEMA_VERSION, "run state")
        object.__setattr__(self, "run_id", _require_text(self.run_id, "run_id"))
        object.__setattr__(self, "session_id", _require_text(self.session_id, "session_id"))
        if self.status not in _RUN_STATUSES:
            raise SessionStateValidationError(f"Unknown run status: {self.status}")
        if self.phase not in _RUN_PHASES:
            raise SessionStateValidationError(f"Unknown run phase: {self.phase}")
        object.__setattr__(
            self,
            "input_message_id",
            _require_text(self.input_message_id, "input_message_id"),
        )
        object.__setattr__(self, "core_state", _serializable_mapping(self.core_state, "core_state"))
        object.__setattr__(self, "stop_reason", _optional_text(self.stop_reason))
        object.__setattr__(self, "latest_message_id", _optional_text(self.latest_message_id))
        object.__setattr__(self, "result_ref", _optional_text(self.result_ref))
        object.__setattr__(self, "last_commit_id", _optional_text(self.last_commit_id))
        if self.last_commit_kind not in {None, "progress", "waiting", "terminal"}:
            raise SessionStateValidationError(
                f"Unknown last commit kind: {self.last_commit_kind}"
            )
        object.__setattr__(
            self,
            "last_commit_digest",
            _optional_text(self.last_commit_digest),
        )
        if self.last_commit_session_revision is not None:
            _require_positive_int(
                self.last_commit_session_revision,
                "last_commit_session_revision",
            )
        object.__setattr__(
            self,
            "last_commit_message_ids",
            _text_tuple(self.last_commit_message_ids, "last_commit_message_ids"),
        )
        commit_fields = (
            self.last_commit_id,
            self.last_commit_kind,
            self.last_commit_digest,
            self.last_commit_session_revision,
        )
        if any(value is not None for value in commit_fields) and any(
            value is None for value in commit_fields
        ):
            raise SessionStateValidationError(
                "last commit id, kind and digest must be set together"
            )
        object.__setattr__(self, "created_at", _require_text(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _require_text(self.updated_at, "updated_at"))
        object.__setattr__(self, "started_at", _optional_text(self.started_at))
        object.__setattr__(self, "ended_at", _optional_text(self.ended_at))
        _require_non_negative_int(self.resume_count, "resume_count")
        _require_positive_int(self.revision, "run revision")
        validate_run_state(self)


# ── 恢复相关类型 ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RecoveryRequest:
    """恢复请求 —— 查询恢复所需数据的请求参数。

    参数:
        session_id: 要恢复的会话 ID
        run_id: 要恢复的运行 ID（可选，默认使用当前运行）
        expected_checkpoint_id: 期望的检查点 ID（用于验证）
        expected_waiting_kind: 期望的等待类型（用于验证）
    """
    session_id: str
    run_id: str | None = None
    expected_checkpoint_id: str | None = None
    expected_waiting_kind: WaitingKind | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _require_text(self.session_id, "session_id"))
        object.__setattr__(self, "run_id", _optional_text(self.run_id))
        object.__setattr__(
            self,
            "expected_checkpoint_id",
            _optional_text(self.expected_checkpoint_id),
        )
        if self.expected_waiting_kind is not None and self.expected_waiting_kind not in _WAITING_KINDS:
            raise SessionStateValidationError(
                f"Unknown expected waiting kind: {self.expected_waiting_kind}"
            )


@dataclass(frozen=True)
class RecoveryIssue:
    """恢复问题 —— 恢复过程中遇到的阻碍。

    参数:
        code: 问题代码（如 "session.not_found"、"run.terminal"）
        message: 人类可读的问题描述
    """
    code: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _require_text(self.code, "recovery issue code"))
        object.__setattr__(self, "message", _require_text(self.message, "recovery issue message"))


@dataclass(frozen=True)
class WorkspaceRecoveryState:
    """工作区恢复状态 —— 恢复时工作区的变化情况。

    参数:
        status: 工作区状态（unchanged / changed / missing / unknown）
        changed_paths: 内容发生变化的文件路径列表
        missing_paths: 已缺失的文件路径列表
    """
    status: WorkspaceRecoveryStatus
    changed_paths: tuple[str, ...] = ()
    missing_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in _WORKSPACE_RECOVERY_STATUSES:
            raise SessionStateValidationError(f"Unknown workspace recovery status: {self.status}")
        object.__setattr__(self, "changed_paths", _text_tuple(self.changed_paths, "changed_paths"))
        object.__setattr__(self, "missing_paths", _text_tuple(self.missing_paths, "missing_paths"))


@dataclass(frozen=True)
class RecoveryBundle:
    """恢复数据包 —— 恢复所需的所有数据的聚合。

    包含会话状态、运行状态、消息链和工作区状态。

    参数:
        session: 会话状态
        run: 运行状态
        messages: 从根到叶子节点的消息链
        workspace_status: 工作区恢复状态
    """
    session: SessionState
    run: RunState
    messages: tuple[MessageRecord, ...]
    workspace_status: WorkspaceRecoveryState

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(self.messages))
        if self.run.session_id != self.session.session_id:
            raise SessionStateValidationError("Recovery run does not belong to recovery session")


@dataclass(frozen=True)
class RecoveryResult:
    """恢复结果 —— 恢复请求的完整响应。

    参数:
        status: 恢复状态
        bundle: 恢复数据包（仅当 status=ready/needs_validation 时有）
        issues: 恢复问题列表（仅当 status=blocked/not_found 时有）
    """
    status: RecoveryStatus
    bundle: RecoveryBundle | None = None
    issues: tuple[RecoveryIssue, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in _RECOVERY_STATUSES:
            raise SessionStateValidationError(f"Unknown recovery status: {self.status}")
        object.__setattr__(self, "issues", tuple(self.issues))
        if self.status in {"ready", "needs_validation"} and self.bundle is None:
            raise SessionStateValidationError(f"Recovery status {self.status} requires a bundle")
        if self.status in {"blocked", "not_found"} and self.bundle is not None:
            raise SessionStateValidationError(f"Recovery status {self.status} cannot include a bundle")


# ── 验证函数 ──────────────────────────────────────────────────────────────────


def validate_run_state(state: RunState) -> None:
    """验证运行状态的不变量 —— 检查单个 RunState 对象的内部一致性。

    验证规则:
    1. 终止状态（completed/failed/cancelled）必须 phase=finished 且无 checkpoint
    2. 非终止状态必须有 checkpoint
    3. 非终止状态的 resume_point 必须与 phase 兼容
    4. latest_message_id 必须与 checkpoint 的 leaf_message_id 一致
    5. waiting 状态必须有 waiting checkpoint
    6. 非 waiting 状态不能有 waiting checkpoint
    7. created 状态必须 phase=received 且 resume_point=before_model

    参数:
        state: 待验证的 RunState
    """
    terminal = state.status in _TERMINAL_RUN_STATUSES
    if terminal:
        if state.phase != "finished":
            raise SessionStateValidationError("Terminal runs require phase=finished")
        if state.checkpoint is not None:
            raise SessionStateValidationError("Terminal runs cannot retain an active checkpoint")
        if state.ended_at is None:
            raise SessionStateValidationError("Terminal runs require ended_at")
        return

    if state.phase == "finished":
        raise SessionStateValidationError("Non-terminal runs cannot use phase=finished")
    if state.checkpoint is None:
        raise SessionStateValidationError("Non-terminal runs require a checkpoint")
    allowed_resume_points = _PHASE_RESUME_POINTS[state.phase]
    if state.checkpoint.resume_point not in allowed_resume_points:
        raise SessionStateValidationError(
            f"Resume point {state.checkpoint.resume_point} is invalid for phase {state.phase}"
        )
    if state.latest_message_id != state.checkpoint.message_cursor.leaf_message_id:
        raise SessionStateValidationError(
            "latest_message_id must match the checkpoint message cursor"
        )
    if state.status == "waiting" and state.checkpoint.waiting is None:
        raise SessionStateValidationError("Waiting runs require waiting checkpoint state")
    if state.status != "waiting" and state.checkpoint.waiting is not None:
        raise SessionStateValidationError("Only waiting runs can contain waiting checkpoint state")
    if state.status == "created":
        if state.phase != "received" or state.checkpoint.resume_point != "before_model":
            raise SessionStateValidationError(
                "Created runs must start at received/before_model"
            )


def validate_run_transition(previous: RunState, current: RunState) -> None:
    """验证运行状态转换 —— 检查两个状态之间的转换是否合法。

    验证规则:
    1. 运行身份不能改变（run_id 和 session_id 必须一致）
    2. revision 必须递增正好 1
    3. 状态转换必须符合 _RUN_TRANSITIONS 中定义的规则

    参数:
        previous: 转换前的状态
        current: 转换后的状态
    """
    if previous.run_id != current.run_id or previous.session_id != current.session_id:
        raise SessionStateValidationError("Run identity cannot change")
    if current.revision != previous.revision + 1:
        raise SessionStateValidationError("Run revision must increase by exactly one")
    allowed = _RUN_TRANSITIONS[previous.status]
    if current.status not in allowed:
        raise SessionStateValidationError(
            f"Invalid run transition: {previous.status} -> {current.status}"
        )


# ── 意图类型（会话层接收的外部请求） ──────────────────────────────────────────


ConvertToLlmFn = Callable[[list[AgentMessage]], list[Message] | Awaitable[list[Message]]]
SystemPromptBuilder = Callable[[RunMode], str]
SessionContinuationKind = Literal[
    "plan_approved",
    "plan_rejected",
    "plan_feedback",
    "plan_clarification",
    "tool_approved",
    "tool_denied",
    "mode_changed",
    "automatic_continuation",
]
_CONTINUATION_KINDS = {
    "plan_approved",
    "plan_rejected",
    "plan_feedback",
    "plan_clarification",
    "tool_approved",
    "tool_denied",
    "mode_changed",
    "automatic_continuation",
}


@dataclass
class SessionOptions:
    """会话选项 —— 打开会话控制器时所需的运行时配置。

    参数:
        model: LLM 模型
        workspace_dir: 工作区目录
        system_prompt: 系统提示词
        system_prompt_builder: 系统提示词构建器（按模式动态生成）
        session_id: 会话 ID（可选，不提供则自动生成）
        messages: 初始消息列表
        thinking_level: 思考级别
        max_tool_calls_per_turn: 每轮最大工具调用数
        memory_enabled: 是否启用记忆
        current_mode: 当前运行模式
        planning_budget_profile: 计划预算配置
        convert_to_llm: 消息转换函数
        get_api_key: API 密钥获取函数
        retry_enabled: 是否启用重试
        max_retries: 最大重试次数
        retry_base_delay_ms: 重试基础延迟（毫秒）
        extension_commands: 扩展命令注册
        before_prompt_hooks: 提示前钩子
        after_prompt_hooks: 提示后钩子
        stream_fn: 流式函数
        prepare_context: 上下文准备函数
    """
    model: Model
    workspace_dir: str | Path
    system_prompt: str = ""
    system_prompt_builder: Optional[SystemPromptBuilder] = None
    session_id: Optional[str] = None
    messages: list[AgentMessage] = field(default_factory=list)
    thinking_level: str = "off"
    max_tool_calls_per_turn: int = 16
    memory_enabled: bool = True
    current_mode: RunMode = "build"
    planning_budget_profile: PlanningBudgetProfile = "balanced"
    convert_to_llm: Optional[ConvertToLlmFn] = None
    get_api_key: Optional[Callable[[str], str | None | Awaitable[str | None]]] = None
    retry_enabled: bool = True
    max_retries: int = 2
    retry_base_delay_ms: int = 1200
    run_timeout_seconds: int | None = None
    extension_commands: dict[str, RegisteredCommand] = field(default_factory=dict)
    before_prompt_hooks: list[LifecycleHook] = field(default_factory=list)
    after_prompt_hooks: list[LifecycleHook] = field(default_factory=list)
    stream_fn: ProviderSimpleStreamFn | None = None
    prepare_context: PrepareContextFn | None = None


@dataclass(frozen=True)
class SessionRunIntent:
    """会话运行意图 —— 提交一个新提示词给会话。

    参数:
        text: 用户输入的文本
        images: 图片数据列表（base64 编码）
        mode_hint: 模式提示（可选，覆盖当前模式）
        run_id: 运行 ID（可选，不提供则自动生成）
    """
    text: str
    images: tuple[str, ...] = ()
    mode_hint: str | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _require_text(self.text, "run text"))
        object.__setattr__(self, "mode_hint", _optional_text(self.mode_hint))
        object.__setattr__(self, "run_id", _optional_text(self.run_id))


@dataclass(frozen=True)
class SessionResumeIntent:
    """会话恢复意图 —— 对审批挑战做出响应。

    参数:
        approval_id: 审批 ID
        decision: 决策（approve / deny）
        reason: 审批理由（可选）
        run_id: 运行 ID（可选）
    """
    approval_id: str
    decision: str
    reason: str = ""
    run_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "approval_id", _require_text(self.approval_id, "approval_id"))
        object.__setattr__(self, "decision", _approval_decision(self.decision))
        object.__setattr__(self, "reason", _optional_text(self.reason) or "")
        object.__setattr__(self, "run_id", _optional_text(self.run_id))


@dataclass(frozen=True)
class SessionContinuationIntent:
    """会话继续意图 —— 自动继续运行（内部使用）。

    用于处理审批后的自动恢复、模式切换后的自动继续等场景。

    参数:
        kind: 继续类型
        run_id: 运行 ID（可选）
        text: 补充文本（如 plan_feedback 时）
        approval_id: 审批 ID（tool_approved/tool_denied 时）
        decision: 审批决策
        reason: 理由
        target_mode: 目标模式（mode_changed 时）
    """
    kind: SessionContinuationKind
    run_id: str | None = None
    text: str = ""
    approval_id: str = ""
    decision: str = ""
    reason: str = ""
    target_mode: str | None = None

    def __post_init__(self) -> None:
        kind = _require_text(self.kind, "continuation kind")
        if kind not in _CONTINUATION_KINDS:
            raise ValueError(f"Unknown continuation kind: {self.kind}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "run_id", _optional_text(self.run_id))
        object.__setattr__(self, "text", _optional_text(self.text) or "")
        object.__setattr__(self, "approval_id", _optional_text(self.approval_id) or "")
        object.__setattr__(self, "decision", _optional_text(self.decision) or "")
        object.__setattr__(self, "reason", _optional_text(self.reason) or "")
        object.__setattr__(self, "target_mode", _optional_text(self.target_mode))
        if kind == "plan_feedback" and not self.text:
            raise ValueError("plan feedback text is required")
        if kind in {"tool_approved", "tool_denied"} and not self.approval_id:
            raise ValueError("tool continuation approval_id is required")


@dataclass(frozen=True)
class SessionCommandIntent:
    """会话命令意图 —— 在会话中执行斜杠命令。

    参数:
        text: 命令文本（如 "/help"）
        tool_catalog: 工具目录（可选，用于命令需要工具信息时）
    """
    text: str
    tool_catalog: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _require_text(self.text, "command text"))
        object.__setattr__(self, "tool_catalog", tuple(self.tool_catalog))


@dataclass(frozen=True)
class CancelRunIntent:
    """取消运行意图 —— 取消当前正在执行的运行。

    参数:
        run_id: 要取消的运行 ID（可选，默认取消当前运行）
        reason: 取消原因（默认 "user"）
    """
    run_id: str | None = None
    reason: str = "user"

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _optional_text(self.run_id))
        object.__setattr__(self, "reason", _optional_text(self.reason) or "user")


SessionIntent = (
    SessionRunIntent
    | SessionResumeIntent
    | SessionContinuationIntent
    | SessionCommandIntent
    | CancelRunIntent
)


# ── 视图类型 ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SessionView:
    """会话视图 —— 会话的轻量摘要（供 CLI/RPC 展示）。

    参数:
        session_id: 会话 ID
        message_count: 消息数
        last_run_id: 最近一次运行的 ID
        current_mode: 当前模式
        context: 上下文摘要
    """
    session_id: str
    message_count: int = 0
    last_run_id: str | None = None
    current_mode: str = "build"
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RollbackBaselineRef:
    """回滚基线引用 —— 指向用于回滚的基线状态。

    参数:
        session_id: 会话 ID
        run_id: 运行 ID（回滚到该运行之前的状态）
        kind: 固定为 "rollback_baseline_ref"
    """
    session_id: str
    run_id: str
    kind: Literal["rollback_baseline_ref"] = field(
        default="rollback_baseline_ref",
        init=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _require_text(self.session_id, "session_id"))
        object.__setattr__(self, "run_id", _require_text(self.run_id, "run_id"))


@dataclass(frozen=True)
class PreparedAgentRun:
    """已准备的 Agent 运行 —— 包含执行所需的所有组件。

    在 SessionCoordinator 准备好一次运行后，
    将各种组件打包为此类传递给 Agent 执行循环。

    参数:
        run_id: 运行 ID
        session_id: 会话 ID
        loop_input: Agent 循环输入
        context_port: 上下文端口
        state_port: 状态端口
        input_messages: 输入消息列表
        rollback_baseline: 回滚基线引用
        context_refs: 上下文引用
        memory_refs: 记忆引用
        plan_refs: 计划引用
    """
    run_id: str
    session_id: str
    loop_input: AgentLoopInput
    context_port: ContextPort | None = None
    state_port: RunStatePort | None = None
    input_messages: list[Message] = field(default_factory=list)
    rollback_baseline: RollbackBaselineRef | None = None
    context_refs: dict[str, Any] = field(default_factory=dict)
    memory_refs: dict[str, Any] = field(default_factory=dict)
    plan_refs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionRunRecord:
    """会话运行记录 —— 一次运行完成后的结果记录。

    参数:
        run_id: 运行 ID
        session_id: 会话 ID
        status: 运行状态
        stop_reason: 停止原因
        new_messages: 运行产生的新消息列表
        final_text: 最终文本
        events: 运行期间产生的事件列表
        outcome: Agent 循环结果
        snapshots: 状态快照
    """
    run_id: str
    session_id: str
    status: AgentRunStatus
    stop_reason: str
    new_messages: list[Message] = field(default_factory=list)
    final_text: str = ""
    events: list[AgentEvent] = field(default_factory=list)
    outcome: AgentLoopOutcome | None = None
    snapshots: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionCommandRecord:
    """会话命令记录 —— 斜杠命令的执行结果。

    参数:
        session_id: 会话 ID
        command: 执行的命令文本
        handled: 是否已被处理
        output_lines: 输出行列表
        switched_session_id: 切换到的会话 ID（如果命令切换了会话）
        data: 额外返回数据
    """
    session_id: str
    command: str
    handled: bool
    output_lines: tuple[str, ...] = ()
    switched_session_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


# ── 内部辅助函数 ──────────────────────────────────────────────────────────────


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("optional text must be a string or None")
    return value.strip() or None


def _approval_decision(value: object) -> str:
    text = _require_text(value, "approval decision").lower()
    if text not in {"approve", "deny"}:
        raise ValueError(f"Unknown approval decision: {value}")
    return text


def _require_schema(actual: object, expected: int, name: str) -> None:
    if actual != expected:
        raise SessionStateValidationError(
            f"Unsupported {name} schema_version: {actual}; expected {expected}"
        )


def _require_positive_int(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SessionStateValidationError(f"{field_name} must be a positive integer")


def _require_non_negative_int(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SessionStateValidationError(f"{field_name} must be a non-negative integer")


def _text_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise SessionStateValidationError(f"{field_name} must be a list or tuple")
    return tuple(_require_text(item, field_name) for item in value)


def _serializable_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SessionStateValidationError(f"{field_name} must be a mapping")
    result = dict(value)
    if not all(isinstance(key, str) for key in result):
        raise SessionStateValidationError(f"{field_name} keys must be strings")
    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SessionStateValidationError(f"{field_name} must be JSON serializable") from exc
    return result


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CancelRunIntent",
    "CheckpointOwner",
    "ComponentCheckpoint",
    "ConvertToLlmFn",
    "MESSAGE_RECORD_SCHEMA_VERSION",
    "MessageCursor",
    "MessageRecord",
    "ModelRef",
    "PreparedAgentRun",
    "RUN_STATE_SCHEMA_VERSION",
    "RecoveryBundle",
    "RecoveryIssue",
    "RecoveryRequest",
    "RecoveryResult",
    "RecoveryStatus",
    "ResumePoint",
    "RollbackBaselineRef",
    "RunCheckpoint",
    "RunCommitKind",
    "RunPhase",
    "RunState",
    "RunStatus",
    "SESSION_STATE_SCHEMA_VERSION",
    "SessionCommandIntent",
    "SessionCommandRecord",
    "SessionContinuationIntent",
    "SessionContinuationKind",
    "SessionIntent",
    "SessionKind",
    "SessionResumeIntent",
    "SessionRunIntent",
    "SessionRunRecord",
    "SessionOptions",
    "SessionState",
    "SessionStateConflictError",
    "SessionStateValidationError",
    "SessionView",
    "SystemPromptBuilder",
    "WaitingKind",
    "WaitingState",
    "WorkspaceCheckpoint",
    "WorkspaceEffectsSnapshot",
    "WorkspaceRecoveryState",
    "WorkspaceRecoveryStatus",
    "validate_run_state",
    "validate_run_transition",
]
