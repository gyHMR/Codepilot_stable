"""会话状态服务 —— 通过一组稳定操作提交 Session 和 Run 状态。

SessionStateService 是 sessions 层的核心服务类，提供：
1. create_session — 创建新会话
2. begin_run — 开始新运行（绑定用户输入消息）
3. commit_run_boundary — 统一提交 Progress/Waiting/Terminal 边界
4. resume_run — 恢复被挂起的运行
6. inspect_recovery — 检查可恢复性
7. 辅助方法：fork_session、append_message、set_leaf 等

所有写操作都使用乐观锁（revision）防止并发冲突。
"""

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from codepilot.observability.events import normalize_event_value
from codepilot.observability.redact import redact_artifact
from codepilot.protocols import AgentRunResult, Message, UserMessage

from .contracts import (
    ComponentCheckpoint,
    MessageCursor,
    MessageRecord,
    ModelRef,
    RecoveryBundle,
    RecoveryIssue,
    RecoveryRequest,
    RecoveryResult,
    RunCheckpoint,
    RunCommitKind,
    RunPhase,
    RunState,
    SessionState,
    SessionStateConflictError,
    SessionStateValidationError,
    WaitingState,
    WorkspaceCheckpoint,
    WorkspaceEffectsSnapshot,
    WorkspaceRecoveryState,
)
from .repository import FileSessionRepository
from .serde import message_to_dict, run_state_to_dict
from .workspace import validate_workspace_checkpoint


# ── 请求/结果类型 ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CreateSessionRequest:
    """创建会话请求。

    参数:
        workspace_root: 工作区根目录
        model: 模型引用
        current_mode: 当前模式
        system_prompt_hash: 系统提示词哈希
        session_id: 会话 ID（可选，不提供则自动生成）
        parent_session_id: 父会话 ID（subagent 时使用）
        parent_run_id: 父运行 ID（subagent 时使用）
        session_kind: 会话类型（primary/subagent）
    """
    workspace_root: str
    model: ModelRef
    current_mode: str
    system_prompt_hash: str
    session_id: str | None = None
    parent_session_id: str | None = None
    parent_run_id: str | None = None
    session_kind: Literal["primary", "subagent"] = "primary"


@dataclass(frozen=True)
class BeginRunRequest:
    """开始运行请求。

    参数:
        session_id: 会话 ID
        user_message: 触发运行的用户消息
        initial_core_state: 初始核心状态
        workspace: 工作区检查点（可选，运行开始时的文件状态快照）
        run_id: 运行 ID（可选，不提供则自动生成）
        message_id: 用户消息的 ID（可选，不提供则自动生成）
    """
    session_id: str
    user_message: UserMessage
    request_id: str = field(default_factory=lambda: _new_id("request"))
    initial_core_state: dict[str, object] = field(default_factory=dict)
    workspace: WorkspaceCheckpoint | None = None
    components: tuple[ComponentCheckpoint, ...] = ()
    run_id: str | None = None
    message_id: str | None = None


@dataclass(frozen=True)
class BeginRunResult:
    """开始运行结果。

    参数:
        session: 更新后的会话状态
        run: 新创建的运行状态
        message: 用户消息记录
    """
    session: SessionState
    run: RunState
    message: MessageRecord
    reused: bool = False


@dataclass(frozen=True)
class CommitRunBoundaryRequest:
    """Persist one progress, waiting, or terminal Run boundary."""

    commit_id: str
    kind: RunCommitKind
    session_id: str
    run_id: str
    expected_run_revision: int
    expected_session_revision: int
    phase: RunPhase | None = None
    resume_point: str | None = None
    core_state: dict[str, object] = field(default_factory=dict)
    new_messages: tuple[Message, ...] = ()
    durable_events: tuple[dict[str, Any], ...] = ()
    waiting: WaitingState | None = None
    components: tuple[ComponentCheckpoint, ...] = ()
    workspace: WorkspaceCheckpoint | None = None
    terminal_status: Literal["completed", "failed", "cancelled"] | None = None
    stop_reason: str | None = None
    result: AgentRunResult | None = None
    workspace_effects: WorkspaceEffectsSnapshot | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.commit_id, str) or not self.commit_id.strip():
            raise ValueError("commit_id is required")
        if self.kind not in {"progress", "waiting", "terminal"}:
            raise ValueError(f"Unknown commit kind: {self.kind}")
        for name, value in (
            ("expected_run_revision", self.expected_run_revision),
            ("expected_session_revision", self.expected_session_revision),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.kind == "terminal":
            if self.terminal_status not in {"completed", "failed", "cancelled"}:
                raise ValueError("Terminal commit requires terminal_status")
            if self.waiting is not None:
                raise ValueError("Terminal commit cannot include waiting state")
        else:
            if self.phase is None or self.resume_point is None:
                raise ValueError("Progress and waiting commits require phase and resume_point")
            if self.workspace is None:
                raise ValueError("Progress and waiting commits require workspace checkpoint")
            if self.terminal_status is not None or self.result is not None:
                raise ValueError("Non-terminal commit cannot include terminal result")
            if (self.kind == "waiting") != (self.waiting is not None):
                raise ValueError("Waiting commit requires exactly one waiting state")
        object.__setattr__(self, "commit_id", self.commit_id.strip())
        object.__setattr__(self, "new_messages", tuple(self.new_messages))
        object.__setattr__(self, "durable_events", tuple(dict(event) for event in self.durable_events))
        object.__setattr__(self, "components", tuple(self.components))


@dataclass(frozen=True)
class CommitRunBoundaryReceipt:
    """Durable receipt returned for a boundary commit or its idempotent retry."""

    commit_id: str
    kind: RunCommitKind
    session: SessionState
    run: RunState
    committed_messages: tuple[MessageRecord, ...]


@dataclass(frozen=True)
class ResumeRunRequest:
    """恢复运行请求。

    参数:
        session_id: 会话 ID
        run_id: 运行 ID
        checkpoint_id: 期望的检查点 ID（验证是否匹配）
        request_id: 等待的请求 ID（可选，验证审批/交互请求是否一致）
    """
    session_id: str
    run_id: str
    checkpoint_id: str
    request_id: str | None = None
    components: tuple[ComponentCheckpoint, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))
        owners = [component.owner for component in self.components]
        if len(owners) != len(set(owners)):
            raise ValueError("Resume component owners must be unique")


# ── 服务类 ──────────────────────────────────────────────────────────────────


class SessionStateService:
    """会话状态服务 —— 通过一组稳定操作提交 Session 和 Run 状态。

    这是 sessions 层的门面（Facade），提供高层操作：
    - 会话创建和查询
    - 运行开始、提交边界、恢复、完成
    - 消息追加和消息链查询
    - 会话分叉（fork）和树状查看
    - 恢复检查和事件记录

    所有写操作提供乐观锁（expected_revision）参数保护并发安全。

    参数:
        workspace_dir: 工作区目录
        repository: 文件系统仓库（可选，默认创建 FileSessionRepository）
    """

    def __init__(
        self,
        workspace_dir: str | Path,
        *,
        repository: FileSessionRepository | None = None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.repository = repository or FileSessionRepository(self.workspace_dir)

    # ── 会话创建 ────────────────────────────────────────────────────────────

    def create_session(self, request: CreateSessionRequest) -> SessionState:
        """创建新会话。

        参数:
            request: 创建会话的请求参数

        返回:
            持久化后的 SessionState
        """
        now = _utc_now_iso()
        state = SessionState(
            session_id=request.session_id or _new_id("session"),
            workspace_root=str(Path(request.workspace_root).resolve()).replace("\\", "/"),
            model=request.model,
            parent_session_id=request.parent_session_id,
            parent_run_id=request.parent_run_id,
            session_kind=request.session_kind,
            current_mode=request.current_mode,
            system_prompt_hash=request.system_prompt_hash,
            created_at=now,
            updated_at=now,
        )
        return self.repository.create_session(state)

    # ── 运行开始 ────────────────────────────────────────────────────────────

    def begin_run(
        self,
        request: BeginRunRequest,
        *,
        expected_session_revision: int,
    ) -> BeginRunResult:
        """开始新的运行。

        处理流程：
        1. 加载并验证会话状态
        2. 检查会话没有其他活动运行
        3. 创建用户消息记录
        4. 创建运行状态
        5. 更新会话状态（绑定 current_run_id）

        参数:
            request: 开始运行请求
            expected_session_revision: 预期的会话 revision

        返回:
            BeginRunResult（更新后的会话、新运行、消息记录）
        """
        session = self._require_session(request.session_id)
        request_id = _required_text(request.request_id, "request_id")
        request_digest = _begin_run_request_digest(request)
        session_runs = self.repository.list_runs(session_id=session.session_id)
        existing = next(
            (
                run
                for run in session_runs
                if run.request_id == request_id
            ),
            None,
        )
        if existing is not None:
            return self._reuse_begin_run(
                session,
                existing,
                request=request,
                request_digest=request_digest,
                expected_session_revision=expected_session_revision,
            )

        orphan_runs = [
            run
            for run in session_runs
            if run.status not in {"completed", "failed", "cancelled"}
            and run.run_id != session.current_run_id
        ]
        if orphan_runs:
            raise SessionStateConflictError(
                "Session has an orphan active run: "
                + ", ".join(run.run_id for run in orphan_runs)
            )

        self._require_revision(session.revision, expected_session_revision, "Session")
        if session.current_run_id is not None:
            raise SessionStateConflictError(
                f"Session already has an active run: {session.current_run_id}"
            )

        now = _utc_now_iso()
        run_id = request.run_id or _request_scoped_id("run", session.session_id, request_id)
        message_id = request.message_id or _request_scoped_id("msg", session.session_id, request_id)
        message = MessageRecord(
            message_id=message_id,
            session_id=session.session_id,
            run_id=run_id,
            parent_id=session.leaf_message_id,
            created_at=now,
            message=request.user_message,
        )
        self.repository.append_message(message)
        checkpoint = RunCheckpoint(
            checkpoint_id=_new_id("checkpoint"),
            resume_point="before_model",
            message_cursor=MessageCursor(message_id),
            workspace=request.workspace,
            components=request.components,
            created_at=now,
        )
        run = RunState(
            run_id=run_id,
            session_id=session.session_id,
            status="created",
            phase="received",
            input_message_id=message_id,
            latest_message_id=message_id,
            core_state=request.initial_core_state,
            request_id=request_id,
            request_digest=request_digest,
            checkpoint=checkpoint,
            created_at=now,
            updated_at=now,
        )
        self.repository.create_run(run)
        updated_session = replace(
            session,
            current_run_id=run_id,
            last_run_id=run_id,
            leaf_message_id=message_id,
            updated_at=now,
            revision=session.revision + 1,
        )
        self.repository.update_session(
            updated_session,
            expected_revision=expected_session_revision,
        )
        self._append_event("run_created", updated_session, run)
        return BeginRunResult(session=updated_session, run=run, message=message)

    def _reuse_begin_run(
        self,
        session: SessionState,
        run: RunState,
        *,
        request: BeginRunRequest,
        request_digest: str,
        expected_session_revision: int,
    ) -> BeginRunResult:
        if run.request_digest != request_digest:
            raise SessionStateConflictError(
                f"request_id was already used with different input: {request.request_id}"
            )
        records = {
            record.message_id: record
            for record in self.repository.load_message_records(session.session_id)
        }
        message = records.get(run.input_message_id)
        if message is None:
            raise SessionStateConflictError(
                f"Run input message is missing for request_id: {request.request_id}"
            )
        if run.status in {"completed", "failed", "cancelled"}:
            raise SessionStateConflictError(
                f"request_id already reached a terminal Run: {request.request_id}"
            )
        if session.current_run_id not in {None, run.run_id}:
            raise SessionStateConflictError(
                f"Session already has an active run: {session.current_run_id}"
            )
        if session.current_run_id == run.run_id:
            return BeginRunResult(session=session, run=run, message=message, reused=True)

        self._require_revision(session.revision, expected_session_revision, "Session")
        now = _utc_now_iso()
        repaired = replace(
            session,
            current_run_id=run.run_id,
            last_run_id=run.run_id,
            leaf_message_id=run.latest_message_id or run.input_message_id,
            updated_at=now,
            revision=session.revision + 1,
        )
        self.repository.update_session(repaired, expected_revision=expected_session_revision)
        self._append_event("run_admission_repaired", repaired, run)
        return BeginRunResult(session=repaired, run=run, message=message, reused=True)

    # ── 提交阶段边界 ────────────────────────────────────────────────────────

    def commit_run_boundary(
        self,
        request: CommitRunBoundaryRequest,
    ) -> CommitRunBoundaryReceipt:
        """Persist a Run boundary through the only Sessions commit protocol."""

        session = self._require_session(request.session_id)
        run = self.repository.load_run(request.run_id)
        if run is None or run.session_id != request.session_id:
            raise FileNotFoundError(f"Run not found for session: {request.run_id}")

        digest = _commit_request_digest(request)
        if run.last_commit_id == request.commit_id:
            return self._repeat_commit_receipt(request, digest, session, run)
        if run.status in {"completed", "failed", "cancelled"}:
            raise SessionStateConflictError(
                f"Terminal run cannot accept another commit: {run.run_id}"
            )
        if session.current_run_id != run.run_id:
            raise SessionStateConflictError(f"Run is not current for session: {run.run_id}")
        self._require_revision(run.revision, request.expected_run_revision, "Run")
        self._require_revision(
            session.revision,
            request.expected_session_revision,
            "Session",
        )
        if run.checkpoint is None:
            raise SessionStateValidationError("Active run has no checkpoint")

        now = _utc_now_iso()
        committed, parent_id = self._commit_run_messages(
            request,
            run,
            now=now,
        )
        target_session_revision = request.expected_session_revision
        if request.kind == "terminal" or parent_id != session.leaf_message_id:
            target_session_revision += 1

        commit_fields = {
            "last_commit_id": request.commit_id,
            "last_commit_kind": request.kind,
            "last_commit_digest": digest,
            "last_commit_session_revision": target_session_revision,
            "last_commit_message_ids": tuple(record.message_id for record in committed),
        }
        if request.kind == "terminal":
            updated_run = self._terminal_run_state(
                request,
                run,
                parent_id=parent_id,
                now=now,
                commit_fields=commit_fields,
            )
        else:
            updated_run = self._checkpoint_run_state(
                request,
                run,
                parent_id=parent_id,
                now=now,
                commit_fields=commit_fields,
            )
        self.repository.update_run(
            updated_run,
            expected_revision=request.expected_run_revision,
        )
        updated_session = self._commit_run_session(
            request,
            session,
            updated_run,
            parent_id=parent_id,
            now=now,
        )
        self._persist_boundary_events(request, updated_session, updated_run)
        return CommitRunBoundaryReceipt(
            commit_id=request.commit_id,
            kind=request.kind,
            session=updated_session,
            run=updated_run,
            committed_messages=tuple(committed),
        )

    # ── 恢复运行 ────────────────────────────────────────────────────────────

    def resume_run(
        self,
        request: ResumeRunRequest,
        *,
        expected_run_revision: int,
    ) -> RunState:
        """恢复被挂起的运行。

        将运行状态从 waiting 转为 running，清除等待状态。
        同时验证检查点和请求 ID 是否匹配（防止并发问题）。

        参数:
            request: 恢复运行请求
            expected_run_revision: 预期的运行 revision

        返回:
            更新后的 RunState
        """
        _, run = self._require_current_run(request.session_id, request.run_id)
        self._require_revision(run.revision, expected_run_revision, "Run")
        if run.status not in {"created", "running", "waiting"} or run.checkpoint is None:
            raise SessionStateValidationError(
                "Only active runs with a checkpoint can be resumed"
            )
        if run.checkpoint.checkpoint_id != request.checkpoint_id:
            raise SessionStateConflictError("Checkpoint changed before resume")
        waiting = run.checkpoint.waiting
        if request.request_id is not None and (
            waiting is None or waiting.request_id != request.request_id
        ):
            raise SessionStateConflictError("Waiting request changed before resume")

        now = _utc_now_iso()
        checkpoint = replace(
            run.checkpoint,
            waiting=None,
            components=_merge_component_checkpoints(
                run.checkpoint.components,
                request.components,
            ),
        )
        updated = replace(
            run,
            status="running",
            checkpoint=checkpoint,
            resume_count=run.resume_count + 1,
            updated_at=now,
            started_at=run.started_at or now,
            revision=run.revision + 1,
        )
        self.repository.update_run(updated, expected_revision=expected_run_revision)
        session = self._require_session(request.session_id)
        self._append_event("run_resumed", session, updated)
        return updated

    # ── 恢复检查 ────────────────────────────────────────────────────────────

    def inspect_recovery(self, request: RecoveryRequest) -> RecoveryResult:
        """检查会话/运行的可恢复性。

        返回包含所有恢复所需数据的 RecoveryBundle，供上层决定如何继续。

        检查内容：
        1. 会话是否存在
        2. 运行是否存在且非终止
        3. 检查点是否匹配期望
        4. 等待状态是否匹配期望
        5. 消息链是否完整
        6. 工作区状态是否变化

        参数:
            request: 恢复请求

        返回:
            RecoveryResult（包含状态、数据包和问题列表）
        """
        session = self.repository.load_session(request.session_id)
        if session is None:
            return RecoveryResult(
                status="not_found",
                issues=(RecoveryIssue("session.not_found", "Session not found"),),
            )
        run_id = request.run_id or session.current_run_id
        if run_id is None:
            return RecoveryResult(
                status="not_found",
                issues=(RecoveryIssue("run.not_found", "No active run"),),
            )
        run = self.repository.load_run(run_id)
        if run is None or run.session_id != session.session_id:
            return RecoveryResult(
                status="blocked",
                issues=(RecoveryIssue("run.invalid", "Run is missing or belongs to another session"),),
            )
        if run.status in {"completed", "failed", "cancelled"} or run.checkpoint is None:
            return RecoveryResult(
                status="blocked",
                issues=(RecoveryIssue("run.terminal", "Run is not recoverable"),),
            )
        if (
            request.expected_checkpoint_id is not None
            and run.checkpoint.checkpoint_id != request.expected_checkpoint_id
        ):
            return RecoveryResult(
                status="blocked",
                issues=(RecoveryIssue("checkpoint.changed", "Checkpoint no longer matches"),),
            )
        waiting_kind = run.checkpoint.waiting.kind if run.checkpoint.waiting is not None else None
        if request.expected_waiting_kind is not None and waiting_kind != request.expected_waiting_kind:
            return RecoveryResult(
                status="blocked",
                issues=(RecoveryIssue("waiting.changed", "Waiting state no longer matches"),),
            )
        try:
            messages = self.repository.load_message_chain(
                session.session_id,
                leaf_id=run.checkpoint.message_cursor.leaf_message_id,
            )
        except ValueError as exc:
            return RecoveryResult(
                status="blocked",
                issues=(RecoveryIssue("messages.invalid", str(exc)),),
            )
        workspace_status = validate_workspace_checkpoint(
            self.workspace_dir,
            run.checkpoint.workspace,
        )
        needs_validation = bool(run.checkpoint.components) or workspace_status.status != "unchanged"
        bundle = RecoveryBundle(
            session=session,
            run=run,
            messages=messages,
            workspace_status=workspace_status,
        )
        return RecoveryResult(
            status="needs_validation" if needs_validation else "ready",
            bundle=bundle,
        )

    # ── 查询方法 ────────────────────────────────────────────────────────────

    def get_session(self, session_id: str) -> SessionState | None:
        """获取会话状态。"""
        return self.repository.load_session(session_id)

    def list_sessions(self) -> tuple[SessionState, ...]:
        """列出所有会话。"""
        return self.repository.list_sessions()

    def delete_session(self, session_id: str) -> bool:
        """删除会话。"""
        return self.repository.delete_session(session_id)

    def update_session_mode(
        self,
        session_id: str,
        mode: str,
        *,
        expected_revision: int,
    ) -> SessionState:
        """更新会话模式。

        参数:
            session_id: 会话 ID
            mode: 新模式（如 "build" / "plan" / "read"）
            expected_revision: 预期的 revision
        """
        session = self._require_session(session_id)
        self._require_revision(session.revision, expected_revision, "Session")
        updated = replace(
            session,
            current_mode=mode,
            updated_at=_utc_now_iso(),
            revision=session.revision + 1,
        )
        return self.repository.update_session(updated, expected_revision=expected_revision)

    def get_run(self, run_id: str) -> RunState | None:
        """获取运行状态。"""
        return self.repository.load_run(run_id)

    def load_messages(
        self,
        session_id: str,
        *,
        leaf_id: str | None = None,
    ) -> tuple[MessageRecord, ...]:
        """加载消息链。"""
        return self.repository.load_message_chain(session_id, leaf_id=leaf_id)

    def list_entry_ids(self, session_id: str) -> list[str]:
        """列出会话中所有消息的 ID。"""
        return [record.message_id for record in self.repository.load_message_records(session_id)]

    def list_entries(self, session_id: str) -> list[dict[str, Any]]:
        """列出会话中所有消息的摘要信息（供 UI 展示）。

        每条摘要包含：ID、父 ID、时间戳、角色、预览文本、是否为叶子节点。
        """
        session = self._require_session(session_id)
        return [
            {
                "id": record.message_id,
                "parent_id": record.parent_id,
                "timestamp": record.created_at,
                "role": getattr(record.message, "role", "unknown"),
                "preview": str(record.message)[:160],
                "is_leaf": record.message_id == session.leaf_message_id,
            }
            for record in self.repository.load_message_records(session_id)
        ]

    def get_entry_path(self, session_id: str, entry_id: str) -> list[str]:
        """获取从根到指定消息的路径（消息 ID 列表）。"""
        return [
            record.message_id
            for record in self.repository.load_message_chain(session_id, leaf_id=entry_id)
        ]

    def get_session_tree(self, session_id: str) -> list[dict[str, Any]]:
        """构建会话的完整消息树（供 UI 展示分叉结构）。"""
        by_id = {
            item["id"]: {**item, "children": []}
            for item in self.list_entries(session_id)
        }
        roots: list[dict[str, Any]] = []
        for item in by_id.values():
            parent = item.get("parent_id")
            if isinstance(parent, str) and parent in by_id:
                by_id[parent]["children"].append(item)
            else:
                roots.append(item)
        return roots

    def append_message(
        self,
        session_id: str,
        message: Message,
        *,
        run_id: str | None = None,
    ) -> MessageRecord:
        """向会话追加一条消息（不绑定运行）。

        参数:
            session_id: 会话 ID
            message: 协议消息
            run_id: 运行 ID（可选）

        返回:
            创建的消息记录
        """
        session = self._require_session(session_id)
        now = _utc_now_iso()
        record = MessageRecord(
            message_id=_new_id("msg"),
            session_id=session_id,
            run_id=run_id,
            parent_id=session.leaf_message_id,
            created_at=now,
            message=message,
        )
        self.repository.append_message(record)
        self.repository.update_session(
            replace(
                session,
                leaf_message_id=record.message_id,
                updated_at=now,
                revision=session.revision + 1,
            ),
            expected_revision=session.revision,
        )
        return record

    def set_leaf(self, session_id: str, entry_id: str) -> SessionState:
        """设置会话的叶子消息 ID（用于消息树的分叉导航）。

        参数:
            session_id: 会话 ID
            entry_id: 新的叶子消息 ID

        返回:
            更新后的会话状态
        """
        self.repository.load_message_chain(session_id, leaf_id=entry_id)
        session = self._require_session(session_id)
        updated = replace(
            session,
            leaf_message_id=entry_id,
            updated_at=_utc_now_iso(),
            revision=session.revision + 1,
        )
        return self.repository.update_session(updated, expected_revision=session.revision)

    def fork_session(
        self,
        session_id: str,
        new_session_id: str,
        *,
        from_entry_id: str | None = None,
    ) -> SessionState:
        """分叉会话 —— 从源会话的某个消息点创建新会话。

        新会话继承源会话的工作区和模型等配置，
        但拥有自己的消息链（从源会话的指定叶子节点开始）。

        参数:
            session_id: 源会话 ID
            new_session_id: 新会话 ID
            from_entry_id: 从哪个消息处分叉（默认使用当前叶子节点）

        返回:
            新创建的会话状态
        """
        source = self._require_session(session_id)
        chain = self.repository.load_message_chain(session_id, leaf_id=from_entry_id)
        now = _utc_now_iso()
        target = replace(
            source,
            session_id=new_session_id,
            parent_session_id=session_id,
            current_run_id=None,
            last_run_id=None,
            leaf_message_id=chain[-1].message_id if chain else None,
            created_at=now,
            updated_at=now,
            revision=1,
        )
        self.repository.create_session(target)
        for record in chain:
            self.repository.append_message(
                replace(record, session_id=new_session_id, run_id=None)
            )
        self.append_event(
            new_session_id,
            {"type": "session_forked", "parent_session_id": session_id},
        )
        return target

    # ── 事件操作 ────────────────────────────────────────────────────────────

    def append_event(
        self,
        session_id: str,
        event: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """追加一个事件。

        对事件数据进行标准化（normalize）和脱敏（redact）处理后写入。

        参数:
            session_id: 会话 ID
            event: 事件数据
            run_id: 运行 ID（可选）

        返回:
            标准化后的事件数据
        """
        self._require_session(session_id)
        payload = redact_artifact(normalize_event_value(dict(event)))
        legacy_keys = {"eventId", "sessionId", "runId"}.intersection(payload)
        if legacy_keys:
            raise ValueError(
                "Event uses legacy field names: " + ", ".join(sorted(legacy_keys))
            )
        payload["event_id"] = str(payload.get("event_id") or _new_id("event"))
        payload["session_id"] = session_id
        effective_run_id = run_id or payload.get("run_id")
        if isinstance(effective_run_id, str) and effective_run_id:
            payload["run_id"] = effective_run_id
        else:
            payload.pop("run_id", None)
        for existing in self.repository.load_events(session_id):
            if existing.get("event_id") != payload["event_id"]:
                continue
            payload.setdefault("created_at", existing.get("created_at"))
            if payload == existing:
                return existing
            raise SessionStateConflictError(
                f"event_id was reused with different content: {payload['event_id']}"
            )
        payload.setdefault("created_at", _utc_now_iso())
        self.repository.append_event(payload)
        return payload

    def load_events(
        self,
        session_id: str,
        *,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """加载事件。"""
        return list(self.repository.load_events(session_id, run_id=run_id, limit=limit))

    def load_run_view(self, session_id: str, run_id: str) -> dict[str, Any]:
        """加载运行的完整视图（含回滚元数据）。"""
        run = self.repository.load_run(run_id)
        if run is None or run.session_id != session_id:
            raise FileNotFoundError(f"Run not found: {run_id}")
        payload = run_state_to_dict(run)
        rollback = self.read_rollback_metadata(run_id)
        if rollback is not None:
            payload["rollback"] = rollback
        return payload

    def load_last_run_view(self, session_id: str) -> dict[str, Any] | None:
        """加载最近一次运行的视图。"""
        session = self._require_session(session_id)
        return (
            self.load_run_view(session_id, session.last_run_id)
            if session.last_run_id is not None
            else None
        )

    def write_rollback_metadata(self, run_id: str, metadata: dict[str, Any]) -> None:
        """写入回滚元数据。"""
        from .filesystem import atomic_write_json

        atomic_write_json(
            self.repository.layout.run_artifacts_dir(run_id) / "rollback.json",
            redact_artifact(metadata),
        )

    def read_rollback_metadata(self, run_id: str) -> dict[str, Any] | None:
        """读取回滚元数据。"""
        from .filesystem import read_json_object

        return read_json_object(
            self.repository.layout.run_artifacts_dir(run_id) / "rollback.json"
        )

    def _commit_run_messages(
        self,
        request: CommitRunBoundaryRequest,
        run: RunState,
        *,
        now: str,
    ) -> tuple[list[MessageRecord], str]:
        if run.checkpoint is None:
            raise SessionStateValidationError("Active run has no checkpoint")
        existing = {
            item.message_id: item
            for item in self.repository.load_message_records(request.session_id)
        }
        parent_id = run.checkpoint.message_cursor.leaf_message_id
        committed: list[MessageRecord] = []
        for index, message in enumerate(request.new_messages):
            message_id = _commit_child_id("msg", request.commit_id, index)
            current = existing.get(message_id)
            record = MessageRecord(
                message_id=message_id,
                session_id=request.session_id,
                run_id=request.run_id,
                parent_id=parent_id,
                created_at=current.created_at if current is not None else now,
                message=message,
            )
            committed.append(self.repository.append_message(record))
            parent_id = message_id
        return committed, parent_id

    def _checkpoint_run_state(
        self,
        request: CommitRunBoundaryRequest,
        run: RunState,
        *,
        parent_id: str,
        now: str,
        commit_fields: dict[str, object],
    ) -> RunState:
        checkpoint = RunCheckpoint(
            checkpoint_id=_commit_child_id("checkpoint", request.commit_id, 0),
            resume_point=request.resume_point,  # type: ignore[arg-type]
            message_cursor=MessageCursor(parent_id),
            waiting=request.waiting,
            components=request.components,
            workspace=request.workspace,
            created_at=now,
        )
        return replace(
            run,
            status="waiting" if request.kind == "waiting" else "running",
            phase=request.phase,  # type: ignore[arg-type]
            latest_message_id=parent_id,
            core_state=request.core_state,
            checkpoint=checkpoint,
            updated_at=now,
            started_at=run.started_at or now,
            revision=run.revision + 1,
            **commit_fields,
        )

    def _terminal_run_state(
        self,
        request: CommitRunBoundaryRequest,
        run: RunState,
        *,
        parent_id: str,
        now: str,
        commit_fields: dict[str, object],
    ) -> RunState:
        result_ref: str | None = None
        if request.result is not None:
            result_payload = normalize_event_value(request.result)
            if not isinstance(result_payload, dict):
                raise SessionStateValidationError("AgentRunResult must serialize to an object")
            result_ref = self.repository.write_result_artifact(run.run_id, result_payload)
        return replace(
            run,
            status=request.terminal_status,  # type: ignore[arg-type]
            phase="finished",
            stop_reason=request.stop_reason,
            latest_message_id=parent_id,
            checkpoint=None,
            result_ref=result_ref,
            workspace_effects=request.workspace_effects or run.workspace_effects,
            updated_at=now,
            ended_at=now,
            revision=run.revision + 1,
            **commit_fields,
        )

    def _commit_run_session(
        self,
        request: CommitRunBoundaryRequest,
        session: SessionState,
        run: RunState,
        *,
        parent_id: str,
        now: str,
    ) -> SessionState:
        if request.kind != "terminal" and parent_id == session.leaf_message_id:
            return session
        updated = replace(
            session,
            current_run_id=None if request.kind == "terminal" else run.run_id,
            last_run_id=run.run_id,
            leaf_message_id=parent_id,
            updated_at=now,
            revision=session.revision + 1,
        )
        self.repository.update_session(
            updated,
            expected_revision=request.expected_session_revision,
        )
        return updated

    def _persist_boundary_events(
        self,
        request: CommitRunBoundaryRequest,
        session: SessionState,
        run: RunState,
    ) -> None:
        for event in request.durable_events:
            self.append_event(session.session_id, event, run_id=run.run_id)
        self.append_event(
            session.session_id,
            {
                "type": f"run_boundary_{request.kind}_committed",
                "event_id": f"{request.commit_id}:receipt",
                "commit_id": request.commit_id,
                "session_revision": session.revision,
                "run_revision": run.revision,
            },
            run_id=run.run_id,
        )

    def _repeat_commit_receipt(
        self,
        request: CommitRunBoundaryRequest,
        digest: str,
        session: SessionState,
        run: RunState,
    ) -> CommitRunBoundaryReceipt:
        if run.last_commit_kind != request.kind or run.last_commit_digest != digest:
            raise SessionStateConflictError(
                f"commit_id was reused with different content: {request.commit_id}"
            )
        target_revision = run.last_commit_session_revision
        if target_revision is None:
            raise SessionStateValidationError("Committed Run is missing Session revision")
        if session.revision < target_revision:
            recovered = replace(
                session,
                current_run_id=(
                    None
                    if request.kind == "terminal"
                    else run.run_id
                ),
                last_run_id=run.run_id,
                leaf_message_id=run.latest_message_id,
                updated_at=run.updated_at,
                revision=target_revision,
            )
            self.repository.update_session(
                recovered,
                expected_revision=session.revision,
            )
            session = recovered
        elif session.revision > target_revision:
            raise SessionStateConflictError(
                f"Session advanced after commit: {request.commit_id}"
            )
        self._persist_boundary_events(request, session, run)
        by_id = {
            item.message_id: item
            for item in self.repository.load_message_records(request.session_id)
        }
        try:
            committed = tuple(by_id[message_id] for message_id in run.last_commit_message_ids)
        except KeyError as exc:
            raise SessionStateValidationError(
                f"Committed message is missing: {exc.args[0]}"
            ) from exc
        return CommitRunBoundaryReceipt(
            commit_id=request.commit_id,
            kind=request.kind,
            session=session,
            run=run,
            committed_messages=committed,
        )

    # ── 内部辅助方法 ────────────────────────────────────────────────────────

    def _require_session(self, session_id: str) -> SessionState:
        """加载会话，如果不存在则抛出 FileNotFoundError。"""
        session = self.repository.load_session(session_id)
        if session is None:
            raise FileNotFoundError(f"Session not found: {session_id}")
        return session

    def _require_current_run(self, session_id: str, run_id: str) -> tuple[SessionState, RunState]:
        """加载会话并验证指定的运行是当前活动运行。"""
        session = self._require_session(session_id)
        if session.current_run_id != run_id:
            raise SessionStateConflictError(f"Run is not current for session: {run_id}")
        run = self.repository.load_run(run_id)
        if run is None or run.session_id != session_id:
            raise FileNotFoundError(f"Run not found for session: {run_id}")
        return session, run

    @staticmethod
    def _require_revision(actual: int, expected: int, name: str) -> None:
        """验证 revision 匹配 —— 乐观锁的核心逻辑。"""
        if actual != expected:
            raise SessionStateConflictError(
                f"{name} revision conflict: expected {expected}, got {actual}"
            )

    def _append_event(self, event_type: str, session: SessionState, run: RunState) -> None:
        """记录一个事件并关联到会话和运行。"""
        self.append_event(
            session.session_id,
            {
                "type": event_type,
                "session_revision": session.revision,
                "run_revision": run.revision,
            },
            run_id=run.run_id,
        )


# ── 工具函数 ──────────────────────────────────────────────────────────────────


def _commit_request_digest(request: CommitRunBoundaryRequest) -> str:
    payload = normalize_event_value(
        {
            "commit_id": request.commit_id,
            "kind": request.kind,
            "session_id": request.session_id,
            "run_id": request.run_id,
            "expected_run_revision": request.expected_run_revision,
            "expected_session_revision": request.expected_session_revision,
            "phase": request.phase,
            "resume_point": request.resume_point,
            "core_state": request.core_state,
            "new_messages": request.new_messages,
            "durable_events": request.durable_events,
            "waiting": request.waiting,
            "components": request.components,
            "workspace": request.workspace,
            "terminal_status": request.terminal_status,
            "stop_reason": request.stop_reason,
            "result": request.result,
            "workspace_effects": request.workspace_effects,
        }
    )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _commit_child_id(prefix: str, commit_id: str, index: int) -> str:
    digest = hashlib.sha256(f"{commit_id}:{index}".encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _request_scoped_id(prefix: str, session_id: str, request_id: str) -> str:
    digest = hashlib.sha256(f"{session_id}\0{request_id}".encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:24]}"


def _begin_run_request_digest(request: BeginRunRequest) -> str:
    payload = {
        "session_id": request.session_id,
        "message": message_to_dict(request.user_message),
        "initial_core_state": request.initial_core_state,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required")
    return value.strip()


def new_session_id() -> str:
    """生成一个新的会话 ID。"""
    return _new_id("session")


def _new_id(prefix: str) -> str:
    """生成带前缀的唯一 ID（UUID 前 12 位十六进制）。"""
    return f"{prefix}_{uuid4().hex[:12]}"


def _utc_now_iso() -> str:
    """获取当前 UTC 时间的 ISO 8601 字符串表示。"""
    return datetime.now(timezone.utc).isoformat()


def _merge_component_checkpoints(
    current: tuple[ComponentCheckpoint, ...],
    replacements: tuple[ComponentCheckpoint, ...],
) -> tuple[ComponentCheckpoint, ...]:
    if not replacements:
        return current
    replacement_by_owner = {component.owner: component for component in replacements}
    merged = [
        replacement_by_owner.pop(component.owner, component)
        for component in current
    ]
    merged.extend(replacement_by_owner.values())
    return tuple(merged)


__all__ = [
    "BeginRunRequest",
    "BeginRunResult",
    "CommitRunBoundaryReceipt",
    "CommitRunBoundaryRequest",
    "CreateSessionRequest",
    "ResumeRunRequest",
    "SessionStateService",
    "new_session_id",
]
