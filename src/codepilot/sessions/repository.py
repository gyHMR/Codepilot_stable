"""文件系统会话仓库 —— 基于文件的 Session、Run、Message 持久化操作。

FileSessionRepository 是 SessionStateService 的底层数据访问层，
将对 SessionState、RunState、MessageRecord 的 CRUD 操作映射为
文件系统上的 JSON/JSONL 文件读写。

核心设计：
- 乐观锁（revision）：每次更新要求提供 expected_revision，防止并发覆盖
- 消息链：通过 parent_id 构建消息链表，支持分叉（fork）
- 原子写入：所有写操作通过 atomic_write_json 确保原子性
- 运行事件：事件同时写入会话事件文件和运行事件文件
"""

import json
import shutil
from pathlib import Path
from typing import Any

from .contracts import (
    MessageRecord,
    RunState,
    SessionState,
    SessionStateConflictError,
    validate_run_transition,
)
from .filesystem import (
    SessionFileLayout,
    append_jsonl,
    atomic_write_json,
    read_json_object,
    read_jsonl,
)
from .serde import (
    message_record_from_dict,
    message_record_to_dict,
    run_state_from_dict,
    run_state_to_dict,
    session_state_from_dict,
    session_state_to_dict,
)


class FileSessionRepository:
    """文件系统会话仓库 —— 对 Sessions v2 状态的类型化文件系统访问。

    将 SessionState、RunState、MessageRecord 等类型对象
    序列化为 JSON/JSONL 文件并持久化到 .codepilot 目录。

    参数:
        workspace_dir: 工作区根目录（路径字符串或 Path）
    """

    def __init__(self, workspace_dir: str | Path) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.layout = SessionFileLayout(self.workspace_dir)

    # ── Session 操作 ──────────────────────────────────────────────────────────

    def create_session(self, state: SessionState) -> SessionState:
        """创建新的会话状态文件。

        如果会话文件已存在，抛出 SessionStateConflictError。

        参数:
            state: 要创建的会话状态

        返回:
            写入后的会话状态
        """
        path = self.layout.session_file(state.session_id)
        if path.exists():
            raise SessionStateConflictError(f"Session already exists: {state.session_id}")
        atomic_write_json(path, session_state_to_dict(state))
        self.layout.messages_file(state.session_id).touch(exist_ok=True)
        self.layout.session_events_file(state.session_id).touch(exist_ok=True)
        return state

    def load_session(self, session_id: str) -> SessionState | None:
        """加载会话状态。

        参数:
            session_id: 会话 ID

        返回:
            SessionState 或 None（未找到时）
        """
        payload = read_json_object(self.layout.session_file(session_id))
        return session_state_from_dict(payload) if payload is not None else None

    def list_sessions(self) -> tuple[SessionState, ...]:
        """列出所有会话（按更新时间降序）。

        遍历 .codepilot/sessions/ 目录下的所有子目录，
        读取每个子目录中的 session.json。

        返回:
            按更新时间降序排列的 SessionState 元组
        """
        sessions_dir = self.layout.codepilot_dir / "sessions"
        if not sessions_dir.is_dir():
            return ()
        sessions = [
            session
            for entry in sessions_dir.iterdir()
            if entry.is_dir()
            for session in [self.load_session(entry.name)]
            if session is not None
        ]
        return tuple(sorted(sessions, key=lambda item: item.updated_at, reverse=True))

    def delete_session(self, session_id: str) -> bool:
        """删除会话及其所有数据文件。

        安全性：对 session_id 做路径检查，防止路径遍历攻击。

        参数:
            session_id: 要删除的会话 ID

        返回:
            True 表示删除成功，False 表示会话不存在
        """
        if not session_id or Path(session_id).name != session_id:
            raise ValueError("Invalid session_id")
        sessions_dir = (self.layout.codepilot_dir / "sessions").resolve()
        target = self.layout.session_dir(session_id).resolve()
        if target.parent != sessions_dir:
            raise ValueError("Session path escapes the sessions directory")
        if not target.is_dir():
            return False
        runs_dir = (self.layout.codepilot_dir / "runs").resolve()
        owned_run_dirs: list[Path] = []
        for run in self.list_runs(session_id=session_id):
            run_dir = self.layout.run_dir(run.run_id).resolve()
            if run_dir.parent != runs_dir:
                raise ValueError("Run path escapes the runs directory")
            owned_run_dirs.append(run_dir)
        for run_dir in owned_run_dirs:
            if run_dir.is_dir():
                shutil.rmtree(run_dir)
        shutil.rmtree(target)
        return True

    def update_session(
        self,
        state: SessionState,
        *,
        expected_revision: int,
    ) -> SessionState:
        """更新会话状态（乐观锁）。

        读取当前状态，验证 revision 是否匹配 expected_revision，
        然后写入新状态。

        参数:
            state: 更新后的会话状态
            expected_revision: 预期的当前 revision（乐观锁）

        返回:
            写入后的会话状态
        """
        current = self.load_session(state.session_id)
        if current is None:
            raise FileNotFoundError(f"Session not found: {state.session_id}")
        if current.revision != expected_revision:
            raise SessionStateConflictError(
                f"Session revision conflict: expected {expected_revision}, got {current.revision}"
            )
        if state.revision != expected_revision + 1:
            raise SessionStateConflictError("Session revision must increase by exactly one")
        atomic_write_json(self.layout.session_file(state.session_id), session_state_to_dict(state))
        return state

    # ── Run 操作 ─────────────────────────────────────────────────────────────

    def create_run(self, state: RunState) -> RunState:
        """创建新的运行状态文件。

        参数:
            state: 要创建的运行状态

        返回:
            写入后的运行状态
        """
        path = self.layout.run_file(state.run_id)
        if path.exists():
            raise SessionStateConflictError(f"Run already exists: {state.run_id}")
        atomic_write_json(path, run_state_to_dict(state))
        self.layout.run_events_file(state.run_id).touch(exist_ok=True)
        return state

    def load_run(self, run_id: str) -> RunState | None:
        """加载运行状态。

        参数:
            run_id: 运行 ID

        返回:
            RunState 或 None
        """
        payload = read_json_object(self.layout.run_file(run_id))
        return run_state_from_dict(payload) if payload is not None else None

    def list_runs(self, *, session_id: str | None = None) -> tuple[RunState, ...]:
        """List persisted Runs, optionally restricted to one Session."""

        runs_dir = self.layout.codepilot_dir / "runs"
        if not runs_dir.is_dir():
            return ()
        runs = [
            run
            for entry in runs_dir.iterdir()
            if entry.is_dir()
            for run in [self.load_run(entry.name)]
            if run is not None and (session_id is None or run.session_id == session_id)
        ]
        return tuple(sorted(runs, key=lambda item: item.created_at))

    def update_run(
        self,
        state: RunState,
        *,
        expected_revision: int,
    ) -> RunState:
        """更新运行状态（乐观锁 + 转换验证）。

        读取当前状态，验证 revision 和状态转换的合法性，
        然后写入新状态。

        参数:
            state: 更新后的运行状态
            expected_revision: 预期的当前 revision

        返回:
            写入后的运行状态
        """
        current = self.load_run(state.run_id)
        if current is None:
            raise FileNotFoundError(f"Run not found: {state.run_id}")
        if current.revision != expected_revision:
            raise SessionStateConflictError(
                f"Run revision conflict: expected {expected_revision}, got {current.revision}"
            )
        validate_run_transition(current, state)
        atomic_write_json(self.layout.run_file(state.run_id), run_state_to_dict(state))
        return state

    # ── Message 操作 ─────────────────────────────────────────────────────────

    def append_message(self, record: MessageRecord) -> MessageRecord:
        """追加一条消息记录。

        幂等处理：如果消息 ID 已存在且内容相同，返回已存在的记录（忽略重复）。
        如果消息 ID 已存在但内容不同，抛出 SessionStateConflictError。

        参数:
            record: 消息记录

        返回:
            写入后的消息记录
        """
        path = self.layout.messages_file(record.session_id)
        existing = {item.message_id: item for item in self.load_message_records(record.session_id)}
        current = existing.get(record.message_id)
        if current is not None:
            if message_record_to_dict(current) == message_record_to_dict(record):
                return current
            raise SessionStateConflictError(f"Message id already exists: {record.message_id}")
        append_jsonl(path, message_record_to_dict(record))
        return record

    def load_message_records(self, session_id: str) -> list[MessageRecord]:
        """加载会话的所有消息记录。

        参数:
            session_id: 会话 ID

        返回:
            所有消息记录的列表（JSONL 中原始顺序）
        """
        return [
            message_record_from_dict(row)
            for row in read_jsonl(self.layout.messages_file(session_id))
        ]

    def load_message_chain(
        self,
        session_id: str,
        *,
        leaf_id: str | None = None,
    ) -> tuple[MessageRecord, ...]:
        """加载从根到叶子节点的消息链。

        从 leaf_id 开始，通过 parent_id 向上回溯到根消息。
        结果按从根到叶子的顺序排列。

        循环检测：如果消息链有环（某个消息的 parent_id 指向自己的后代），
        抛出 ValueError。

        参数:
            session_id: 会话 ID
            leaf_id: 叶子消息的 ID（可选，默认使用 session.leaf_message_id）

        返回:
            从根到叶子的 MessageRecord 元组
        """
        records = self.load_message_records(session_id)
        if not records:
            return ()
        by_id = {record.message_id: record for record in records}
        current = leaf_id
        if current is None:
            session = self.load_session(session_id)
            current = session.leaf_message_id if session is not None else None
        if current is None:
            return ()
        if current not in by_id:
            raise ValueError(f"Message leaf not found: {current}")
        chain: list[MessageRecord] = []
        seen: set[str] = set()
        while current is not None:
            if current in seen:
                raise ValueError(f"Message chain contains a cycle at: {current}")
            seen.add(current)
            record = by_id.get(current)
            if record is None:
                raise ValueError(f"Message parent not found: {current}")
            chain.append(record)
            current = record.parent_id
        chain.reverse()
        return tuple(chain)

    # ── Event 操作 ──────────────────────────────────────────────────────────

    def append_event(self, event: dict[str, Any]) -> None:
        """追加一个事件。

        事件同时写入会话事件文件和运行事件文件。
        如果事件包含 run_id，还会写入运行级别的事件文件。

        参数:
            event: 事件数据字典（必须包含 event_id、session_id）
        """
        legacy_keys = {"eventId", "sessionId", "runId"}.intersection(event)
        if legacy_keys:
            raise ValueError(
                "Event uses legacy field names: " + ", ".join(sorted(legacy_keys))
            )
        _event_text(event, "event_id")
        session_id = _event_text(event, "session_id")
        append_jsonl(self.layout.session_events_file(session_id), event)
        run_id = event.get("run_id")
        if isinstance(run_id, str) and run_id:
            append_jsonl(self.layout.run_events_file(run_id), event)

    def load_events(
        self,
        session_id: str,
        *,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """加载事件。

        如果指定了 run_id，从运行事件文件加载；
        否则从会话事件文件加载。

        支持 limit 参数限制返回的事件数量（取最新的 N 条）。

        参数:
            session_id: 会话 ID
            run_id: 运行 ID（可选）
            limit: 最大返回数量

        返回:
            事件字典元组
        """
        path = (
            self.layout.run_events_file(run_id)
            if run_id is not None
            else self.layout.session_events_file(session_id)
        )
        events = read_jsonl(path)
        if run_id is not None:
            events = [event for event in events if event.get("session_id") == session_id]
        return tuple(events[-limit:] if limit is not None else events)

    def write_result_artifact(self, run_id: str, payload: dict[str, Any]) -> str:
        """写入运行结果文件（artifact）。

        参数:
            run_id: 运行 ID
            payload: 结果数据

        返回:
            相对于运行目录的 artifact 路径
        """
        path = self.layout.run_artifacts_dir(run_id) / "result.json"
        atomic_write_json(path, payload)
        return "artifacts/result.json"


def _event_text(event: dict[str, Any], key: str) -> str:
    """验证事件字典中某字段的合法性。

    必须是非空字符串，且整个事件必须可 JSON 序列化。
    """
    value = event.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Event {key} is required")
    try:
        json.dumps(event, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Event must be JSON serializable") from exc
    return value.strip()


__all__ = ["FileSessionRepository"]
