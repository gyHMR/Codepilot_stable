"""工具执行状态 —— ToolRuntime 审批和恢复流程使用的状态端口。

本文件定义了工具尝试（attempt）的状态管理和数据模型：
1. ToolAttemptState     — 工具尝试的生命周期状态枚举
2. ToolAttemptRecord    — 单次工具尝试的完整状态记录
3. ToolStateStore       — 状态存储协议接口（支持持久化注入）
4. InMemoryToolStateStore — 默认的内存状态存储
5. InteractionRequest/Response — 用户交互请求与响应
6. 辅助函数 — attempt_id_for()、transition()、build_interaction_request()
"""

from dataclasses import dataclass, replace
import hashlib
import json
from threading import RLock
import time
from types import MappingProxyType
from typing import Literal, Mapping, Protocol
from uuid import uuid4

from .contracts import ToolExecutionRequest
from .results import ToolResult
from .security import ApprovalChallenge, ApprovalGrant, ToolAccessRequest


# ── 类型别名 ──────────────────────────────────────────────────────────────────


ToolAttemptState = Literal[
    "received",           # 已收到请求
    "validating",         # 验证中（参数、注册）
    "resolving_access",   # 解析访问权限中
    "awaiting_approval",  # 等待用户审批
    "awaiting_input",     # 等待用户输入
    "queued",             # 已排队（等待执行）
    "running",            # 执行中
    "succeeded",          # 执行成功
    "failed",             # 执行失败
    "denied",             # 被拒绝
    "timed_out",          # 超时
    "cancelled",          # 被取消
    "interrupted",        # 被中断（同一批中的前驱工具挂起时）
]


# ── 交互类型 ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class InteractionRequest:
    """用户交互请求 —— 工具需要用户输入时创建的请求。

    某些工具（如交互式选择工具）在执行过程中需要用户提供输入。
    此对象描述需要用户回答的问题和可选的选项。

    参数:
        interaction_id: 交互的唯一 ID
        request_fingerprint: 请求的指纹（用于验证响应）
        session_id: 会话 ID
        tool_call_id: 工具调用 ID
        tool_name: 工具名称
        registration_id: 注册 ID
        prompt: 向用户展示的提示文本
        options: 预定义的选项列表（可选）
        allow_free_text: 是否允许用户自由输入
        created_at_ms: 创建时间戳
    """

    interaction_id: str
    request_fingerprint: str
    session_id: str
    tool_call_id: str
    tool_name: str
    registration_id: str
    prompt: str
    options: tuple[str, ...] = ()
    allow_free_text: bool = True
    created_at_ms: int = 0

    def __post_init__(self) -> None:
        for name in (
            "interaction_id",
            "request_fingerprint",
            "session_id",
            "tool_call_id",
            "tool_name",
            "registration_id",
            "prompt",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        options = tuple(str(value).strip() for value in self.options)
        if any(not value for value in options):
            raise ValueError("interaction options cannot contain empty values")
        if len(options) != len(set(options)):
            raise ValueError("interaction options must be unique")
        if not isinstance(self.allow_free_text, bool):
            raise TypeError("allow_free_text must be bool")
        if isinstance(self.created_at_ms, bool) or not isinstance(self.created_at_ms, int):
            raise TypeError("created_at_ms must be int")
        object.__setattr__(self, "options", options)

    def to_dict(self) -> dict[str, object]:
        """将交互请求转为可序列化的字典。"""
        return {
            "interaction_id": self.interaction_id,
            "request_fingerprint": self.request_fingerprint,
            "session_id": self.session_id,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "registration_id": self.registration_id,
            "prompt": self.prompt,
            "options": list(self.options),
            "allow_free_text": self.allow_free_text,
            "created_at_ms": self.created_at_ms,
        }


@dataclass(frozen=True)
class InteractionResponse:
    """用户交互响应 —— 用户对交互请求的回复。

    参数:
        interaction_id: 对应的交互请求 ID
        request_fingerprint: 请求指纹
        session_id: 会话 ID
        tool_call_id: 工具调用 ID
        tool_name: 工具名称
        registration_id: 注册 ID
        answers: 用户提供的答案映射
    """

    interaction_id: str
    request_fingerprint: str
    session_id: str
    tool_call_id: str
    tool_name: str
    registration_id: str
    answers: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in (
            "interaction_id",
            "request_fingerprint",
            "session_id",
            "tool_call_id",
            "tool_name",
            "registration_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        if not isinstance(self.answers, Mapping) or not self.answers:
            raise ValueError("interaction answers must be a non-empty mapping")
        copied = json.loads(json.dumps(dict(self.answers), ensure_ascii=False))
        object.__setattr__(self, "answers", MappingProxyType(copied))


def build_interaction_request(
    request: ToolExecutionRequest,
    *,
    prompt: str,
    options: tuple[str, ...] = (),
    allow_free_text: bool = True,
) -> InteractionRequest:
    """从执行请求构建交互请求。

    生成 request_fingerprint（基于请求内容的 SHA256 哈希）
    和 interaction_id（UUID），用于验证响应的合法性。

    参数:
        request: 原始的工具执行请求
        prompt: 向用户展示的提示文本
        options: 预定义的选项（可选）
        allow_free_text: 是否允许自由输入

    返回:
        InteractionRequest 对象
    """
    payload = {
        "session_id": request.session_id,
        "tool_call_id": request.tool_call_id,
        "tool_name": request.tool_name,
        "registration_id": request.registration_id,
        "prompt": str(prompt).strip(),
        "options": list(options),
        "allow_free_text": allow_free_text,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return InteractionRequest(
        interaction_id=f"interaction_{uuid4().hex[:20]}",
        request_fingerprint=fingerprint,
        session_id=request.session_id,
        tool_call_id=request.tool_call_id,
        tool_name=request.tool_name,
        registration_id=request.registration_id,
        prompt=payload["prompt"],
        options=options,
        allow_free_text=allow_free_text,
        created_at_ms=int(time.time() * 1000),
    )


# ── 状态存储 ──────────────────────────────────────────────────────────────────


class ToolStateConflictError(RuntimeError):
    """工具状态冲突 —— compare-and-set 操作发现状态与预期不符。

    当两个并发操作试图同时修改同一个 attempt 的状态时触发，
    后到的操作会看到 "预期状态 ≠ 实际状态"，从而抛出此异常。
    """


@dataclass(frozen=True)
class ToolAttemptRecord:
    """工具尝试记录 —— 一次工具调用的完整状态数据。

    每次工具调用从接受请求到返回结果，
    其状态变化都记录在此对象中。

    参数:
        attempt_id: 尝试的唯一 ID（格式: session_id:run_id:tool_call_id）
        request: 原始执行请求
        state: 当前状态
        challenge: 审批挑战（当需要审批时设置）
        interaction: 交互请求（当需要用户输入时设置）
        interaction_consumed: 交互是否已被消费
        grant: 审批授权（用户批准后设置）
        grant_consumed: 授权是否已被消费
        result: 执行结果
        cleanup_errors: 清理操作中的错误列表
    """

    attempt_id: str
    request: ToolExecutionRequest
    state: ToolAttemptState = "received"
    challenge: ApprovalChallenge | None = None
    interaction: InteractionRequest | None = None
    interaction_consumed: bool = False
    grant: ApprovalGrant | None = None
    grant_consumed: bool = False
    result: ToolResult | None = None
    cleanup_errors: tuple[str, ...] = ()


class ToolStateStore(Protocol):
    """工具状态存储协议 —— 持久化存储的接口定义。

    提供操作：
    - create(): 创建新的 attempt 记录
    - get(): 查询 attempt 记录
    - find_by_approval_id(): 按审批 ID 查找
    - find_by_interaction_id(): 按交互 ID 查找
    - pending_challenges(): 获取所有待审批的挑战
    - find_reusable_grant(): 查找可重用的审批授权
    - compare_and_set(): CAS 更新（保证并发安全）
    """

    def create(self, record: ToolAttemptRecord) -> None:
        """创建一个新的 attempt 记录。"""

    def get(self, attempt_id: str) -> ToolAttemptRecord | None:
        """按 ID 查询 attempt 记录。"""

    def find_by_approval_id(self, approval_id: str) -> ToolAttemptRecord | None:
        """按审批挑战 ID 查找对应的 attempt 记录。"""

    def find_by_interaction_id(self, interaction_id: str) -> ToolAttemptRecord | None:
        """按交互请求 ID 查找对应的 attempt 记录。"""

    def pending_challenges(self) -> tuple[ApprovalChallenge, ...]:
        """返回所有处于 awaiting_approval 状态的审批挑战。"""

    def find_reusable_grant(
        self,
        request: ToolExecutionRequest,
        access: ToolAccessRequest,
    ) -> ApprovalGrant | None:
        """查找可重用的审批授权（scope=session/project 的缓存的授权）。"""

    def compare_and_set(
        self,
        attempt_id: str,
        expected_state: ToolAttemptState,
        record: ToolAttemptRecord,
    ) -> None:
        """比较并设置（CAS 操作）：只有在当前状态等于 expected_state 时更新。"""


class InMemoryToolStateStore:
    """默认的内存状态存储 —— 使用线程锁保证并发安全。

    这是默认的工具状态存储实现，适合单进程场景。
    Session 可以注入持久化的实现（如基于文件或数据库的存储）。

    状态索引：
    - _attempts: attempt_id → ToolAttemptRecord
    - _approval_index: approval_id → attempt_id
    - _interaction_index: interaction_id → attempt_id
    """

    def __init__(self) -> None:
        self._attempts: dict[str, ToolAttemptRecord] = {}
        self._approval_index: dict[str, str] = {}
        self._interaction_index: dict[str, str] = {}
        self._lock = RLock()

    def create(self, record: ToolAttemptRecord) -> None:
        """创建一个新的 attempt 记录。

        如果 attempt_id 已存在，抛出 ToolStateConflictError。

        参数:
            record: 新记录

        抛出:
            ToolStateConflictError: 记录已存在
        """
        with self._lock:
            if record.attempt_id in self._attempts:
                raise ToolStateConflictError(f"Tool attempt already exists: {record.attempt_id}")
            self._attempts[record.attempt_id] = record
            self._index(record)

    def get(self, attempt_id: str) -> ToolAttemptRecord | None:
        """按 ID 查询 attempt 记录。"""
        with self._lock:
            return self._attempts.get(attempt_id)

    def find_by_approval_id(self, approval_id: str) -> ToolAttemptRecord | None:
        """按审批挑战 ID 查找 attempt 记录。"""
        with self._lock:
            attempt_id = self._approval_index.get(approval_id)
            return self._attempts.get(attempt_id) if attempt_id is not None else None

    def find_by_interaction_id(self, interaction_id: str) -> ToolAttemptRecord | None:
        """按交互请求 ID 查找 attempt 记录。"""
        with self._lock:
            attempt_id = self._interaction_index.get(interaction_id)
            return self._attempts.get(attempt_id) if attempt_id is not None else None

    def pending_challenges(self) -> tuple[ApprovalChallenge, ...]:
        """返回所有待审批的挑战。"""
        with self._lock:
            return tuple(
                record.challenge
                for record in self._attempts.values()
                if record.state == "awaiting_approval" and record.challenge is not None
            )

    def find_reusable_grant(
        self,
        request: ToolExecutionRequest,
        access: ToolAccessRequest,
    ) -> ApprovalGrant | None:
        """查找可重用的审批授权。

        从最新的 attempt 开始反向查找，找到第一个满足条件的 grant：
        1. grant 不是 "once" scope（once 只使用一次）
        2. grant 没有被消费过（grant_consumed=False）
        3. grant 没有过期
        4. registration_id 匹配
        5. scope="session" 时 session_id 匹配
        6. actions、resources、effects、risk 都在授权范围内

        参数:
            request: 当前执行请求
            access: 当前访问请求

        返回:
            ApprovalGrant 或 None（未找到可复用的授权）
        """
        wanted_resources = tuple(resource.uri for resource in access.resources)
        with self._lock:
            for record in reversed(tuple(self._attempts.values())):
                grant = record.grant
                if grant is None or grant.scope == "once" or record.grant_consumed or grant.expired():
                    continue
                if record.request.registration_id != request.registration_id:
                    continue
                if grant.scope == "session" and record.request.session_id != request.session_id:
                    continue
                if grant.actions != access.actions:
                    continue
                if tuple(resource.uri for resource in grant.resources) != wanted_resources:
                    continue
                if not access.effects <= grant.effects:
                    continue
                if _RISK_RANK[access.risk] > _RISK_RANK[grant.risk]:
                    continue
                return grant
        return None

    def compare_and_set(
        self,
        attempt_id: str,
        expected_state: ToolAttemptState,
        record: ToolAttemptRecord,
    ) -> None:
        """比较并设置（CAS 操作）。

        只有在当前状态等于 expected_state 时才执行更新。
        这是并发安全的关键机制：防止两个协程同时修改同一个 attempt。

        参数:
            attempt_id: 要更新的 attempt ID
            expected_state: 期望的当前状态
            record: 新的记录

        抛出:
            ToolStateConflictError: 当前状态与预期不符
        """
        with self._lock:
            current = self._attempts.get(attempt_id)
            if current is None or current.state != expected_state:
                actual = current.state if current is not None else "missing"
                raise ToolStateConflictError(
                    f"Tool attempt state conflict: expected {expected_state}, got {actual}"
                )
            if record.attempt_id != attempt_id:
                raise ValueError("replacement attempt_id must match")
            self._attempts[attempt_id] = record
            self._index(record)

    def _index(self, record: ToolAttemptRecord) -> None:
        """更新辅助索引。

        当记录中包含 challenge 或 interaction 时，
        更新 approval_index 和 interaction_index 以支持按 ID 查询。
        """
        if record.challenge is not None:
            self._approval_index[record.challenge.approval_id] = record.attempt_id
        if record.interaction is not None:
            self._interaction_index[record.interaction.interaction_id] = record.attempt_id


# ── 辅助函数 ──────────────────────────────────────────────────────────────────


def transition(record: ToolAttemptRecord, state: ToolAttemptState, **changes) -> ToolAttemptRecord:
    """状态转换 —— 基于当前记录和新状态生成新记录。

    使用 dataclasses.replace() 创建记录的副本并更新指定字段。

    参数:
        record: 当前记录
        state: 新状态
        changes: 其他要更新的字段（如 result=xxx, challenge=xxx）

    返回:
        更新后的新记录
    """
    return replace(record, state=state, **changes)


def attempt_id_for(request: ToolExecutionRequest) -> str:
    """为执行请求生成 attempt 唯一 ID。

    ID 格式: "session_id:run_id:tool_call_id"
    三元组在一个会话内是唯一的。

    参数:
        request: 工具执行请求

    返回:
        格式化的 attempt ID
    """
    return f"{request.session_id}:{request.run_id}:{request.tool_call_id}"


_RISK_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


__all__ = [
    "InMemoryToolStateStore",
    "InteractionRequest",
    "InteractionResponse",
    "ToolAttemptRecord",
    "ToolAttemptState",
    "ToolStateConflictError",
    "ToolStateStore",
    "attempt_id_for",
    "build_interaction_request",
    "transition",
]