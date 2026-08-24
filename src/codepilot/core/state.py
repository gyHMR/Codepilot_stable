"""保存 Core 决策循环的唯一内存状态及其严格物化/评估逻辑。"""

from __future__ import annotations

"""Canonical task state and observable run facts owned by Core."""

from dataclasses import asdict, dataclass, field
from typing import Literal, Mapping, cast

from .errors import CoreContractError
from .plan import PlanState, load_plan_state


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _non_negative_int(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = _optional_text(item)
        if text is not None:
            result.append(text)
    return result


CORE_STATE_SCHEMA_VERSION = 2

TaskStatus = Literal["active", "blocked", "satisfied", "abandoned"]
TaskBlockerKind = Literal[
    "user_input_required",
    "tool_unavailable",
    "verification_failed",
    "plan_incomplete",
    "replan_required",
]
VerificationFactStatus = Literal[
    "none",
    "unknown",
    "passed",
    "failed",
    "stale",
    "unavailable",
]
CoreAssessmentStatus = Literal[
    "active",
    "blocked",
    "needs_verification",
    "needs_replan",
    "ready_to_finish",
    "satisfied",
]


@dataclass(frozen=True)
class TaskBlocker:
    """阻止 Core 宣布完成的结构化原因。"""
    kind: TaskBlockerKind
    reason: str
    evidence_refs: tuple[str, ...] = ()
    recoverable: bool = True

    def __post_init__(self) -> None:
        if self.kind not in {
            "user_input_required",
            "tool_unavailable",
            "verification_failed",
            "plan_incomplete",
            "replan_required",
        }:
            raise ValueError(f"Unknown task blocker: {self.kind}")
        object.__setattr__(
            self, "reason", _required_core_text(self.reason, "blocker reason")
        )
        object.__setattr__(self, "evidence_refs", _text_tuple(self.evidence_refs))
        if not isinstance(self.recoverable, bool):
            raise TypeError("recoverable must be bool")


@dataclass(frozen=True)
class TaskState:
    """任务目标、完成状态和阻塞项的权威快照。"""
    original_request: str
    current_goal: str
    status: TaskStatus = "active"
    plan: PlanState | None = None
    blockers: tuple[TaskBlocker, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "original_request",
            _required_core_text(self.original_request, "original_request"),
        )
        object.__setattr__(
            self,
            "current_goal",
            _required_core_text(self.current_goal, "current_goal"),
        )
        if self.status not in {"active", "blocked", "satisfied", "abandoned"}:
            raise ValueError(f"Unknown task status: {self.status}")
        if self.plan is not None and not isinstance(self.plan, PlanState):
            raise TypeError("plan must be PlanState or None")
        blockers = tuple(self.blockers)
        if any(not isinstance(item, TaskBlocker) for item in blockers):
            raise TypeError("blockers must contain TaskBlocker values")
        object.__setattr__(self, "blockers", blockers)


@dataclass(frozen=True)
class CoreCounters:
    """Core 循环中的模型和工具计数器。"""
    model_turns: int = 0
    model_attempts: int | None = None
    tool_iterations: int = 0
    tool_calls: int = 0
    total_recoveries: int = 0

    def __post_init__(self) -> None:
        if self.model_attempts is None:
            object.__setattr__(self, "model_attempts", self.model_turns)
        for name in (
            "model_turns",
            "model_attempts",
            "tool_iterations",
            "tool_calls",
            "total_recoveries",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class WorkspaceFacts:
    """Core 已知的工作区变更事实。"""
    revision: int = 0
    changed: bool = False
    affected_paths: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.revision, int)
            or isinstance(self.revision, bool)
            or self.revision < 0
        ):
            raise ValueError("workspace revision must be a non-negative integer")
        if not isinstance(self.changed, bool):
            raise TypeError("workspace changed must be bool")
        object.__setattr__(
            self, "affected_paths", tuple(sorted(set(_text_tuple(self.affected_paths))))
        )
        object.__setattr__(self, "evidence_refs", _text_tuple(self.evidence_refs))


@dataclass(frozen=True)
class VerificationFacts:
    """Core 已知的验证状态与摘要。"""
    status: VerificationFactStatus = "none"
    verified_revision: int | None = None
    attempted_checks: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {
            "none",
            "unknown",
            "passed",
            "failed",
            "stale",
            "unavailable",
        }:
            raise ValueError(f"Unknown verification status: {self.status}")
        if self.verified_revision is not None and (
            not isinstance(self.verified_revision, int)
            or isinstance(self.verified_revision, bool)
            or self.verified_revision < 0
        ):
            raise ValueError("verified_revision must be a non-negative integer or None")
        object.__setattr__(self, "attempted_checks", _text_tuple(self.attempted_checks))
        object.__setattr__(self, "evidence_refs", _text_tuple(self.evidence_refs))
        object.__setattr__(
            self, "unavailable_reason", _optional_text(self.unavailable_reason)
        )
        if self.status == "unavailable" and self.unavailable_reason is None:
            raise ValueError("unavailable verification requires a reason")


@dataclass(frozen=True)
class FailureRecord:
    """一次可观测失败及其关联调用。"""
    code: str
    source: str
    message: str
    recoverable: bool
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _required_core_text(self.code, "failure code"))
        object.__setattr__(
            self, "source", _required_core_text(self.source, "failure source")
        )
        object.__setattr__(
            self, "message", _required_core_text(self.message, "failure message")
        )
        if not isinstance(self.recoverable, bool):
            raise TypeError("failure recoverable must be bool")
        object.__setattr__(self, "evidence_refs", _text_tuple(self.evidence_refs))


@dataclass(frozen=True)
class FailureCount:
    """按错误指纹聚合的失败次数。"""
    code: str
    count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "code", _required_core_text(self.code, "failure count code")
        )
        if (
            not isinstance(self.count, int)
            or isinstance(self.count, bool)
            or self.count <= 0
        ):
            raise ValueError("failure count must be positive")


@dataclass(frozen=True)
class FailureFacts:
    """失败记录及重试聚合信息。"""
    latest: FailureRecord | None = None
    counts: tuple[FailureCount, ...] = ()

    def __post_init__(self) -> None:
        if self.latest is not None and not isinstance(self.latest, FailureRecord):
            raise TypeError("latest failure must be FailureRecord or None")
        counts = tuple(self.counts)
        if any(not isinstance(item, FailureCount) for item in counts):
            raise TypeError("failure counts must contain FailureCount values")
        if len({item.code for item in counts}) != len(counts):
            raise ValueError("failure count codes must be unique")
        object.__setattr__(
            self, "counts", tuple(sorted(counts, key=lambda item: item.code))
        )

    def count_for(self, code: str) -> int:
        """返回指定失败代码累计出现的次数。"""
        return next((item.count for item in self.counts if item.code == code), 0)


@dataclass(frozen=True)
class LoopGuardFacts:
    """用于识别重复模型/工具循环的守卫状态。"""
    last_tool_fingerprint: str | None = None
    repeated_tool_calls: int = 0
    repeated_no_progress: int = 0
    seen_tool_call_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "last_tool_fingerprint", _optional_text(self.last_tool_fingerprint)
        )
        for name in ("repeated_tool_calls", "repeated_no_progress"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(
            self, "seen_tool_call_ids", _text_tuple(self.seen_tool_call_ids)
        )


@dataclass(frozen=True)
class ObservationLedger:
    """已消费观察值的序号账本，保证 reducer 幂等。"""
    applied_observation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "applied_observation_ids",
            _text_tuple(self.applied_observation_ids),
        )


@dataclass(frozen=True)
class RunFacts:
    """Run 级别的消息、错误和停止事实。"""
    counters: CoreCounters = field(default_factory=CoreCounters)
    workspace: WorkspaceFacts = field(default_factory=WorkspaceFacts)
    verification: VerificationFacts = field(default_factory=VerificationFacts)
    failures: FailureFacts = field(default_factory=FailureFacts)
    loop_guards: LoopGuardFacts = field(default_factory=LoopGuardFacts)
    observation_ledger: ObservationLedger = field(default_factory=ObservationLedger)


@dataclass(frozen=True)
class CoreAssessment:
    """对当前 CoreState 是否可完成的评估结果。"""
    status: CoreAssessmentStatus
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CoreState:
    """Core 决策循环的唯一状态容器。"""
    task: TaskState
    facts: RunFacts = field(default_factory=RunFacts)
    schema_version: int = CORE_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CORE_STATE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported CoreState schema: {self.schema_version}")
        if not isinstance(self.task, TaskState):
            raise TypeError("task must be TaskState")
        if not isinstance(self.facts, RunFacts):
            raise TypeError("facts must be RunFacts")

    @classmethod
    def new(cls, original_request: str, current_goal: str | None = None) -> "CoreState":
        """从用户原始请求创建初始 CoreState。"""
        request = _required_core_text(original_request, "original_request")
        goal = _optional_text(current_goal) or request
        return cls(task=TaskState(original_request=request, current_goal=goal))

    def to_dict(self) -> dict[str, object]:
        """编码为当前唯一 schema 的持久化映射。"""
        return {
            "schema_version": self.schema_version,
            "task": {
                "original_request": self.task.original_request,
                "current_goal": self.task.current_goal,
                "status": self.task.status,
                "plan": (
                    self.task.plan.to_dict() if self.task.plan is not None else None
                ),
                "blockers": [
                    {
                        "kind": item.kind,
                        "reason": item.reason,
                        "evidence_refs": list(item.evidence_refs),
                        "recoverable": item.recoverable,
                    }
                    for item in self.task.blockers
                ],
            },
            "facts": {
                "counters": asdict(self.facts.counters),
                "workspace": {
                    **asdict(self.facts.workspace),
                    "affected_paths": list(self.facts.workspace.affected_paths),
                    "evidence_refs": list(self.facts.workspace.evidence_refs),
                },
                "verification": {
                    **asdict(self.facts.verification),
                    "attempted_checks": list(self.facts.verification.attempted_checks),
                    "evidence_refs": list(self.facts.verification.evidence_refs),
                },
                "failures": {
                    "latest": (
                        asdict(self.facts.failures.latest)
                        if self.facts.failures.latest
                        else None
                    ),
                    "counts": [asdict(item) for item in self.facts.failures.counts],
                },
                "loop_guards": {
                    **asdict(self.facts.loop_guards),
                    "seen_tool_call_ids": list(
                        self.facts.loop_guards.seen_tool_call_ids
                    ),
                },
                "observation_ledger": {
                    "applied_observation_ids": list(
                        self.facts.observation_ledger.applied_observation_ids
                    )
                },
            },
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "CoreState":
        """从严格映射恢复 CoreState，不执行历史 schema 升级。"""
        if not isinstance(raw, Mapping):
            raise TypeError("CoreState must be an object")
        if raw.get("schema_version") != CORE_STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported CoreState schema")
        task_raw = _mapping(raw.get("task"), "task")
        facts_raw = _mapping(raw.get("facts"), "facts")
        counters_raw = _mapping(facts_raw.get("counters"), "facts.counters")
        workspace_raw = _mapping(facts_raw.get("workspace"), "facts.workspace")
        verification_raw = _mapping(facts_raw.get("verification"), "facts.verification")
        failures_raw = _mapping(facts_raw.get("failures"), "facts.failures")
        loop_raw = _mapping(facts_raw.get("loop_guards"), "facts.loop_guards")
        ledger_raw = _mapping(
            facts_raw.get("observation_ledger"), "facts.observation_ledger"
        )
        blocker_values = _mapping_list(task_raw.get("blockers"), "task.blockers")
        count_values = _mapping_list(
            failures_raw.get("counts"), "facts.failures.counts"
        )
        latest_raw = failures_raw.get("latest")
        latest = (
            _failure_record_from_mapping(_mapping(latest_raw, "facts.failures.latest"))
            if latest_raw is not None
            else None
        )
        return cls(
            task=TaskState(
                original_request=_required_core_text(
                    task_raw.get("original_request"), "original_request"
                ),
                current_goal=_required_core_text(
                    task_raw.get("current_goal"), "current_goal"
                ),
                status=cast(TaskStatus, task_raw.get("status")),
                plan=load_plan_state(task_raw.get("plan")),
                blockers=tuple(
                    TaskBlocker(
                        kind=cast(TaskBlockerKind, item.get("kind")),
                        reason=_required_core_text(
                            item.get("reason"), "blocker reason"
                        ),
                        evidence_refs=tuple(_string_list(item.get("evidence_refs"))),
                        recoverable=bool(item.get("recoverable")),
                    )
                    for item in blocker_values
                ),
            ),
            facts=RunFacts(
                counters=CoreCounters(
                    model_turns=_non_negative_int(counters_raw.get("model_turns")),
                    model_attempts=_non_negative_int(
                        counters_raw.get(
                            "model_attempts",
                            counters_raw.get("model_turns"),
                        )
                    ),
                    tool_iterations=_non_negative_int(
                        counters_raw.get("tool_iterations")
                    ),
                    tool_calls=_non_negative_int(counters_raw.get("tool_calls")),
                    total_recoveries=_non_negative_int(
                        counters_raw.get("total_recoveries")
                    ),
                ),
                workspace=WorkspaceFacts(
                    revision=_non_negative_int(workspace_raw.get("revision")),
                    changed=bool(workspace_raw.get("changed")),
                    affected_paths=tuple(
                        _string_list(workspace_raw.get("affected_paths"))
                    ),
                    evidence_refs=tuple(
                        _string_list(workspace_raw.get("evidence_refs"))
                    ),
                ),
                verification=VerificationFacts(
                    status=cast(VerificationFactStatus, verification_raw.get("status")),
                    verified_revision=_optional_int(
                        verification_raw.get("verified_revision")
                    ),
                    attempted_checks=tuple(
                        _string_list(verification_raw.get("attempted_checks"))
                    ),
                    evidence_refs=tuple(
                        _string_list(verification_raw.get("evidence_refs"))
                    ),
                    unavailable_reason=_optional_text(
                        verification_raw.get("unavailable_reason")
                    ),
                ),
                failures=FailureFacts(
                    latest=latest,
                    counts=tuple(
                        FailureCount(
                            code=_required_core_text(
                                item.get("code"), "failure count code"
                            ),
                            count=_positive_int(item.get("count"), "failure count"),
                        )
                        for item in count_values
                    ),
                ),
                loop_guards=LoopGuardFacts(
                    last_tool_fingerprint=_optional_text(
                        loop_raw.get("last_tool_fingerprint")
                    ),
                    repeated_tool_calls=_non_negative_int(
                        loop_raw.get("repeated_tool_calls")
                    ),
                    repeated_no_progress=_non_negative_int(
                        loop_raw.get("repeated_no_progress")
                    ),
                    seen_tool_call_ids=tuple(
                        _string_list(loop_raw.get("seen_tool_call_ids"))
                    ),
                ),
                observation_ledger=ObservationLedger(
                    applied_observation_ids=tuple(
                        _string_list(ledger_raw.get("applied_observation_ids"))
                    )
                ),
            ),
        )


def load_core_state(
    raw: CoreState | Mapping[str, object],
) -> CoreState:
    """Load only the current authoritative CoreState schema."""

    if isinstance(raw, CoreState):
        return raw
    if not isinstance(raw, Mapping):
        raise CoreContractError("Core state must be an object")
    schema_version = raw.get("schema_version")
    if schema_version != CORE_STATE_SCHEMA_VERSION:
        raise CoreContractError(f"Unsupported CoreState schema: {schema_version}")
    try:
        return CoreState.from_mapping(raw)
    except (TypeError, ValueError) as exc:
        raise CoreContractError(f"Invalid CoreState schema: {exc}") from exc


def assess_core_state(state: CoreState) -> CoreAssessment:
    """评估任务是否完成、阻塞或仍需继续执行。"""
    if state.task.status == "satisfied":
        return CoreAssessment("satisfied")
    if state.task.status in {"blocked", "abandoned"}:
        return CoreAssessment("blocked", (f"task_{state.task.status}",))
    if any(item.kind == "replan_required" for item in state.task.blockers):
        return CoreAssessment("needs_replan", ("replan_required",))
    workspace = state.facts.workspace
    verification = state.facts.verification
    if workspace.changed and not (
        verification.status in {"passed", "unavailable"}
        and verification.verified_revision == workspace.revision
    ):
        return CoreAssessment("needs_verification", ("verification_not_fresh",))
    plan = state.task.plan
    if (
        plan is not None
        and plan.status == "active"
        and plan.close_request is not None
        and all(item.status == "completed" for item in plan.steps)
    ):
        return CoreAssessment("ready_to_finish")
    return CoreAssessment("active")


def _required_core_text(value: object, field_name: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{field_name} is required")
    return text


def _text_tuple(values: object) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list, set, frozenset)):
        raise TypeError("expected a sequence of text values")
    return tuple(
        dict.fromkeys(_required_core_text(value, "text value") for value in values)
    )


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    return value


def _mapping_list(value: object, field_name: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or any(
        not isinstance(item, Mapping) for item in value
    ):
        raise TypeError(f"{field_name} must be a list of objects")
    return list(value)


def _failure_record_from_mapping(raw: Mapping[str, object]) -> FailureRecord:
    return FailureRecord(
        code=_required_core_text(raw.get("code"), "failure code"),
        source=_required_core_text(raw.get("source"), "failure source"),
        message=_required_core_text(raw.get("message"), "failure message"),
        recoverable=bool(raw.get("recoverable")),
        evidence_refs=tuple(_string_list(raw.get("evidence_refs"))),
    )


def _positive_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


__all__ = [
    "CORE_STATE_SCHEMA_VERSION",
    "CoreAssessment",
    "CoreAssessmentStatus",
    "CoreCounters",
    "CoreState",
    "FailureCount",
    "FailureFacts",
    "FailureRecord",
    "LoopGuardFacts",
    "ObservationLedger",
    "RunFacts",
    "TaskBlocker",
    "TaskBlockerKind",
    "TaskState",
    "TaskStatus",
    "VerificationFactStatus",
    "VerificationFacts",
    "WorkspaceFacts",
    "assess_core_state",
    "load_core_state",
]
