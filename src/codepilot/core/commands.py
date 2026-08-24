"""定义 Core 可消费的计划与运行控制命令及其严格序列化契约。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from .plan import (
    MAX_PLAN_ITEMS,
    PlanDefinition,
    PlanRevisionReason,
    PlanStepDefinition,
    PlanStepUpdate,
    ensure_plan_revision_reason,
)


CommandStatus = Literal["applied", "rejected"]


@dataclass(frozen=True)
class SubmitPlan:
    """提交一份新的执行计划。"""
    command_id: str
    definition: PlanDefinition
    steps: tuple[PlanStepDefinition, ...]

    def __post_init__(self) -> None:
        _set_command_id(self)
        if not isinstance(self.definition, PlanDefinition):
            raise TypeError("definition must be PlanDefinition")
        object.__setattr__(self, "steps", _step_definitions(self.steps))


@dataclass(frozen=True)
class UpdatePlanProgress:
    """更新当前计划项状态和验证信息。"""
    command_id: str
    expected_revision: int
    updates: tuple[PlanStepUpdate, ...]

    def __post_init__(self) -> None:
        _set_command_id(self)
        object.__setattr__(
            self,
            "expected_revision",
            _non_negative_int(self.expected_revision, "expected_revision"),
        )
        updates = tuple(self.updates)
        if not updates:
            raise ValueError("updates must contain at least one item")
        if any(not isinstance(item, PlanStepUpdate) for item in updates):
            raise TypeError("updates must contain PlanStepUpdate values")
        if len({item.step_id for item in updates}) != len(updates):
            raise ValueError("updates must contain unique step ids")
        object.__setattr__(self, "updates", updates)


@dataclass(frozen=True)
class ProposePlanRevision:
    """提出对活动计划的结构化修订。"""
    command_id: str
    expected_revision: int
    reason: PlanRevisionReason
    definition: PlanDefinition
    steps: tuple[PlanStepDefinition, ...]

    def __post_init__(self) -> None:
        _set_command_id(self)
        object.__setattr__(
            self,
            "expected_revision",
            _non_negative_int(self.expected_revision, "expected_revision"),
        )
        object.__setattr__(self, "reason", ensure_plan_revision_reason(self.reason))
        if not isinstance(self.definition, PlanDefinition):
            raise TypeError("definition must be PlanDefinition")
        object.__setattr__(self, "steps", _step_definitions(self.steps))


@dataclass(frozen=True)
class RequestPlanClose:
    """请求关闭活动计划。"""
    command_id: str
    expected_revision: int
    summary: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        _set_command_id(self)
        object.__setattr__(
            self,
            "expected_revision",
            _non_negative_int(self.expected_revision, "expected_revision"),
        )
        object.__setattr__(self, "summary", _required_text(self.summary, "summary"))
        refs = _text_tuple(self.evidence_refs, "evidence_refs")
        if not refs:
            raise ValueError("evidence_refs must contain at least one item")
        object.__setattr__(self, "evidence_refs", refs)


@dataclass(frozen=True)
class ApprovePlan:
    """批准待审批计划并允许进入执行阶段。"""
    command_id: str
    expected_revision: int

    def __post_init__(self) -> None:
        _set_command_id(self)
        object.__setattr__(
            self,
            "expected_revision",
            _non_negative_int(self.expected_revision, "expected_revision"),
        )


@dataclass(frozen=True)
class RejectPlan:
    """拒绝待审批计划并保留拒绝原因。"""
    command_id: str
    expected_revision: int
    reason: str = "user_rejected"

    def __post_init__(self) -> None:
        _set_command_id(self)
        object.__setattr__(
            self,
            "expected_revision",
            _non_negative_int(self.expected_revision, "expected_revision"),
        )
        object.__setattr__(self, "reason", _required_text(self.reason, "reason"))


@dataclass(frozen=True)
class ApprovePlanRevision:
    """批准待处理的计划修订。"""
    command_id: str
    expected_revision: int

    def __post_init__(self) -> None:
        _set_command_id(self)
        object.__setattr__(
            self,
            "expected_revision",
            _non_negative_int(self.expected_revision, "expected_revision"),
        )


@dataclass(frozen=True)
class RejectPlanRevision:
    """拒绝待处理的计划修订。"""
    command_id: str
    expected_revision: int
    reason: str = "user_rejected"

    def __post_init__(self) -> None:
        _set_command_id(self)
        object.__setattr__(
            self,
            "expected_revision",
            _non_negative_int(self.expected_revision, "expected_revision"),
        )
        object.__setattr__(self, "reason", _required_text(self.reason, "reason"))


@dataclass(frozen=True)
class AbandonPlan:
    """明确放弃当前活动计划。"""
    command_id: str
    expected_revision: int
    reason: str

    def __post_init__(self) -> None:
        _set_command_id(self)
        object.__setattr__(
            self,
            "expected_revision",
            _non_negative_int(self.expected_revision, "expected_revision"),
        )
        object.__setattr__(self, "reason", _required_text(self.reason, "reason"))


@dataclass(frozen=True)
class ReportVerificationUnavailable:
    """记录当前环境无法完成验证的事实。"""
    command_id: str
    reason: str
    attempted_checks: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _set_command_id(self)
        object.__setattr__(self, "reason", _required_text(self.reason, "reason"))
        object.__setattr__(
            self,
            "attempted_checks",
            _text_tuple(self.attempted_checks, "attempted_checks"),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _text_tuple(self.evidence_refs, "evidence_refs"),
        )


CoreCommand: TypeAlias = (
    SubmitPlan
    | UpdatePlanProgress
    | ProposePlanRevision
    | RequestPlanClose
    | ApprovePlan
    | RejectPlan
    | ApprovePlanRevision
    | RejectPlanRevision
    | AbandonPlan
    | ReportVerificationUnavailable
)

_CORE_COMMAND_TYPES = (
    SubmitPlan,
    UpdatePlanProgress,
    ProposePlanRevision,
    RequestPlanClose,
    ApprovePlan,
    RejectPlan,
    ApprovePlanRevision,
    RejectPlanRevision,
    AbandonPlan,
    ReportVerificationUnavailable,
)


@dataclass(frozen=True)
class CommandResult:
    """Core 命令处理后的统一结果。"""
    command_id: str
    status: CommandStatus
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "command_id", _required_text(self.command_id, "command_id")
        )
        if self.status not in {"applied", "rejected"}:
            raise ValueError(f"Unknown command status: {self.status}")
        object.__setattr__(self, "reason", str(self.reason).strip())


def is_core_command(value: object) -> bool:
    """判断对象是否为 Core 支持的命令类型。"""
    return isinstance(value, _CORE_COMMAND_TYPES)


def core_command_to_dict(command: CoreCommand) -> dict[str, object]:
    """将 Core 命令编码为严格可持久化字典。"""
    if isinstance(command, SubmitPlan):
        return {
            "kind": "submit_plan",
            "command_id": command.command_id,
            "definition": command.definition.to_dict(),
            "steps": [item.to_dict() for item in command.steps],
        }
    if isinstance(command, UpdatePlanProgress):
        return {
            "kind": "update_plan_progress",
            "command_id": command.command_id,
            "expected_revision": command.expected_revision,
            "updates": [item.to_dict() for item in command.updates],
        }
    if isinstance(command, ProposePlanRevision):
        return {
            "kind": "propose_plan_revision",
            "command_id": command.command_id,
            "expected_revision": command.expected_revision,
            "reason": command.reason,
            "definition": command.definition.to_dict(),
            "steps": [item.to_dict() for item in command.steps],
        }
    if isinstance(command, RequestPlanClose):
        return {
            "kind": "request_plan_close",
            "command_id": command.command_id,
            "expected_revision": command.expected_revision,
            "summary": command.summary,
            "evidence_refs": list(command.evidence_refs),
        }
    if isinstance(command, (ApprovePlan, ApprovePlanRevision)):
        return {
            "kind": (
                "approve_plan"
                if isinstance(command, ApprovePlan)
                else "approve_plan_revision"
            ),
            "command_id": command.command_id,
            "expected_revision": command.expected_revision,
        }
    if isinstance(command, (RejectPlan, RejectPlanRevision, AbandonPlan)):
        kind = {
            RejectPlan: "reject_plan",
            RejectPlanRevision: "reject_plan_revision",
            AbandonPlan: "abandon_plan",
        }[type(command)]
        return {
            "kind": kind,
            "command_id": command.command_id,
            "expected_revision": command.expected_revision,
            "reason": command.reason,
        }
    if isinstance(command, ReportVerificationUnavailable):
        return {
            "kind": "report_verification_unavailable",
            "command_id": command.command_id,
            "reason": command.reason,
            "attempted_checks": list(command.attempted_checks),
            "evidence_refs": list(command.evidence_refs),
        }
    raise TypeError(f"Unknown Core command: {type(command).__name__}")


def core_command_from_mapping(raw: Mapping[str, object]) -> CoreCommand:
    """从严格字段映射解码 Core 命令。"""
    kind = _required_text(raw.get("kind"), "kind")
    command_id = _required_text(raw.get("command_id"), "command_id")
    if kind == "submit_plan":
        return SubmitPlan(
            command_id,
            PlanDefinition.from_mapping(_mapping(raw.get("definition"), "definition")),
            _step_definitions_from_raw(raw.get("steps")),
        )
    if kind == "update_plan_progress":
        return UpdatePlanProgress(
            command_id,
            _non_negative_int(raw.get("expected_revision"), "expected_revision"),
            tuple(
                PlanStepUpdate.from_mapping(item)
                for item in _mapping_list(raw.get("updates"), "updates")
            ),
        )
    if kind == "propose_plan_revision":
        return ProposePlanRevision(
            command_id,
            _non_negative_int(raw.get("expected_revision"), "expected_revision"),
            ensure_plan_revision_reason(raw.get("reason")),
            PlanDefinition.from_mapping(_mapping(raw.get("definition"), "definition")),
            _step_definitions_from_raw(raw.get("steps")),
        )
    if kind == "request_plan_close":
        return RequestPlanClose(
            command_id,
            _non_negative_int(raw.get("expected_revision"), "expected_revision"),
            _required_text(raw.get("summary"), "summary"),
            _string_list(raw.get("evidence_refs"), "evidence_refs"),
        )
    if kind == "approve_plan":
        return ApprovePlan(command_id, _expected_revision(raw))
    if kind == "reject_plan":
        return RejectPlan(
            command_id,
            _expected_revision(raw),
            _required_text(raw.get("reason", "user_rejected"), "reason"),
        )
    if kind == "approve_plan_revision":
        return ApprovePlanRevision(command_id, _expected_revision(raw))
    if kind == "reject_plan_revision":
        return RejectPlanRevision(
            command_id,
            _expected_revision(raw),
            _required_text(raw.get("reason", "user_rejected"), "reason"),
        )
    if kind == "abandon_plan":
        return AbandonPlan(
            command_id,
            _expected_revision(raw),
            _required_text(raw.get("reason"), "reason"),
        )
    if kind == "report_verification_unavailable":
        return ReportVerificationUnavailable(
            command_id,
            _required_text(raw.get("reason"), "reason"),
            _string_list(raw.get("attempted_checks", ()), "attempted_checks"),
            _string_list(raw.get("evidence_refs", ()), "evidence_refs"),
        )
    raise ValueError(f"Unknown Core command kind: {kind}")


def _set_command_id(command: object) -> None:
    object.__setattr__(
        command,
        "command_id",
        _required_text(getattr(command, "command_id"), "command_id"),
    )


def _step_definitions(
    values: tuple[PlanStepDefinition, ...],
) -> tuple[PlanStepDefinition, ...]:
    steps = tuple(values)
    if not 1 <= len(steps) <= MAX_PLAN_ITEMS:
        raise ValueError(f"steps must contain between 1 and {MAX_PLAN_ITEMS} items")
    if any(not isinstance(item, PlanStepDefinition) for item in steps):
        raise TypeError("steps must contain PlanStepDefinition values")
    return steps


def _step_definitions_from_raw(value: object) -> tuple[PlanStepDefinition, ...]:
    return tuple(
        PlanStepDefinition.from_mapping(item)
        for item in _mapping_list(value, "steps")
    )


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    return value


def _mapping_list(
    value: object,
    field_name: str,
) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list")
    if any(not isinstance(item, Mapping) for item in value):
        raise TypeError(f"{field_name} must contain objects")
    return tuple(cast(Mapping[str, object], item) for item in value)


def _expected_revision(raw: Mapping[str, object]) -> int:
    return _non_negative_int(raw.get("expected_revision"), "expected_revision")


def _string_list(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list")
    return tuple(str(item) for item in value)


def _text_tuple(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            _required_text(value, f"{field_name} item") for value in values
        )
    )


def _required_text(value: object, field_name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _non_negative_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


__all__ = [
    "AbandonPlan",
    "ApprovePlan",
    "ApprovePlanRevision",
    "CommandResult",
    "CommandStatus",
    "CoreCommand",
    "ProposePlanRevision",
    "RejectPlan",
    "RejectPlanRevision",
    "ReportVerificationUnavailable",
    "RequestPlanClose",
    "SubmitPlan",
    "UpdatePlanProgress",
    "core_command_from_mapping",
    "core_command_to_dict",
    "is_core_command",
]
