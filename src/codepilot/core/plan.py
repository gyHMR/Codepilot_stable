"""定义计划状态、计划项和计划变更的权威 Core 数据模型。"""

from __future__ import annotations

"""Core-owned Task Plan values.

Plan values are immutable data.  Every state transition is performed by the
Core Reducer from a command; this module intentionally exposes no transition
methods and performs no persistence.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from codepilot.protocols import PlanSummary


RunMode = Literal["read", "plan", "build"]
PlanningBudgetProfile = Literal["conservative", "balanced", "wide"]
PlanOrigin = Literal["plan_mode", "build_mode"]
PlanStatus = Literal["proposed", "active", "completed", "rejected", "abandoned"]
PlanStepStatus = Literal["pending", "in_progress", "completed"]
PlanRevisionReason = Literal[
    "user_request",
    "repeated_execution_failure",
    "new_evidence",
]

PlanOperation = Literal[
    "propose_plan",
    "create_build_plan",
    "update_plan_progress",
    "close_plan",
]

PLAN_STATE_SCHEMA_VERSION = 7
MAX_PLAN_ITEMS = 20
QUALIFIED_FAILURES_FOR_REVISION = 5

_RUN_MODES = frozenset({"read", "plan", "build"})
_BUDGET_PROFILES = frozenset({"conservative", "balanced", "wide"})
_PLAN_ORIGINS = frozenset({"plan_mode", "build_mode"})
_PLAN_STATUSES = frozenset(
    {"proposed", "active", "completed", "rejected", "abandoned"}
)
_STEP_STATUSES = frozenset({"pending", "in_progress", "completed"})
_REVISION_REASONS = frozenset(
    {"user_request", "repeated_execution_failure", "new_evidence"}
)
_PLAN_OPERATIONS = frozenset(
    {"propose_plan", "create_build_plan", "update_plan_progress", "close_plan"}
)


class PlanValidationError(ValueError):
    """计划结构或状态违反当前契约时抛出的领域错误。"""
    pass


@dataclass(frozen=True)
class PlanDefinition:
    """计划创建时的目标、范围、风险和完成标准定义。"""
    summary: str
    completion_criteria: tuple[str, ...]
    task_understanding: str = ""
    current_implementation: str = ""
    target_design: str = ""
    impact_scope: str = ""
    risks_and_open_questions: tuple[str, ...] = ()
    verification_plan: str = ""
    explanation: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", _required_text(self.summary, "summary"))
        criteria = _text_tuple(
            self.completion_criteria,
            "completion_criteria",
            min_items=1,
            max_items=5,
        )
        object.__setattr__(self, "completion_criteria", criteria)
        for name in (
            "task_understanding",
            "current_implementation",
            "target_design",
            "impact_scope",
            "verification_plan",
            "explanation",
        ):
            object.__setattr__(self, name, _optional_text(getattr(self, name)) or "")
        object.__setattr__(
            self,
            "risks_and_open_questions",
            _text_tuple(
                self.risks_and_open_questions,
                "risks_and_open_questions",
                min_items=0,
                max_items=8,
            ),
        )

    def require_plan_mode_details(self) -> None:
        for name in (
            "task_understanding",
            "current_implementation",
            "target_design",
            "impact_scope",
            "verification_plan",
        ):
            if not getattr(self, name):
                raise PlanValidationError(f"{name} is required in plan mode")
        if not self.risks_and_open_questions:
            raise PlanValidationError(
                "risks_and_open_questions must contain at least 1 item in plan mode"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "completion_criteria": list(self.completion_criteria),
            "task_understanding": self.task_understanding,
            "current_implementation": self.current_implementation,
            "target_design": self.target_design,
            "impact_scope": self.impact_scope,
            "risks_and_open_questions": list(self.risks_and_open_questions),
            "verification_plan": self.verification_plan,
            "explanation": self.explanation,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "PlanDefinition":
        allowed = {
            "summary",
            "completion_criteria",
            "task_understanding",
            "current_implementation",
            "target_design",
            "impact_scope",
            "risks_and_open_questions",
            "verification_plan",
            "explanation",
        }
        _reject_unknown(raw, allowed, "plan definition")
        return cls(
            summary=_required_text(raw.get("summary"), "summary"),
            completion_criteria=_sequence(raw.get("completion_criteria"), "completion_criteria"),
            task_understanding=_optional_text(raw.get("task_understanding")) or "",
            current_implementation=_optional_text(raw.get("current_implementation")) or "",
            target_design=_optional_text(raw.get("target_design")) or "",
            impact_scope=_optional_text(raw.get("impact_scope")) or "",
            risks_and_open_questions=_sequence(
                raw.get("risks_and_open_questions", ()),
                "risks_and_open_questions",
            ),
            verification_plan=_optional_text(raw.get("verification_plan")) or "",
            explanation=_optional_text(raw.get("explanation")) or "",
        )


@dataclass(frozen=True)
class PlanStepDefinition:
    """计划项的声明式定义。"""
    step: str
    details: str
    verification: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "step", _required_text(self.step, "step"))
        object.__setattr__(self, "details", _required_text(self.details, "details"))
        object.__setattr__(
            self,
            "verification",
            _required_text(self.verification, "verification"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "step": self.step,
            "details": self.details,
            "verification": self.verification,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "PlanStepDefinition":
        _reject_unknown(raw, {"step", "details", "verification"}, "plan step")
        return cls(
            step=_required_text(raw.get("step"), "step"),
            details=_required_text(raw.get("details"), "details"),
            verification=_required_text(raw.get("verification"), "verification"),
        )


@dataclass(frozen=True)
class PlanStep:
    """带运行时状态的计划项。"""
    step_id: str
    step: str
    details: str
    verification: str
    status: PlanStepStatus = "pending"
    completion_note: str = ""
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _required_text(self.step_id, "step_id"))
        object.__setattr__(self, "step", _required_text(self.step, "step"))
        object.__setattr__(self, "details", _required_text(self.details, "details"))
        object.__setattr__(
            self,
            "verification",
            _required_text(self.verification, "verification"),
        )
        object.__setattr__(self, "status", ensure_plan_step_status(self.status))
        note = _optional_text(self.completion_note) or ""
        refs = _text_tuple(
            self.evidence_refs,
            "evidence_refs",
            min_items=0,
            max_items=64,
        )
        if self.status == "completed" and not note:
            raise PlanValidationError("completed plan step requires completion_note")
        if self.status != "completed" and (note or refs):
            raise PlanValidationError(
                "only completed plan steps can contain completion evidence"
            )
        object.__setattr__(self, "completion_note", note)
        object.__setattr__(self, "evidence_refs", refs)

    @property
    def id(self) -> str:
        return self.step_id

    def definition(self) -> PlanStepDefinition:
        return PlanStepDefinition(self.step, self.details, self.verification)

    def to_dict(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "step": self.step,
            "details": self.details,
            "verification": self.verification,
            "status": self.status,
            "completion_note": self.completion_note,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "PlanStep":
        _reject_unknown(
            raw,
            {
                "step_id",
                "step",
                "details",
                "verification",
                "status",
                "completion_note",
                "evidence_refs",
            },
            "plan step",
        )
        return cls(
            step_id=_required_text(raw.get("step_id"), "step_id"),
            step=_required_text(raw.get("step"), "step"),
            details=_required_text(raw.get("details"), "details"),
            verification=_required_text(raw.get("verification"), "verification"),
            status=ensure_plan_step_status(raw.get("status")),
            completion_note=_optional_text(raw.get("completion_note")) or "",
            evidence_refs=_sequence(raw.get("evidence_refs", ()), "evidence_refs"),
        )


@dataclass(frozen=True)
class PlanStepUpdate:
    """对现有计划项执行的一次状态更新。"""
    step_id: str
    status: PlanStepStatus
    completion_note: str = ""
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _required_text(self.step_id, "step_id"))
        object.__setattr__(self, "status", ensure_plan_step_status(self.status))
        object.__setattr__(
            self,
            "completion_note",
            _optional_text(self.completion_note) or "",
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _text_tuple(
                self.evidence_refs,
                "evidence_refs",
                min_items=0,
                max_items=64,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "status": self.status,
            "completion_note": self.completion_note,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "PlanStepUpdate":
        _reject_unknown(
            raw,
            {"step_id", "status", "completion_note", "evidence_refs"},
            "plan step update",
        )
        return cls(
            step_id=_required_text(raw.get("step_id"), "step_id"),
            status=ensure_plan_step_status(raw.get("status")),
            completion_note=_optional_text(raw.get("completion_note")) or "",
            evidence_refs=_sequence(raw.get("evidence_refs", ()), "evidence_refs"),
        )


@dataclass(frozen=True)
class PendingPlanRevision:
    """等待用户决定的计划修订请求。"""
    reason: PlanRevisionReason
    definition: PlanDefinition
    steps: tuple[PlanStep, ...]
    proposed_at_revision: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", ensure_plan_revision_reason(self.reason))
        if not isinstance(self.definition, PlanDefinition):
            raise TypeError("definition must be PlanDefinition")
        steps = _validate_steps(self.steps)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(
            self,
            "proposed_at_revision",
            _non_negative_int(self.proposed_at_revision, "proposed_at_revision"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "reason": self.reason,
            "definition": self.definition.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "proposed_at_revision": self.proposed_at_revision,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "PendingPlanRevision":
        _reject_unknown(
            raw,
            {"reason", "definition", "steps", "proposed_at_revision"},
            "pending plan revision",
        )
        return cls(
            reason=ensure_plan_revision_reason(raw.get("reason")),
            definition=PlanDefinition.from_mapping(
                _mapping(raw.get("definition"), "definition")
            ),
            steps=tuple(
                PlanStep.from_mapping(item)
                for item in _mapping_sequence(raw.get("steps"), "steps")
            ),
            proposed_at_revision=_non_negative_int(
                raw.get("proposed_at_revision"), "proposed_at_revision"
            ),
        )


@dataclass(frozen=True)
class PlanCloseRequest:
    """关闭活动计划所需的请求与验证信息。"""
    summary: str
    evidence_refs: tuple[str, ...]
    requested_at_revision: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", _required_text(self.summary, "summary"))
        object.__setattr__(
            self,
            "evidence_refs",
            _text_tuple(
                self.evidence_refs,
                "evidence_refs",
                min_items=1,
                max_items=64,
            ),
        )
        object.__setattr__(
            self,
            "requested_at_revision",
            _non_negative_int(self.requested_at_revision, "requested_at_revision"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "evidence_refs": list(self.evidence_refs),
            "requested_at_revision": self.requested_at_revision,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "PlanCloseRequest":
        _reject_unknown(
            raw,
            {"summary", "evidence_refs", "requested_at_revision"},
            "plan close request",
        )
        return cls(
            summary=_required_text(raw.get("summary"), "summary"),
            evidence_refs=_sequence(raw.get("evidence_refs"), "evidence_refs"),
            requested_at_revision=_non_negative_int(
                raw.get("requested_at_revision"), "requested_at_revision"
            ),
        )


@dataclass(frozen=True)
class PlanState:
    """Core 唯一拥有的完整计划状态。"""
    plan_id: str
    origin: PlanOrigin
    status: PlanStatus
    revision: int
    definition: PlanDefinition
    steps: tuple[PlanStep, ...]
    pending_revision: PendingPlanRevision | None = None
    close_request: PlanCloseRequest | None = None
    schema_version: int = field(default=PLAN_STATE_SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        if self.schema_version != PLAN_STATE_SCHEMA_VERSION:
            raise PlanValidationError("unsupported plan state schema")
        object.__setattr__(self, "plan_id", _required_text(self.plan_id, "plan_id"))
        object.__setattr__(self, "origin", ensure_plan_origin(self.origin))
        object.__setattr__(self, "status", ensure_plan_status(self.status))
        object.__setattr__(self, "revision", _non_negative_int(self.revision, "revision"))
        if not isinstance(self.definition, PlanDefinition):
            raise TypeError("definition must be PlanDefinition")
        object.__setattr__(self, "steps", _validate_steps(self.steps))
        if self.pending_revision is not None and not isinstance(
            self.pending_revision, PendingPlanRevision
        ):
            raise TypeError("pending_revision must be PendingPlanRevision or None")
        if self.close_request is not None and not isinstance(
            self.close_request, PlanCloseRequest
        ):
            raise TypeError("close_request must be PlanCloseRequest or None")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "origin": self.origin,
            "status": self.status,
            "revision": self.revision,
            "definition": self.definition.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "pending_revision": (
                self.pending_revision.to_dict()
                if self.pending_revision is not None
                else None
            ),
            "close_request": (
                self.close_request.to_dict() if self.close_request is not None else None
            ),
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "PlanState":
        allowed = {
            "schema_version",
            "plan_id",
            "origin",
            "status",
            "revision",
            "definition",
            "steps",
            "pending_revision",
            "close_request",
        }
        _reject_unknown(raw, allowed, "plan state")
        if raw.get("schema_version") != PLAN_STATE_SCHEMA_VERSION:
            raise PlanValidationError("unsupported plan state schema")
        pending_raw = raw.get("pending_revision")
        close_raw = raw.get("close_request")
        return cls(
            schema_version=PLAN_STATE_SCHEMA_VERSION,
            plan_id=_required_text(raw.get("plan_id"), "plan_id"),
            origin=ensure_plan_origin(raw.get("origin")),
            status=ensure_plan_status(raw.get("status")),
            revision=_non_negative_int(raw.get("revision"), "revision"),
            definition=PlanDefinition.from_mapping(
                _mapping(raw.get("definition"), "definition")
            ),
            steps=tuple(
                PlanStep.from_mapping(item)
                for item in _mapping_sequence(raw.get("steps"), "steps")
            ),
            pending_revision=(
                PendingPlanRevision.from_mapping(
                    _mapping(pending_raw, "pending_revision")
                )
                if pending_raw is not None
                else None
            ),
            close_request=(
                PlanCloseRequest.from_mapping(_mapping(close_raw, "close_request"))
                if close_raw is not None
                else None
            ),
        )

    def to_summary(self) -> PlanSummary:
        """Project canonical plan state into the public runtime result contract."""

        return PlanSummary(
            schema_version=self.schema_version,
            plan_id=self.plan_id,
            owner_run_id=self.plan_id,
            status=self.status,
            origin_mode="plan" if self.origin == "plan_mode" else "build",
            raw_user_request="",
            interpreted_goal="",
            summary=self.definition.summary,
            task_understanding=self.definition.task_understanding,
            current_implementation=self.definition.current_implementation,
            target_design=self.definition.target_design,
            impact_scope=self.definition.impact_scope,
            risks_and_open_questions=list(self.definition.risks_and_open_questions),
            verification_plan=self.definition.verification_plan,
            completion_criteria=list(self.definition.completion_criteria),
            items=[
                {
                    "id": item.step_id,
                    "step": item.step,
                    "details": item.details,
                    "verification": item.verification,
                    "status": item.status,
                }
                for item in self.steps
            ],
            revision=self.revision,
            explanation=self.definition.explanation,
        )


def load_plan_state(raw: object) -> PlanState | None:
    """从持久化对象加载当前计划状态。"""
    if raw is None:
        return None
    if isinstance(raw, PlanState):
        return raw
    if not isinstance(raw, Mapping):
        raise PlanValidationError("plan state must be an object")
    if raw.get("schema_version") == PLAN_STATE_SCHEMA_VERSION:
        return PlanState.from_mapping(raw)
    raise PlanValidationError("unsupported plan state schema")


def plan_state_to_dict(
    state: PlanState | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """将计划状态编码为当前唯一 schema 的字典。"""
    loaded = load_plan_state(state)
    return loaded.to_dict() if loaded is not None else None


def ensure_run_mode(value: object) -> RunMode:
    """校验并收窄 Core 运行模式。"""
    text = str(value).strip() if value is not None else ""
    if text not in _RUN_MODES:
        raise ValueError(f"Unknown run mode: {value}")
    return cast(RunMode, text)


def ensure_planning_budget_profile(value: object) -> PlanningBudgetProfile:
    """校验并收窄计划预算配置。"""
    text = str(value).strip() if value is not None else ""
    if text not in _BUDGET_PROFILES:
        raise ValueError(f"Unknown planning budget profile: {value}")
    return cast(PlanningBudgetProfile, text)


def ensure_plan_origin(value: object) -> PlanOrigin:
    """校验并收窄计划来源。"""
    text = str(value).strip() if value is not None else ""
    if text not in _PLAN_ORIGINS:
        raise PlanValidationError(f"Unknown plan origin: {value}")
    return cast(PlanOrigin, text)


def ensure_plan_status(value: object) -> PlanStatus:
    """校验并收窄计划状态。"""
    text = str(value).strip() if value is not None else ""
    if text not in _PLAN_STATUSES:
        raise PlanValidationError(f"Unknown plan status: {value}")
    return cast(PlanStatus, text)


def ensure_plan_step_status(value: object) -> PlanStepStatus:
    """校验并收窄计划项状态。"""
    text = str(value).strip() if value is not None else ""
    if text not in _STEP_STATUSES:
        raise PlanValidationError(f"Unknown plan step status: {value}")
    return cast(PlanStepStatus, text)


def ensure_plan_revision_reason(value: object) -> PlanRevisionReason:
    """校验并收窄计划修订原因。"""
    text = str(value).strip() if value is not None else ""
    if text not in _REVISION_REASONS:
        raise PlanValidationError(f"Unknown plan revision reason: {value}")
    return cast(PlanRevisionReason, text)


def ensure_plan_operation(value: object) -> PlanOperation:
    """校验并收窄计划操作类型。"""
    text = str(value).strip() if value is not None else ""
    if text not in _PLAN_OPERATIONS:
        raise PlanValidationError(f"Unknown plan operation: {value}")
    return cast(PlanOperation, text)


def _validate_steps(values: Sequence[PlanStep]) -> tuple[PlanStep, ...]:
    steps = tuple(values)
    if not steps or len(steps) > MAX_PLAN_ITEMS:
        raise PlanValidationError(
            f"plan must contain between 1 and {MAX_PLAN_ITEMS} steps"
        )
    if any(not isinstance(step, PlanStep) for step in steps):
        raise TypeError("steps must contain PlanStep values")
    if len({step.step_id for step in steps}) != len(steps):
        raise PlanValidationError("plan step ids must be unique")
    if sum(step.status == "in_progress" for step in steps) > 1:
        raise PlanValidationError("plan can contain at most one in_progress step")
    return steps


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PlanValidationError(f"{field_name} must be an object")
    return value


def _mapping_sequence(
    value: object,
    field_name: str,
) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, (list, tuple)):
        raise PlanValidationError(f"{field_name} must be a list")
    if any(not isinstance(item, Mapping) for item in value):
        raise PlanValidationError(f"{field_name} must contain objects")
    return tuple(cast(Mapping[str, object], item) for item in value)


def _sequence(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise PlanValidationError(f"{field_name} must be a list")
    return tuple(str(item) for item in value)


def _text_tuple(
    values: Sequence[object],
    field_name: str,
    *,
    min_items: int,
    max_items: int,
) -> tuple[str, ...]:
    items = tuple(
        dict.fromkeys(
            _required_text(value, f"{field_name}[{index}]")
            for index, value in enumerate(values)
        )
    )
    if not min_items <= len(items) <= max_items:
        raise PlanValidationError(
            f"{field_name} must contain between {min_items} and {max_items} items"
        )
    return items


def _reject_unknown(
    raw: Mapping[str, object],
    allowed: set[str],
    field_name: str,
) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise PlanValidationError(
            f"unknown {field_name} fields: " + ", ".join(unknown)
        )


def _required_text(value: object, field_name: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise PlanValidationError(f"{field_name} is required")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    return text or None


def _non_negative_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PlanValidationError(f"{field_name} must be a non-negative integer")
    return value


__all__ = [
    "MAX_PLAN_ITEMS",
    "PLAN_STATE_SCHEMA_VERSION",
    "QUALIFIED_FAILURES_FOR_REVISION",
    "PendingPlanRevision",
    "PlanCloseRequest",
    "PlanDefinition",
    "PlanOperation",
    "PlanOrigin",
    "PlanRevisionReason",
    "PlanState",
    "PlanStatus",
    "PlanStep",
    "PlanStepDefinition",
    "PlanStepStatus",
    "PlanStepUpdate",
    "PlanValidationError",
    "PlanningBudgetProfile",
    "RunMode",
    "ensure_plan_operation",
    "ensure_plan_origin",
    "ensure_plan_revision_reason",
    "ensure_plan_status",
    "ensure_plan_step_status",
    "ensure_planning_budget_profile",
    "ensure_run_mode",
    "load_plan_state",
    "plan_state_to_dict",
]
