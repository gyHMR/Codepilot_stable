from __future__ import annotations

"""Canonical Task Plan state and semantic snapshot update protocol."""

from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, cast
from uuid import uuid4

from codepilot.protocols import PLAN_ITEM_LIMIT, PlanSummary


RunMode = Literal["read", "plan", "build"]
PlanningBudgetProfile = Literal["conservative", "balanced", "wide"]
PlanStatus = Literal["proposed", "active", "completed", "rejected", "abandoned"]
PlanItemStatus = Literal["pending", "in_progress", "completed"]
PlanChangeReason = Literal["user_request", "repeated_execution_failure"]
PlanOperation = Literal[
    "propose_plan",
    "create_build_plan",
    "update_plan_progress",
    "close_plan",
]

PLAN_STATE_SCHEMA_VERSION = 6
MAX_PLAN_ITEMS = PLAN_ITEM_LIMIT
QUALIFIED_FAILURES_FOR_REVISION = 5

_RUN_MODES = frozenset({"read", "plan", "build"})
_PLANNING_BUDGET_PROFILES = frozenset({"conservative", "balanced", "wide"})
_PLAN_STATUSES = frozenset({"proposed", "active", "completed", "rejected", "abandoned"})
_ITEM_STATUSES = frozenset({"pending", "in_progress", "completed"})
_CHANGE_REASONS = frozenset({"user_request", "repeated_execution_failure"})
_PLAN_OPERATIONS = frozenset(
    {"propose_plan", "create_build_plan", "update_plan_progress", "close_plan"}
)
_PLAN_KEYS = frozenset(
    {
        "schema_version",
        "plan_id",
        "owner_run_id",
        "status",
        "origin_mode",
        "raw_user_request",
        "interpreted_goal",
        "task_understanding",
        "current_implementation",
        "target_design",
        "impact_scope",
        "risks_and_open_questions",
        "verification_plan",
        "summary",
        "completion_criteria",
        "items",
        "revision",
        "explanation",
        "created_at",
        "updated_at",
        "completed_at",
        "completion_source",
    }
)
_ITEM_KEYS = frozenset({"id", "step", "details", "verification", "status"})
_SNAPSHOT_KEYS = frozenset(
    {
        "raw_user_request",
        "interpreted_goal",
        "task_understanding",
        "current_implementation",
        "target_design",
        "impact_scope",
        "risks_and_open_questions",
        "verification_plan",
        "summary",
        "completion_criteria",
        "items",
        "status",
        "change_reason",
        "explanation",
    }
)
_SNAPSHOT_ITEM_KEYS = frozenset({"id", "step", "details", "verification", "status"})


class PlanValidationError(ValueError):
    """Raised when a plan snapshot violates the canonical plan protocol."""


@dataclass(frozen=True)
class PlanItem:
    id: str
    step: str
    details: str
    verification: str
    status: PlanItemStatus = "pending"

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_text(self.id, "plan item id"))
        object.__setattr__(self, "step", _required_text(self.step, "plan item step"))
        object.__setattr__(self, "details", _required_text(self.details, "plan item details"))
        object.__setattr__(self, "verification", _required_text(self.verification, "plan item verification"))
        object.__setattr__(self, "status", ensure_plan_item_status(self.status))

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "step": self.step,
            "details": self.details,
            "verification": self.verification,
            "status": self.status,
        }


@dataclass(frozen=True)
class PlanSnapshotItem:
    id: str | None
    step: str
    details: str
    verification: str
    status: PlanItemStatus

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _optional_text(self.id))
        object.__setattr__(self, "step", _required_text(self.step, "snapshot item step"))
        object.__setattr__(self, "details", _required_text(self.details, "snapshot item details"))
        object.__setattr__(self, "verification", _required_text(self.verification, "snapshot item verification"))
        object.__setattr__(self, "status", ensure_plan_item_status(self.status))


@dataclass(frozen=True)
class PlanSnapshot:
    raw_user_request: str | None
    interpreted_goal: str | None
    summary: str
    completion_criteria: tuple[str, ...]
    items: tuple[PlanSnapshotItem, ...]
    task_understanding: str | None = None
    current_implementation: str | None = None
    target_design: str | None = None
    impact_scope: str | None = None
    risks_and_open_questions: tuple[str, ...] = field(default_factory=tuple)
    verification_plan: str | None = None
    status: Literal["active", "completed"] | None = None
    change_reason: PlanChangeReason | None = None
    explanation: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_user_request", _optional_text(self.raw_user_request))
        object.__setattr__(self, "interpreted_goal", _optional_text(self.interpreted_goal))
        object.__setattr__(self, "task_understanding", _optional_text(self.task_understanding))
        object.__setattr__(self, "current_implementation", _optional_text(self.current_implementation))
        object.__setattr__(self, "target_design", _optional_text(self.target_design))
        object.__setattr__(self, "impact_scope", _optional_text(self.impact_scope))
        object.__setattr__(
            self,
            "risks_and_open_questions",
            _normalize_text_tuple(self.risks_and_open_questions, "risks_and_open_questions", max_items=8),
        )
        object.__setattr__(self, "verification_plan", _optional_text(self.verification_plan))
        object.__setattr__(self, "summary", _required_text(self.summary, "plan summary"))
        criteria = tuple(_required_text(value, "completion criterion") for value in self.completion_criteria)
        if not criteria or len(criteria) > 5:
            raise PlanValidationError("completion_criteria must contain between 1 and 5 items")
        object.__setattr__(self, "completion_criteria", criteria)
        items = tuple(self.items)
        if not items or len(items) > MAX_PLAN_ITEMS:
            raise PlanValidationError(f"plan must contain between 1 and {MAX_PLAN_ITEMS} items")
        if sum(item.status == "in_progress" for item in items) > 1:
            raise PlanValidationError("plan can contain at most one in_progress item")
        ids = [item.id for item in items if item.id is not None]
        if len(ids) != len(set(ids)):
            raise PlanValidationError("snapshot item ids must be unique")
        object.__setattr__(self, "items", items)
        if self.status not in {None, "active", "completed"}:
            raise PlanValidationError("snapshot status must be active or completed")
        object.__setattr__(self, "change_reason", ensure_plan_change_reason(self.change_reason))
        object.__setattr__(self, "explanation", _optional_text(self.explanation) or "")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PlanSnapshot":
        unknown = sorted(set(raw) - _SNAPSHOT_KEYS)
        if unknown:
            raise PlanValidationError("unknown plan snapshot fields: " + ", ".join(unknown))
        missing = sorted({"summary", "completion_criteria", "items"} - set(raw))
        if missing:
            raise PlanValidationError("missing plan snapshot fields: " + ", ".join(missing))
        criteria = raw.get("completion_criteria")
        if not isinstance(criteria, (list, tuple)):
            raise PlanValidationError("completion_criteria must be a list")
        risks = raw.get("risks_and_open_questions", [])
        if not isinstance(risks, (list, tuple)):
            raise PlanValidationError("risks_and_open_questions must be a list")
        items_raw = raw.get("items")
        if not isinstance(items_raw, (list, tuple)):
            raise PlanValidationError("items must be a list")
        items: list[PlanSnapshotItem] = []
        for index, item in enumerate(items_raw):
            if not isinstance(item, Mapping):
                raise PlanValidationError(f"items[{index}] must be an object")
            unknown_item = sorted(set(item) - _SNAPSHOT_ITEM_KEYS)
            if unknown_item:
                raise PlanValidationError(f"items[{index}] has unknown fields: " + ", ".join(unknown_item))
            missing_item = sorted({"step", "details", "verification", "status"} - set(item))
            if missing_item:
                raise PlanValidationError(f"items[{index}] missing fields: " + ", ".join(missing_item))
            items.append(
                PlanSnapshotItem(
                    id=_optional_text(item.get("id")),
                    step=_required_text(item.get("step"), f"items[{index}].step"),
                    details=_required_text(item.get("details"), f"items[{index}].details"),
                    verification=_required_text(item.get("verification"), f"items[{index}].verification"),
                    status=ensure_plan_item_status(item.get("status")),
                )
            )
        return cls(
            raw_user_request=_optional_text(raw.get("raw_user_request")),
            interpreted_goal=_optional_text(raw.get("interpreted_goal")),
            summary=_required_text(raw.get("summary"), "plan summary"),
            completion_criteria=tuple(criteria),
            items=tuple(items),
            task_understanding=_optional_text(raw.get("task_understanding")),
            current_implementation=_optional_text(raw.get("current_implementation")),
            target_design=_optional_text(raw.get("target_design")),
            impact_scope=_optional_text(raw.get("impact_scope")),
            risks_and_open_questions=tuple(risks),
            verification_plan=_optional_text(raw.get("verification_plan")),
            status=cast(Any, raw.get("status")),
            change_reason=ensure_plan_change_reason(raw.get("change_reason")),
            explanation=_optional_text(raw.get("explanation")) or "",
        )

    def as_proposal(self) -> "PlanSnapshot":
        return replace(
            self,
            status=None,
            items=tuple(replace(item, status="pending") for item in self.items),
        )


@dataclass(frozen=True)
class PlanState:
    schema_version: int = PLAN_STATE_SCHEMA_VERSION
    plan_id: str = field(default_factory=lambda: f"plan_{uuid4().hex[:12]}")
    owner_run_id: str = ""
    status: PlanStatus = "active"
    origin_mode: RunMode = "build"
    raw_user_request: str = ""
    interpreted_goal: str = ""
    task_understanding: str = ""
    current_implementation: str = ""
    target_design: str = ""
    impact_scope: str = ""
    risks_and_open_questions: tuple[str, ...] = field(default_factory=tuple)
    verification_plan: str = ""
    summary: str = ""
    completion_criteria: tuple[str, ...] = field(default_factory=tuple)
    items: tuple[PlanItem, ...] = field(default_factory=tuple)
    revision: int = 0
    explanation: str = ""
    created_at: str = field(default_factory=lambda: _utc_now_iso())
    updated_at: str = field(default_factory=lambda: _utc_now_iso())
    completed_at: str | None = None
    completion_source: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != PLAN_STATE_SCHEMA_VERSION:
            raise PlanValidationError("unsupported plan state schema")
        object.__setattr__(self, "plan_id", _required_text(self.plan_id, "plan_id"))
        object.__setattr__(self, "owner_run_id", _required_text(self.owner_run_id, "owner_run_id"))
        object.__setattr__(self, "status", ensure_plan_status(self.status))
        object.__setattr__(self, "origin_mode", ensure_run_mode(self.origin_mode))
        object.__setattr__(self, "raw_user_request", _required_text(self.raw_user_request, "raw_user_request"))
        object.__setattr__(self, "interpreted_goal", _required_text(self.interpreted_goal, "interpreted_goal"))
        object.__setattr__(self, "task_understanding", _optional_text(self.task_understanding) or "")
        object.__setattr__(self, "current_implementation", _optional_text(self.current_implementation) or "")
        object.__setattr__(self, "target_design", _optional_text(self.target_design) or "")
        object.__setattr__(self, "impact_scope", _optional_text(self.impact_scope) or "")
        object.__setattr__(
            self,
            "risks_and_open_questions",
            _normalize_text_tuple(self.risks_and_open_questions, "risks_and_open_questions", max_items=8),
        )
        object.__setattr__(self, "verification_plan", _optional_text(self.verification_plan) or "")
        object.__setattr__(self, "summary", _optional_text(self.summary) or "")
        criteria = tuple(_required_text(value, "completion criterion") for value in self.completion_criteria)
        if not criteria or len(criteria) > 5:
            raise PlanValidationError("completion_criteria must contain between 1 and 5 items")
        object.__setattr__(self, "completion_criteria", criteria)
        items = tuple(self.items)
        if not items or len(items) > MAX_PLAN_ITEMS:
            raise PlanValidationError(f"plan must contain between 1 and {MAX_PLAN_ITEMS} items")
        if sum(item.status == "in_progress" for item in items) > 1:
            raise PlanValidationError("plan can contain at most one in_progress item")
        ids = [item.id for item in items]
        if len(ids) != len(set(ids)):
            raise PlanValidationError("plan item ids must be unique")
        object.__setattr__(self, "items", items)
        if not isinstance(self.revision, int) or isinstance(self.revision, bool) or self.revision < 0:
            raise PlanValidationError("revision must be a non-negative integer")
        object.__setattr__(self, "explanation", _optional_text(self.explanation) or "")
        object.__setattr__(self, "created_at", _required_text(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _required_text(self.updated_at, "updated_at"))
        object.__setattr__(self, "completed_at", _optional_text(self.completed_at))
        object.__setattr__(self, "completion_source", _optional_text(self.completion_source))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PlanState":
        if raw.get("schema_version") != PLAN_STATE_SCHEMA_VERSION:
            raise PlanValidationError("unsupported plan state schema")
        unknown = sorted(set(raw) - _PLAN_KEYS)
        if unknown:
            raise PlanValidationError("unknown plan state fields: " + ", ".join(unknown))
        missing = sorted(_PLAN_KEYS - set(raw))
        if missing:
            raise PlanValidationError("missing plan state fields: " + ", ".join(missing))
        items_raw = raw.get("items")
        if not isinstance(items_raw, (list, tuple)):
            raise PlanValidationError("items must be a list")
        items = tuple(
            PlanItem(
                id=_required_text(item.get("id"), f"items[{index}].id"),
                step=_required_text(item.get("step"), f"items[{index}].step"),
                details=_required_text(item.get("details"), f"items[{index}].details"),
                verification=_required_text(item.get("verification"), f"items[{index}].verification"),
                status=ensure_plan_item_status(item.get("status")),
            )
            for index, item in enumerate(items_raw)
            if isinstance(item, Mapping)
        )
        if len(items) != len(items_raw):
            raise PlanValidationError("plan items must be objects")
        criteria = raw.get("completion_criteria")
        if not isinstance(criteria, (list, tuple)):
            raise PlanValidationError("completion_criteria must be a list")
        risks = raw.get("risks_and_open_questions")
        if not isinstance(risks, (list, tuple)):
            raise PlanValidationError("risks_and_open_questions must be a list")
        return cls(
            schema_version=PLAN_STATE_SCHEMA_VERSION,
            plan_id=_required_text(raw.get("plan_id"), "plan_id"),
            owner_run_id=_required_text(raw.get("owner_run_id"), "owner_run_id"),
            status=ensure_plan_status(raw.get("status")),
            origin_mode=ensure_run_mode(raw.get("origin_mode")),
            raw_user_request=_required_text(raw.get("raw_user_request"), "raw_user_request"),
            interpreted_goal=_required_text(raw.get("interpreted_goal"), "interpreted_goal"),
            task_understanding=_optional_text(raw.get("task_understanding")) or "",
            current_implementation=_optional_text(raw.get("current_implementation")) or "",
            target_design=_optional_text(raw.get("target_design")) or "",
            impact_scope=_optional_text(raw.get("impact_scope")) or "",
            risks_and_open_questions=tuple(risks),
            verification_plan=_optional_text(raw.get("verification_plan")) or "",
            summary=_required_text(raw.get("summary"), "summary"),
            completion_criteria=tuple(criteria),
            items=items,
            revision=_ensure_non_negative_int(raw.get("revision"), "revision"),
            explanation=_optional_text(raw.get("explanation")) or "",
            created_at=_required_text(raw.get("created_at"), "created_at"),
            updated_at=_required_text(raw.get("updated_at"), "updated_at"),
            completed_at=_optional_text(raw.get("completed_at")),
            completion_source=_optional_text(raw.get("completion_source")),
        )

    def approve(self) -> "PlanState":
        if self.status != "proposed":
            raise PlanValidationError("only a proposed plan can be approved")
        return replace(self, status="active", updated_at=_utc_now_iso())

    def reject(self) -> "PlanState":
        if self.status != "proposed":
            raise PlanValidationError("only a proposed plan can be rejected")
        return replace(self, status="rejected", updated_at=_utc_now_iso())

    def abandon(self, *, source: str) -> "PlanState":
        if self.status in {"completed", "rejected", "abandoned"}:
            return self
        now = _utc_now_iso()
        return replace(self, status="abandoned", completed_at=now, completion_source=_required_text(source, "completion_source"), updated_at=now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "owner_run_id": self.owner_run_id,
            "status": self.status,
            "origin_mode": self.origin_mode,
            "raw_user_request": self.raw_user_request,
            "interpreted_goal": self.interpreted_goal,
            "task_understanding": self.task_understanding,
            "current_implementation": self.current_implementation,
            "target_design": self.target_design,
            "impact_scope": self.impact_scope,
            "risks_and_open_questions": list(self.risks_and_open_questions),
            "verification_plan": self.verification_plan,
            "summary": self.summary,
            "completion_criteria": list(self.completion_criteria),
            "items": [item.to_dict() for item in self.items],
            "revision": self.revision,
            "explanation": self.explanation,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "completion_source": self.completion_source,
        }

    def to_summary(self) -> PlanSummary:
        return PlanSummary(
            schema_version=self.schema_version,
            plan_id=self.plan_id,
            owner_run_id=self.owner_run_id,
            status=self.status,
            origin_mode=self.origin_mode,
            raw_user_request=self.raw_user_request,
            interpreted_goal=self.interpreted_goal,
            task_understanding=self.task_understanding,
            current_implementation=self.current_implementation,
            target_design=self.target_design,
            impact_scope=self.impact_scope,
            risks_and_open_questions=list(self.risks_and_open_questions),
            verification_plan=self.verification_plan,
            summary=self.summary,
            completion_criteria=list(self.completion_criteria),
            items=[item.to_dict() for item in self.items],
            revision=self.revision,
            explanation=self.explanation,
            created_at=self.created_at,
            updated_at=self.updated_at,
            completed_at=self.completed_at,
            completion_source=self.completion_source,
        )


def apply_plan_snapshot(
    state: PlanState | None,
    snapshot: PlanSnapshot,
    *,
    mode: RunMode,
    run_id: str,
    operation: PlanOperation | str | None = None,
    qualified_failure_count: int = 0,
) -> PlanState:
    run_mode = ensure_run_mode(mode)
    plan_operation = ensure_plan_operation(operation or _default_plan_operation(run_mode, state))
    if run_mode == "read":
        raise PlanValidationError("read mode cannot update a plan")
    if state is None:
        if plan_operation == "propose_plan" and run_mode == "plan":
            return _create_plan(snapshot, mode=run_mode, run_id=run_id)
        if plan_operation == "create_build_plan" and run_mode == "build":
            return _create_plan(snapshot, mode=run_mode, run_id=run_id)
        raise PlanValidationError(f"{plan_operation} cannot create a plan in {run_mode} mode")
    if plan_operation == "create_build_plan":
        raise PlanValidationError("cannot create a build plan while a current Task Plan exists")
    if state.status in {"completed", "rejected", "abandoned"}:
        raise PlanValidationError(f"{state.status} plan cannot be changed by the model")
    if state.status == "proposed":
        if run_mode != "plan" or plan_operation != "propose_plan":
            raise PlanValidationError("proposed plan must be approved before execution")
        return _revise_proposal(state, snapshot)
    if run_mode != "build" or plan_operation not in {"update_plan_progress", "close_plan"}:
        raise PlanValidationError("active plan can only be updated in build mode")
    return _update_active_plan(
        state,
        snapshot,
        operation=plan_operation,
        qualified_failure_count=qualified_failure_count,
    )


def load_plan_state(raw: object) -> PlanState | None:
    if raw is None:
        return None
    if isinstance(raw, PlanState):
        return raw
    if not isinstance(raw, Mapping):
        raise PlanValidationError("plan state must be an object")
    return PlanState.from_mapping(raw)


def plan_state_to_dict(state: PlanState | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if state is None:
        return None
    if isinstance(state, PlanState):
        return state.to_dict()
    return deepcopy(dict(state))


def ensure_run_mode(value: object) -> RunMode:
    text = str(value).strip() if value is not None else ""
    if text not in _RUN_MODES:
        raise ValueError(f"Unknown run mode: {value}")
    return cast(RunMode, text)


def ensure_planning_budget_profile(value: object) -> PlanningBudgetProfile:
    text = str(value).strip() if value is not None else ""
    if text not in _PLANNING_BUDGET_PROFILES:
        raise ValueError(f"Unknown planning budget profile: {value}")
    return cast(PlanningBudgetProfile, text)


def ensure_plan_status(value: object) -> PlanStatus:
    text = str(value).strip() if value is not None else ""
    if text not in _PLAN_STATUSES:
        raise PlanValidationError(f"Unknown plan status: {value}")
    return cast(PlanStatus, text)


def ensure_plan_item_status(value: object) -> PlanItemStatus:
    text = str(value).strip() if value is not None else ""
    if text not in _ITEM_STATUSES:
        raise PlanValidationError(f"Unknown plan item status: {value}")
    return cast(PlanItemStatus, text)


def ensure_plan_change_reason(value: object) -> PlanChangeReason | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text not in _CHANGE_REASONS:
        raise PlanValidationError(f"Unknown plan change reason: {value}")
    return cast(PlanChangeReason, text)


def ensure_plan_operation(value: object) -> PlanOperation:
    text = str(value).strip() if value is not None else ""
    if text not in _PLAN_OPERATIONS:
        raise PlanValidationError(f"Unknown plan operation: {value}")
    return cast(PlanOperation, text)


def _default_plan_operation(run_mode: RunMode, state: PlanState | None) -> PlanOperation:
    if run_mode == "plan":
        return "propose_plan"
    if state is None:
        return "create_build_plan"
    return "update_plan_progress"


def _create_plan(snapshot: PlanSnapshot, *, mode: RunMode, run_id: str) -> PlanState:
    raw_user_request = _required_text(snapshot.raw_user_request, "raw_user_request")
    interpreted_goal = _required_text(snapshot.interpreted_goal, "interpreted_goal")
    if mode == "plan":
        snapshot = snapshot.as_proposal()
        _require_proposal_details(snapshot)
        status: PlanStatus = "proposed"
    else:
        if snapshot.status == "completed":
            raise PlanValidationError("new build plan cannot be completed")
        status = "active"
    now = _utc_now_iso()
    return PlanState(
        plan_id=f"plan_{uuid4().hex[:12]}",
        owner_run_id=_required_text(run_id, "run_id"),
        status=status,
        origin_mode=mode,
        raw_user_request=raw_user_request,
        interpreted_goal=interpreted_goal,
        task_understanding=snapshot.task_understanding or "",
        current_implementation=snapshot.current_implementation or "",
        target_design=snapshot.target_design or "",
        impact_scope=snapshot.impact_scope or "",
        risks_and_open_questions=snapshot.risks_and_open_questions,
        verification_plan=snapshot.verification_plan or "",
        summary=snapshot.summary,
        completion_criteria=snapshot.completion_criteria,
        items=tuple(_materialize_items(snapshot.items)),
        revision=1,
        explanation=snapshot.explanation,
        created_at=now,
        updated_at=now,
    )


def _revise_proposal(state: PlanState, snapshot: PlanSnapshot) -> PlanState:
    if snapshot.status is not None:
        raise PlanValidationError("plan proposal cannot set status")
    proposal = snapshot.as_proposal()
    _require_proposal_details(proposal)
    return replace(
        state,
        raw_user_request=proposal.raw_user_request or state.raw_user_request,
        interpreted_goal=proposal.interpreted_goal or state.interpreted_goal,
        task_understanding=proposal.task_understanding or state.task_understanding,
        current_implementation=proposal.current_implementation or state.current_implementation,
        target_design=proposal.target_design or state.target_design,
        impact_scope=proposal.impact_scope or state.impact_scope,
        risks_and_open_questions=proposal.risks_and_open_questions,
        verification_plan=proposal.verification_plan or state.verification_plan,
        summary=proposal.summary,
        completion_criteria=proposal.completion_criteria,
        items=tuple(_materialize_items(proposal.items, existing=state.items)),
        revision=state.revision + 1,
        explanation=proposal.explanation,
        updated_at=_utc_now_iso(),
    )


def _update_active_plan(
    state: PlanState,
    snapshot: PlanSnapshot,
    *,
    operation: PlanOperation,
    qualified_failure_count: int,
) -> PlanState:
    if snapshot.raw_user_request is not None:
        raise PlanValidationError("active plan updates cannot set raw_user_request")
    if snapshot.interpreted_goal is not None:
        raise PlanValidationError("active plan updates cannot set interpreted_goal")
    missing_ids = any(item.id is None for item in snapshot.items)
    if missing_ids and snapshot.change_reason is None:
        raise PlanValidationError("active plan snapshots must include canonical item ids")
    candidate = tuple(_materialize_items(snapshot.items, existing=state.items))
    structure_changed = _structure_changed(state, snapshot, candidate)
    if structure_changed:
        if state.origin_mode == "plan":
            raise PlanValidationError("Plan-mode approved active plan cannot be structurally replaced by Build")
        if snapshot.change_reason is None:
            raise PlanValidationError("active plan structure changed without a change reason")
        if snapshot.change_reason == "repeated_execution_failure" and qualified_failure_count < QUALIFIED_FAILURES_FOR_REVISION:
            raise PlanValidationError("active plan structure changed before five qualified failures")
    elif snapshot.change_reason is not None:
        raise PlanValidationError("change_reason requires a structural plan change")
    if snapshot.status not in {None, "active", "completed"}:
        raise PlanValidationError("invalid active plan status")
    if operation == "update_plan_progress" and snapshot.status == "completed":
        raise PlanValidationError("use close_plan to complete the current Task Plan")
    if operation == "close_plan" and snapshot.status is None:
        raise PlanValidationError("close_plan must set status to active or completed")
    now = _utc_now_iso()
    next_state = replace(
        state,
        summary=snapshot.summary,
        completion_criteria=snapshot.completion_criteria,
        items=candidate,
        revision=state.revision + 1,
        explanation=snapshot.explanation,
        updated_at=now,
    )
    if snapshot.status == "completed":
        return replace(
            next_state,
            status="completed",
            completed_at=now,
            completion_source="model_closeout",
        )
    return next_state


def _materialize_items(
    items: tuple[PlanSnapshotItem, ...],
    *,
    existing: tuple[PlanItem, ...] = (),
    require_ids: bool = False,
) -> list[PlanItem]:
    existing_ids = {item.id for item in existing}
    allocated = set(existing_ids)
    materialized: list[PlanItem] = []
    for index, item in enumerate(items, start=1):
        if require_ids and item.id is None:
            raise PlanValidationError("active plan snapshots must include canonical item ids")
        item_id = item.id or _next_item_id(index, allocated)
        allocated.add(item_id)
        materialized.append(
            PlanItem(
                id=item_id,
                step=item.step,
                details=item.details,
                verification=item.verification,
                status=item.status,
            )
        )
    return materialized


def _next_item_id(index: int, allocated: set[str]) -> str:
    candidate = f"item_{index}"
    while candidate in allocated:
        index += 1
        candidate = f"item_{index}"
    return candidate


def _structure_changed(state: PlanState, snapshot: PlanSnapshot, items: tuple[PlanItem, ...]) -> bool:
    if tuple(snapshot.completion_criteria) != state.completion_criteria:
        return True
    if len(items) != len(state.items):
        return True
    for previous, current in zip(state.items, items):
        if (
            previous.id != current.id
            or previous.step != current.step
            or previous.details != current.details
            or previous.verification != current.verification
        ):
            return True
    return False


def _require_proposal_details(snapshot: PlanSnapshot) -> None:
    required = {
        "task_understanding": snapshot.task_understanding,
        "current_implementation": snapshot.current_implementation,
        "target_design": snapshot.target_design,
        "impact_scope": snapshot.impact_scope,
        "verification_plan": snapshot.verification_plan,
    }
    for field_name, value in required.items():
        _required_text(value, field_name)
    if not snapshot.risks_and_open_questions:
        raise PlanValidationError("risks_and_open_questions must contain at least 1 item")


def _ensure_non_negative_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PlanValidationError(f"{field_name} must be a non-negative integer")
    return value


def _normalize_text_tuple(
    values: tuple[str, ...] | list[str],
    field_name: str,
    *,
    max_items: int,
) -> tuple[str, ...]:
    items = tuple(_required_text(value, f"{field_name}[{index}]") for index, value in enumerate(values))
    if len(items) > max_items:
        raise PlanValidationError(f"{field_name} cannot contain more than {max_items} items")
    return items


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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "MAX_PLAN_ITEMS",
    "PLAN_STATE_SCHEMA_VERSION",
    "QUALIFIED_FAILURES_FOR_REVISION",
    "PlanChangeReason",
    "PlanItem",
    "PlanItemStatus",
    "PlanOperation",
    "PlanSnapshot",
    "PlanSnapshotItem",
    "PlanState",
    "PlanStatus",
    "PlanValidationError",
    "PlanningBudgetProfile",
    "RunMode",
    "apply_plan_snapshot",
    "ensure_plan_change_reason",
    "ensure_plan_item_status",
    "ensure_plan_operation",
    "ensure_plan_status",
    "ensure_planning_budget_profile",
    "ensure_run_mode",
    "load_plan_state",
    "plan_state_to_dict",
]
