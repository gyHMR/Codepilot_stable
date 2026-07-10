from __future__ import annotations

"""Canonical task-plan state and the single snapshot update protocol."""

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

PLAN_STATE_SCHEMA_VERSION = 4
MAX_PLAN_ITEMS = PLAN_ITEM_LIMIT
QUALIFIED_FAILURES_FOR_REVISION = 5

_RUN_MODES = frozenset({"read", "plan", "build"})
_PLANNING_BUDGET_PROFILES = frozenset({"conservative", "balanced", "wide"})
_PLAN_STATUSES = frozenset({"proposed", "active", "completed", "rejected", "abandoned"})
_ITEM_STATUSES = frozenset({"pending", "in_progress", "completed"})
_CHANGE_REASONS = frozenset({"user_request", "repeated_execution_failure"})
_PLAN_KEYS = frozenset(
    {
        "schema_version",
        "plan_id",
        "owner_run_id",
        "status",
        "origin_mode",
        "objective",
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
        "execution_objective",
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
    execution_objective: str | None
    summary: str
    completion_criteria: tuple[str, ...]
    items: tuple[PlanSnapshotItem, ...]
    status: Literal["active", "completed"] | None = None
    change_reason: PlanChangeReason | None = None
    explanation: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution_objective", _optional_text(self.execution_objective))
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
        if not isinstance(criteria, list):
            raise PlanValidationError("completion_criteria must be a list")
        items_raw = raw.get("items")
        if not isinstance(items_raw, list):
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
            execution_objective=_optional_text(raw.get("execution_objective")),
            summary=_required_text(raw.get("summary"), "plan summary"),
            completion_criteria=tuple(criteria),
            items=tuple(items),
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
    objective: str = ""
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
        object.__setattr__(self, "objective", _required_text(self.objective, "objective"))
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
        if not isinstance(items_raw, list):
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
        if not isinstance(criteria, list):
            raise PlanValidationError("completion_criteria must be a list")
        return cls(
            schema_version=PLAN_STATE_SCHEMA_VERSION,
            plan_id=_required_text(raw.get("plan_id"), "plan_id"),
            owner_run_id=_required_text(raw.get("owner_run_id"), "owner_run_id"),
            status=ensure_plan_status(raw.get("status")),
            origin_mode=ensure_run_mode(raw.get("origin_mode")),
            objective=_required_text(raw.get("objective"), "objective"),
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
            "objective": self.objective,
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
            objective=self.objective,
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
    qualified_failure_count: int = 0,
) -> PlanState:
    run_mode = ensure_run_mode(mode)
    if run_mode == "read":
        raise PlanValidationError("read mode cannot update a plan")
    if state is None:
        return _create_plan(snapshot, mode=run_mode, run_id=run_id)
    if state.owner_run_id != _required_text(run_id, "run_id"):
        raise PlanValidationError("plan belongs to a different run")
    if state.status in {"completed", "rejected", "abandoned"}:
        raise PlanValidationError(f"{state.status} plan cannot be changed by the model")
    if state.status == "proposed":
        if run_mode != "plan":
            raise PlanValidationError("proposed plan must be approved before execution")
        return _revise_proposal(state, snapshot)
    if run_mode != "build":
        raise PlanValidationError("active plan can only be updated in build mode")
    return _update_active_plan(state, snapshot, qualified_failure_count=qualified_failure_count)


def apply_plan_snapshot_metadata(
    state: PlanState | None,
    metadata: Mapping[str, Any],
    *,
    mode: RunMode,
    run_id: str,
    qualified_failure_count: int = 0,
) -> PlanState | None:
    raw_snapshot = metadata.get("plan_snapshot")
    if not isinstance(raw_snapshot, Mapping):
        return state
    return apply_plan_snapshot(
        state,
        PlanSnapshot.from_mapping(raw_snapshot),
        mode=mode,
        run_id=run_id,
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


def _create_plan(snapshot: PlanSnapshot, *, mode: RunMode, run_id: str) -> PlanState:
    objective = _required_text(snapshot.execution_objective, "execution_objective")
    if mode == "plan":
        snapshot = snapshot.as_proposal()
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
        objective=objective,
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
    return replace(
        state,
        objective=proposal.execution_objective or state.objective,
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
    qualified_failure_count: int,
) -> PlanState:
    if snapshot.execution_objective is not None:
        raise PlanValidationError("active plan updates cannot set execution_objective")
    missing_ids = any(item.id is None for item in snapshot.items)
    if missing_ids and snapshot.change_reason is None:
        raise PlanValidationError("active plan snapshots must include canonical item ids")
    candidate = tuple(_materialize_items(snapshot.items, existing=state.items))
    structure_changed = _structure_changed(state, snapshot, candidate)
    if structure_changed:
        if snapshot.change_reason is None:
            raise PlanValidationError("active plan structure changed without a change reason")
        if snapshot.change_reason == "repeated_execution_failure" and qualified_failure_count < QUALIFIED_FAILURES_FOR_REVISION:
            raise PlanValidationError("active plan structure changed before five qualified failures")
    elif snapshot.change_reason is not None:
        raise PlanValidationError("change_reason requires a structural plan change")
    if snapshot.status not in {None, "active", "completed"}:
        raise PlanValidationError("invalid active plan status")
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


def _ensure_non_negative_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PlanValidationError(f"{field_name} must be a non-negative integer")
    return value


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
    "PlanSnapshot",
    "PlanSnapshotItem",
    "PlanState",
    "PlanStatus",
    "PlanValidationError",
    "PlanningBudgetProfile",
    "RunMode",
    "apply_plan_snapshot",
    "apply_plan_snapshot_metadata",
    "ensure_plan_change_reason",
    "ensure_plan_item_status",
    "ensure_plan_status",
    "ensure_planning_budget_profile",
    "ensure_run_mode",
    "load_plan_state",
    "plan_state_to_dict",
]
