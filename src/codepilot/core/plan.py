from __future__ import annotations

"""Soft plan state used by the agent loop.

PlanState is a model-visible progress board. It helps the assistant explain and
resume work, but it never proves that the user's task is complete.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, cast
from uuid import uuid4

from codepilot.protocols import PlanSummary


RunMode = Literal["read", "plan", "build"]
PlanningBudgetProfile = Literal["conservative", "balanced", "wide"]
PlanStatus = Literal["none", "proposed", "active", "completed", "rejected", "abandoned"]
PlanApprovalState = Literal["none", "proposed", "approved", "rejected"]
PlanItemStatus = Literal["pending", "in_progress", "completed"]

_RUN_MODES = frozenset({"read", "plan", "build"})
_PLANNING_BUDGET_PROFILES = frozenset({"conservative", "balanced", "wide"})
_PLAN_STATUSES = frozenset(
    {"none", "proposed", "active", "completed", "rejected", "abandoned"}
)
_APPROVAL_STATES = frozenset({"none", "proposed", "approved", "rejected"})
_ITEM_STATUSES = frozenset({"pending", "in_progress", "completed"})
_PLAN_KEYS = frozenset(
    {
        "schema_version",
        "plan_id",
        "status",
        "approval_state",
        "origin_mode",
        "objective",
        "items",
        "explanation",
        "created_at",
        "updated_at",
        "last_update_run_id",
    }
)
_ITEM_KEYS = frozenset({"id", "step", "status"})

PLAN_STATE_SCHEMA_VERSION = 1
MAX_PLAN_ITEMS = 10


class PlanValidationError(ValueError):
    """Raised when a plan payload cannot be used as soft PlanState."""


@dataclass(frozen=True)
class PlanItem:
    id: str
    step: str
    status: PlanItemStatus = "pending"

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_text(self.id, "plan item id"))
        object.__setattr__(self, "step", _required_text(self.step, "plan item step"))
        object.__setattr__(self, "status", ensure_plan_item_status(self.status))

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "step": self.step, "status": self.status}


@dataclass(frozen=True)
class PlanUpdateItem:
    step: str
    status: PlanItemStatus

    def __post_init__(self) -> None:
        object.__setattr__(self, "step", _required_text(self.step, "plan update step"))
        object.__setattr__(self, "status", ensure_plan_item_status(self.status))


@dataclass(frozen=True)
class PlanUpdate:
    explanation: str = ""
    items: tuple[PlanUpdateItem, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "explanation", _optional_text(self.explanation) or "")
        items = tuple(self.items)
        if not items:
            raise PlanValidationError("plan must contain at least one item")
        if len(items) > MAX_PLAN_ITEMS:
            raise PlanValidationError(f"plan cannot contain more than {MAX_PLAN_ITEMS} items")
        in_progress = sum(item.status == "in_progress" for item in items)
        if in_progress > 1:
            raise PlanValidationError("plan can contain at most one in_progress item")
        object.__setattr__(self, "items", items)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PlanUpdate":
        plan = raw.get("plan")
        if not isinstance(plan, list):
            raise PlanValidationError("plan must be a list")
        items: list[PlanUpdateItem] = []
        for index, item in enumerate(plan):
            if not isinstance(item, Mapping):
                raise PlanValidationError(f"plan[{index}] must be an object")
            items.append(
                PlanUpdateItem(
                    step=_required_text(item.get("step"), f"plan[{index}].step"),
                    status=ensure_plan_item_status(item.get("status")),
                )
            )
        return cls(
            explanation=_optional_text(raw.get("explanation")) or "",
            items=tuple(items),
        )


@dataclass(frozen=True)
class PlanState:
    schema_version: int = PLAN_STATE_SCHEMA_VERSION
    plan_id: str = field(default_factory=lambda: f"plan_{uuid4().hex[:12]}")
    status: PlanStatus = "none"
    approval_state: PlanApprovalState = "none"
    origin_mode: RunMode = "build"
    objective: str = ""
    items: tuple[PlanItem, ...] = field(default_factory=tuple)
    explanation: str = ""
    created_at: str = field(default_factory=lambda: _utc_now_iso())
    updated_at: str = field(default_factory=lambda: _utc_now_iso())
    last_update_run_id: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != PLAN_STATE_SCHEMA_VERSION:
            raise PlanValidationError("unsupported plan state schema")
        object.__setattr__(self, "plan_id", _required_text(self.plan_id, "plan_id"))
        object.__setattr__(self, "status", ensure_plan_status(self.status))
        object.__setattr__(
            self,
            "approval_state",
            ensure_plan_approval_state(self.approval_state),
        )
        object.__setattr__(self, "origin_mode", ensure_run_mode(self.origin_mode))
        object.__setattr__(self, "objective", _optional_text(self.objective) or "")
        items = tuple(self.items)
        if len(items) > MAX_PLAN_ITEMS:
            raise PlanValidationError(f"plan cannot contain more than {MAX_PLAN_ITEMS} items")
        if sum(item.status == "in_progress" for item in items) > 1:
            raise PlanValidationError("plan can contain at most one in_progress item")
        ids = [item.id for item in items]
        if len(ids) != len(set(ids)):
            raise PlanValidationError("plan item ids must be unique")
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "explanation", _optional_text(self.explanation) or "")
        object.__setattr__(self, "created_at", _required_text(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _required_text(self.updated_at, "updated_at"))
        object.__setattr__(
            self,
            "last_update_run_id",
            _optional_text(self.last_update_run_id),
        )

    @classmethod
    def new(
        cls,
        *,
        objective: str = "",
        origin_mode: RunMode = "build",
        run_id: str | None = None,
    ) -> "PlanState":
        now = _utc_now_iso()
        return cls(
            objective=_optional_text(objective) or "",
            origin_mode=ensure_run_mode(origin_mode),
            created_at=now,
            updated_at=now,
            last_update_run_id=_optional_text(run_id),
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PlanState":
        unknown = sorted(set(raw) - _PLAN_KEYS)
        if unknown:
            raise PlanValidationError("unknown plan state fields: " + ", ".join(unknown))
        missing = sorted(_PLAN_KEYS - set(raw))
        if missing:
            raise PlanValidationError("missing plan state fields: " + ", ".join(missing))
        items_raw = raw.get("items")
        if not isinstance(items_raw, list):
            raise PlanValidationError("items must be a list")
        items: list[PlanItem] = []
        for index, item in enumerate(items_raw):
            if not isinstance(item, Mapping):
                raise PlanValidationError(f"items[{index}] must be an object")
            keys = set(item)
            unknown_item = sorted(keys - _ITEM_KEYS)
            if unknown_item:
                raise PlanValidationError(
                    f"items[{index}] has unknown fields: " + ", ".join(unknown_item)
                )
            missing_item = sorted(_ITEM_KEYS - keys)
            if missing_item:
                raise PlanValidationError(
                    f"items[{index}] missing fields: " + ", ".join(missing_item)
                )
            items.append(
                PlanItem(
                    id=_required_text(item.get("id"), f"items[{index}].id"),
                    step=_required_text(item.get("step"), f"items[{index}].step"),
                    status=ensure_plan_item_status(item.get("status")),
                )
            )
        return cls(
            schema_version=_ensure_schema_version(raw.get("schema_version")),
            plan_id=_required_text(raw.get("plan_id"), "plan_id"),
            status=ensure_plan_status(raw.get("status")),
            approval_state=ensure_plan_approval_state(raw.get("approval_state")),
            origin_mode=ensure_run_mode(raw.get("origin_mode")),
            objective=_optional_text(raw.get("objective")) or "",
            items=tuple(items),
            explanation=_optional_text(raw.get("explanation")) or "",
            created_at=_required_text(raw.get("created_at"), "created_at"),
            updated_at=_required_text(raw.get("updated_at"), "updated_at"),
            last_update_run_id=_optional_text(raw.get("last_update_run_id")),
        )

    def apply_update(
        self,
        update: PlanUpdate,
        *,
        mode: RunMode,
        run_id: str,
    ) -> "PlanState":
        if self.status == "rejected":
            raise PlanValidationError("rejected plan cannot be changed by the model")
        now = _utc_now_iso()
        next_status = self.status
        if next_status in {"none", "abandoned"}:
            next_status = "proposed" if ensure_run_mode(mode) == "plan" else "active"
        if next_status == "active" and all(item.status == "completed" for item in update.items):
            next_status = "completed"
        approval_state = self.approval_state
        if next_status == "proposed":
            approval_state = "proposed"
        elif next_status in {"active", "completed"} and approval_state == "none":
            approval_state = "approved"
        return PlanState(
            plan_id=self.plan_id,
            status=next_status,
            approval_state=approval_state,
            origin_mode=ensure_run_mode(mode),
            objective=self.objective,
            items=tuple(_items_from_update(self.items, update)),
            explanation=update.explanation,
            created_at=self.created_at,
            updated_at=now,
            last_update_run_id=run_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "status": self.status,
            "approval_state": self.approval_state,
            "origin_mode": self.origin_mode,
            "objective": self.objective,
            "items": [item.to_dict() for item in self.items],
            "explanation": self.explanation,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_update_run_id": self.last_update_run_id,
        }

    def summary(self) -> PlanSummary:
        return PlanSummary(
            schema_version=self.schema_version,
            plan_id=self.plan_id,
            status=self.status,
            approval_state=self.approval_state,
            origin_mode=self.origin_mode,
            objective=self.objective,
            items=[item.to_dict() for item in self.items],
            explanation=self.explanation,
            created_at=self.created_at,
            updated_at=self.updated_at,
            last_update_run_id=self.last_update_run_id,
        )


def load_plan_state(raw: object) -> PlanState | None:
    if raw is None:
        return None
    if isinstance(raw, PlanState):
        return raw
    if not isinstance(raw, Mapping):
        raise PlanValidationError("plan state must be an object")
    return PlanState.from_mapping(raw)


def apply_plan_update_metadata(
    state: PlanState | None,
    metadata: Mapping[str, Any],
    *,
    mode: RunMode,
    objective: str,
    run_id: str,
) -> PlanState | None:
    raw_update = metadata.get("plan_update")
    if not isinstance(raw_update, Mapping):
        return state
    update = PlanUpdate.from_mapping(raw_update)
    current = state or PlanState.new(objective=objective, origin_mode=mode, run_id=run_id)
    if not current.objective and objective:
        current = PlanState(
            plan_id=current.plan_id,
            status=current.status,
            approval_state=current.approval_state,
            origin_mode=current.origin_mode,
            objective=objective,
            items=current.items,
            explanation=current.explanation,
            created_at=current.created_at,
            updated_at=current.updated_at,
            last_update_run_id=current.last_update_run_id,
        )
    return current.apply_update(update, mode=mode, run_id=run_id)


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


def ensure_plan_approval_state(value: object) -> PlanApprovalState:
    text = str(value).strip() if value is not None else ""
    if text not in _APPROVAL_STATES:
        raise PlanValidationError(f"Unknown plan approval state: {value}")
    return cast(PlanApprovalState, text)


def ensure_plan_item_status(value: object) -> PlanItemStatus:
    text = str(value).strip() if value is not None else ""
    if text not in _ITEM_STATUSES:
        raise PlanValidationError(f"Unknown plan item status: {value}")
    return cast(PlanItemStatus, text)


def _items_from_update(
    current_items: tuple[PlanItem, ...],
    update: PlanUpdate,
) -> list[PlanItem]:
    ids_by_step = {item.step: item.id for item in current_items}
    items: list[PlanItem] = []
    for index, item in enumerate(update.items):
        item_id = ids_by_step.get(item.step) or f"item_{index + 1}"
        items.append(PlanItem(id=item_id, step=item.step, status=item.status))
    return items


def _ensure_schema_version(value: object) -> int:
    if value != PLAN_STATE_SCHEMA_VERSION:
        raise PlanValidationError("unsupported plan state schema")
    return PLAN_STATE_SCHEMA_VERSION


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


def plan_state_to_dict(state: PlanState | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if state is None:
        return None
    if isinstance(state, PlanState):
        return state.to_dict()
    return deepcopy(dict(state))


__all__ = [
    "MAX_PLAN_ITEMS",
    "PLAN_STATE_SCHEMA_VERSION",
    "PlanApprovalState",
    "PlanItem",
    "PlanItemStatus",
    "PlanState",
    "PlanStatus",
    "PlanUpdate",
    "PlanUpdateItem",
    "PlanValidationError",
    "PlanningBudgetProfile",
    "RunMode",
    "apply_plan_update_metadata",
    "ensure_plan_approval_state",
    "ensure_plan_item_status",
    "ensure_plan_status",
    "ensure_planning_budget_profile",
    "ensure_run_mode",
    "load_plan_state",
    "plan_state_to_dict",
]
