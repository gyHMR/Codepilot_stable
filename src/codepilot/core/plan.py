from __future__ import annotations

"""Soft plan state used by the agent loop.

PlanState is a model-visible progress board. It helps the assistant explain and
resume work, but it never proves that the user's task is complete.
"""

from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, cast
from uuid import uuid4

from codepilot.protocols import PLAN_ITEM_LIMIT, PlanSummary


RunMode = Literal["read", "plan", "build"]
PlanningBudgetProfile = Literal["conservative", "balanced", "wide"]
PlanStatus = Literal["proposed", "active", "completed", "rejected", "abandoned"]
PlanCloseoutStatus = Literal["active", "completed"]
PlanApprovalState = Literal["pending", "approved", "not_required", "rejected"]
PlanItemStatus = Literal["pending", "in_progress", "completed"]

_RUN_MODES = frozenset({"read", "plan", "build"})
_PLANNING_BUDGET_PROFILES = frozenset({"conservative", "balanced", "wide"})
_PLAN_STATUSES = frozenset({"proposed", "active", "completed", "rejected", "abandoned"})
_PLAN_CLOSEOUT_STATUSES = frozenset({"active", "completed"})
_APPROVAL_STATES = frozenset({"pending", "approved", "not_required", "rejected"})
_ITEM_STATUSES = frozenset({"pending", "in_progress", "completed"})
_PLAN_KEYS = frozenset(
    {
        "schema_version",
        "plan_id",
        "owner_run_id",
        "status",
        "approval_state",
        "origin_mode",
        "objective",
        "summary",
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

PLAN_STATE_SCHEMA_VERSION = 2
MAX_PLAN_ITEMS = PLAN_ITEM_LIMIT


class PlanValidationError(ValueError):
    """Raised when a plan payload cannot be used as soft PlanState."""


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
        object.__setattr__(
            self,
            "verification",
            _required_text(self.verification, "plan item verification"),
        )
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
class PlanUpdateItem:
    step: str
    details: str
    verification: str
    status: PlanItemStatus

    def __post_init__(self) -> None:
        object.__setattr__(self, "step", _required_text(self.step, "plan update step"))
        object.__setattr__(self, "details", _required_text(self.details, "plan update details"))
        object.__setattr__(
            self,
            "verification",
            _required_text(self.verification, "plan update verification"),
        )
        object.__setattr__(self, "status", ensure_plan_item_status(self.status))


@dataclass(frozen=True)
class PlanUpdate:
    summary: str
    explanation: str = ""
    items: tuple[PlanUpdateItem, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", _required_text(self.summary, "plan summary"))
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
    def from_mapping(cls, raw: Mapping[str, Any], *, proposal: bool = False) -> "PlanUpdate":
        plan = raw.get("plan")
        if not isinstance(plan, list):
            raise PlanValidationError("plan must be a list")
        items: list[PlanUpdateItem] = []
        for index, item in enumerate(plan):
            if not isinstance(item, Mapping):
                raise PlanValidationError(f"plan[{index}] must be an object")
            status = ensure_plan_item_status(item.get("status"))
            items.append(
                PlanUpdateItem(
                    step=_required_text(item.get("step"), f"plan[{index}].step"),
                    details=_required_text(item.get("details"), f"plan[{index}].details"),
                    verification=_required_text(
                        item.get("verification"),
                        f"plan[{index}].verification",
                    ),
                    status="pending" if proposal else status,
                )
            )
        return cls(
            summary=_required_text(raw.get("summary"), "plan summary"),
            explanation=_optional_text(raw.get("explanation")) or "",
            items=tuple(items),
        )

    def as_proposal(self) -> "PlanUpdate":
        if all(item.status == "pending" for item in self.items):
            return self
        return PlanUpdate(
            summary=self.summary,
            explanation=self.explanation,
            items=tuple(
                PlanUpdateItem(
                    step=item.step,
                    details=item.details,
                    verification=item.verification,
                    status="pending",
                )
                for item in self.items
            ),
        )


@dataclass(frozen=True)
class PlanState:
    schema_version: int = PLAN_STATE_SCHEMA_VERSION
    plan_id: str = field(default_factory=lambda: f"plan_{uuid4().hex[:12]}")
    owner_run_id: str = ""
    status: PlanStatus = "active"
    approval_state: PlanApprovalState = "not_required"
    origin_mode: RunMode = "build"
    objective: str = ""
    summary: str = ""
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
        object.__setattr__(
            self,
            "owner_run_id",
            _required_text(self.owner_run_id, "owner_run_id"),
        )
        object.__setattr__(self, "status", ensure_plan_status(self.status))
        object.__setattr__(
            self,
            "approval_state",
            ensure_plan_approval_state(self.approval_state),
        )
        object.__setattr__(self, "origin_mode", ensure_run_mode(self.origin_mode))
        object.__setattr__(self, "objective", _required_text(self.objective, "objective"))
        object.__setattr__(self, "summary", _optional_text(self.summary) or "")
        items = tuple(self.items)
        if len(items) > MAX_PLAN_ITEMS:
            raise PlanValidationError(f"plan cannot contain more than {MAX_PLAN_ITEMS} items")
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
        object.__setattr__(
            self,
            "completed_at",
            _optional_text(self.completed_at),
        )
        object.__setattr__(
            self,
            "completion_source",
            _optional_text(self.completion_source),
        )

    @classmethod
    def new(
        cls,
        *,
        objective: str,
        origin_mode: RunMode = "build",
        run_id: str,
    ) -> "PlanState":
        now = _utc_now_iso()
        mode = ensure_run_mode(origin_mode)
        return cls(
            owner_run_id=_required_text(run_id, "owner_run_id"),
            status="proposed" if mode == "plan" else "active",
            approval_state="pending" if mode == "plan" else "not_required",
            objective=_required_text(objective, "objective"),
            origin_mode=mode,
            created_at=now,
            updated_at=now,
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
                    details=_required_text(
                        item.get("details"),
                        f"items[{index}].details",
                    ),
                    verification=_required_text(
                        item.get("verification"),
                        f"items[{index}].verification",
                    ),
                    status=ensure_plan_item_status(item.get("status")),
                )
            )
        return cls(
            schema_version=_ensure_schema_version(raw.get("schema_version")),
            plan_id=_required_text(raw.get("plan_id"), "plan_id"),
            owner_run_id=_required_text(raw.get("owner_run_id"), "owner_run_id"),
            status=ensure_plan_status(raw.get("status")),
            approval_state=ensure_plan_approval_state(raw.get("approval_state")),
            origin_mode=ensure_run_mode(raw.get("origin_mode")),
            objective=_required_text(raw.get("objective"), "objective"),
            summary=_optional_text(raw.get("summary")) or "",
            items=tuple(items),
            revision=_ensure_non_negative_int(raw.get("revision"), "revision"),
            explanation=_optional_text(raw.get("explanation")) or "",
            created_at=_required_text(raw.get("created_at"), "created_at"),
            updated_at=_required_text(raw.get("updated_at"), "updated_at"),
            completed_at=_optional_text(raw.get("completed_at")),
            completion_source=_optional_text(raw.get("completion_source")),
        )

    def apply_update(
        self,
        update: PlanUpdate,
        *,
        mode: RunMode,
        run_id: str,
    ) -> "PlanState":
        if self.status in {"completed", "rejected", "abandoned"}:
            raise PlanValidationError(f"{self.status} plan cannot be changed by the model")
        if self.owner_run_id != _required_text(run_id, "run_id"):
            raise PlanValidationError("plan belongs to a different run")
        run_mode = ensure_run_mode(mode)
        if run_mode == "plan" and self.status != "proposed":
            raise PlanValidationError("active plan must be abandoned before replanning")
        if run_mode != "plan" and self.status == "proposed":
            raise PlanValidationError("proposed plan must be approved before execution")
        if run_mode == "plan":
            update = update.as_proposal()
        now = _utc_now_iso()
        return replace(
            self,
            summary=update.summary,
            items=tuple(_items_from_update(self.items, update)),
            revision=self.revision + 1,
            explanation=update.explanation,
            updated_at=now,
        )

    def approve(self) -> "PlanState":
        if self.status != "proposed" or self.approval_state != "pending":
            raise PlanValidationError("only a pending proposed plan can be approved")
        return replace(
            self,
            status="active",
            approval_state="approved",
            updated_at=_utc_now_iso(),
        )

    def reject(self) -> "PlanState":
        if self.status != "proposed" or self.approval_state != "pending":
            raise PlanValidationError("only a pending proposed plan can be rejected")
        return replace(
            self,
            status="rejected",
            approval_state="rejected",
            updated_at=_utc_now_iso(),
        )

    def complete(self, *, source: str) -> "PlanState":
        if self.status != "active":
            raise PlanValidationError("only an active plan can be completed")
        now = _utc_now_iso()
        return replace(
            self,
            status="completed",
            completed_at=now,
            completion_source=_required_text(source, "completion_source"),
            updated_at=now,
        )

    def abandon(self, *, source: str) -> "PlanState":
        if self.status in {"completed", "rejected", "abandoned"}:
            return self
        now = _utc_now_iso()
        return replace(
            self,
            status="abandoned",
            completed_at=now,
            completion_source=_required_text(source, "completion_source"),
            updated_at=now,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "owner_run_id": self.owner_run_id,
            "status": self.status,
            "approval_state": self.approval_state,
            "origin_mode": self.origin_mode,
            "objective": self.objective,
            "summary": self.summary,
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
            approval_state=self.approval_state,
            origin_mode=self.origin_mode,
            objective=self.objective,
            summary=self.summary,
            items=[item.to_dict() for item in self.items],
            revision=self.revision,
            explanation=self.explanation,
            created_at=self.created_at,
            updated_at=self.updated_at,
            completed_at=self.completed_at,
            completion_source=self.completion_source,
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
    run_mode = ensure_run_mode(mode)
    update = PlanUpdate.from_mapping(raw_update, proposal=run_mode == "plan")
    current = state
    if current is not None and (
        current.owner_run_id != run_id
        or (
            run_mode == "plan"
            and current.status in {"active", "completed", "rejected", "abandoned"}
        )
    ):
        current = None
    current = current or PlanState.new(objective=objective, origin_mode=run_mode, run_id=run_id)
    current = current.apply_update(update, mode=run_mode, run_id=run_id)
    closeout = ensure_plan_closeout_status(metadata.get("plan_status"))
    if closeout == "completed":
        if run_mode == "plan":
            raise PlanValidationError("plan mode cannot complete a plan")
        return current.complete(source="model_closeout")
    return current


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


def ensure_plan_closeout_status(value: object) -> PlanCloseoutStatus | None:
    if value is None:
        return None
    text = str(value).strip()
    if text not in _PLAN_CLOSEOUT_STATUSES:
        raise PlanValidationError(f"Unknown plan closeout status: {value}")
    return cast(PlanCloseoutStatus, text)


def _items_from_update(
    current_items: tuple[PlanItem, ...],
    update: PlanUpdate,
) -> list[PlanItem]:
    ids_by_step = {item.step: item.id for item in current_items}
    items: list[PlanItem] = []
    for index, item in enumerate(update.items):
        item_id = ids_by_step.get(item.step) or f"item_{index + 1}"
        items.append(
            PlanItem(
                id=item_id,
                step=item.step,
                details=item.details,
                verification=item.verification,
                status=item.status,
            )
        )
    return items


def _ensure_schema_version(value: object) -> int:
    if value != PLAN_STATE_SCHEMA_VERSION:
        raise PlanValidationError("unsupported plan state schema")
    return PLAN_STATE_SCHEMA_VERSION


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
    "PlanCloseoutStatus",
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
    "ensure_plan_closeout_status",
    "ensure_plan_item_status",
    "ensure_plan_status",
    "ensure_planning_budget_profile",
    "ensure_run_mode",
    "load_plan_state",
    "plan_state_to_dict",
]
