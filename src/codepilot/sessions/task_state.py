from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4


TASK_STATE_SCHEMA_VERSION = 2

TASK_STATE_KEYS = frozenset(
    {
        "schema_version",
        "task_id",
        "raw_user_request",
        "current_mode",
        "approval_state",
        "goal",
        "user_constraints",
        "proposed_plan",
        "approved_plan",
        "current_step_id",
        "steps",
        "verification_status",
        "evidence_refs",
        "blocked_reason",
        "recovery_summary",
        "source_run_id",
        "created_at",
        "updated_at",
    }
)
LEGACY_TASK_STATE_KEYS = frozenset(
    {
        "task_" + "progress",
        "task_" + "mode",
        "task_" + "recovery",
        "recovery_" + "projection",
    }
)
TASK_MODES = frozenset({"read", "plan", "build"})
APPROVAL_STATES = frozenset(
    {"none", "proposed", "approved", "rejected", "revision_needed"}
)
VERIFICATION_STATUSES = frozenset(
    {"unknown", "passed", "failed", "stale", "not_required"}
)
STEP_KINDS = frozenset({"read", "plan", "edit", "verify", "summarize", "other"})
STEP_STATUSES = frozenset({"pending", "in_progress", "completed", "blocked"})
STEP_KEYS = frozenset(
    {
        "id",
        "title",
        "kind",
        "status",
        "acceptance",
        "verification_hint",
        "summary",
        "evidence_refs",
        "failure_count",
    }
)


class TaskStateValidationError(ValueError):
    """Raised when task_state.json is not the canonical schema."""


class TaskStateStore:
    """Session-owned canonical task_state.json store."""

    def __init__(self, session_store: Any) -> None:
        self.session_store = session_store

    def load(self) -> dict[str, Any] | None:
        return self.session_store.load_task_state()

    def current(self) -> dict[str, Any] | None:
        return self.load()

    def save(self, state: Mapping[str, Any]) -> dict[str, Any]:
        payload = validate_task_state_payload(state)
        self.session_store.save_task_state(payload)
        return payload

    def begin(
        self,
        raw_user_request: str,
        *,
        current_mode: str = "build",
        run_id: str | None = None,
    ) -> dict[str, Any]:
        state = new_task_state(
            raw_user_request,
            current_mode=current_mode,
            run_id=run_id,
        )
        return self.save(state)

    def apply_event(self, event: Mapping[str, Any]) -> dict[str, Any] | None:
        state = self.load()
        event_type = str(event.get("type") or "")
        if event_type == "user_request_started":
            text = _required_text(event.get("raw_user_request"), "raw_user_request")
            return self.begin(
                text,
                current_mode=str(event.get("current_mode") or "build"),
                run_id=_optional_text(event.get("run_id")),
            )
        if state is None:
            return None
        next_state = deepcopy(state)
        if event_type == "mode_changed":
            next_state["current_mode"] = _task_mode(event.get("current_mode"))
        elif event_type == "context_compacted":
            next_state["recovery_summary"] = _optional_text(
                event.get("recovery_summary")
            ) or ""
        elif event_type == "planner_proposed_plan":
            next_state["proposed_plan"] = _copy_plan(event.get("proposed_plan"))
            next_state["approval_state"] = "proposed"
            next_state["current_mode"] = "plan"
        elif event_type == "user_approved_plan":
            proposed = _copy_plan(next_state.get("proposed_plan"))
            if proposed is None:
                raise TaskStateValidationError("cannot approve without proposed_plan")
            next_state["approved_plan"] = proposed
            next_state["proposed_plan"] = None
            next_state["approval_state"] = "approved"
            next_state["current_mode"] = "build"
            next_state["current_step_id"] = _first_open_step_id(next_state["steps"])
        else:
            return state
        next_state["updated_at"] = _utc_now_iso()
        return self.save(next_state)


def new_task_state(
    raw_user_request: str,
    *,
    current_mode: str = "build",
    run_id: str | None = None,
) -> dict[str, Any]:
    now = _utc_now_iso()
    request = _required_text(raw_user_request, "raw_user_request")
    return {
        "schema_version": TASK_STATE_SCHEMA_VERSION,
        "task_id": f"task_{uuid4().hex[:12]}",
        "raw_user_request": request,
        "current_mode": _task_mode(current_mode),
        "approval_state": "none",
        "goal": {
            "value": request,
            "source": "user",
            "confidence": "explicit",
        },
        "user_constraints": [],
        "proposed_plan": None,
        "approved_plan": None,
        "current_step_id": None,
        "steps": [],
        "verification_status": "unknown",
        "evidence_refs": [],
        "blocked_reason": None,
        "recovery_summary": "",
        "source_run_id": run_id,
        "created_at": now,
        "updated_at": now,
    }


def validate_task_state_payload(raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TaskStateValidationError("task state must be a JSON object")
    keys = set(raw)
    legacy = sorted(keys & LEGACY_TASK_STATE_KEYS)
    if legacy:
        raise TaskStateValidationError(
            "legacy task state fields are not supported: " + ", ".join(legacy)
        )
    unknown = sorted(keys - TASK_STATE_KEYS)
    if unknown:
        raise TaskStateValidationError(
            "unknown task state fields: " + ", ".join(unknown)
        )
    missing = sorted(TASK_STATE_KEYS - keys)
    if missing:
        raise TaskStateValidationError(
            "missing task state fields: " + ", ".join(missing)
        )
    if raw.get("schema_version") != TASK_STATE_SCHEMA_VERSION:
        raise TaskStateValidationError("legacy task state schema is not supported")
    state = deepcopy(dict(raw))
    state["current_mode"] = _task_mode(state.get("current_mode"))
    state["approval_state"] = _enum(
        state.get("approval_state"),
        APPROVAL_STATES,
        "approval_state",
    )
    state["verification_status"] = _enum(
        state.get("verification_status"),
        VERIFICATION_STATUSES,
        "verification_status",
    )
    _validate_goal(state.get("goal"))
    _validate_constraints(state.get("user_constraints"))
    _validate_plan_or_none(state.get("proposed_plan"), "proposed_plan")
    _validate_plan_or_none(state.get("approved_plan"), "approved_plan")
    state["steps"] = _validate_steps(state.get("steps"))
    state["evidence_refs"] = _string_list(state.get("evidence_refs"), "evidence_refs")
    state["raw_user_request"] = _required_text(
        state.get("raw_user_request"),
        "raw_user_request",
    )
    state["task_id"] = _required_text(state.get("task_id"), "task_id")
    state["recovery_summary"] = _optional_text(state.get("recovery_summary")) or ""
    state["blocked_reason"] = _optional_text(state.get("blocked_reason"))
    state["source_run_id"] = _optional_text(state.get("source_run_id"))
    state["created_at"] = _required_text(state.get("created_at"), "created_at")
    state["updated_at"] = _required_text(state.get("updated_at"), "updated_at")
    step_ids = {step["id"] for step in state["steps"]}
    current_step_id = _optional_text(state.get("current_step_id"))
    if current_step_id is not None and current_step_id not in step_ids:
        raise TaskStateValidationError("current_step_id must reference an existing step")
    state["current_step_id"] = current_step_id
    _validate_plan_step_refs(state.get("approved_plan"), step_ids, "approved_plan")
    _validate_plan_step_refs(state.get("proposed_plan"), step_ids, "proposed_plan")
    if (
        state["current_mode"] == "build"
        and state["approval_state"] == "proposed"
        and state.get("proposed_plan") is not None
    ):
        raise TaskStateValidationError(
            "build mode cannot execute an unapproved proposed_plan"
        )
    return state


def _validate_steps(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TaskStateValidationError("steps must be a list")
    steps: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_step in enumerate(value):
        if not isinstance(raw_step, Mapping):
            raise TaskStateValidationError(f"steps[{index}] must be an object")
        keys = set(raw_step)
        unknown = sorted(keys - STEP_KEYS)
        if unknown:
            raise TaskStateValidationError(
                f"steps[{index}] has unknown fields: " + ", ".join(unknown)
            )
        missing = sorted(STEP_KEYS - keys)
        if missing:
            raise TaskStateValidationError(
                f"steps[{index}] missing fields: " + ", ".join(missing)
            )
        step = deepcopy(dict(raw_step))
        step["id"] = _required_text(step.get("id"), f"steps[{index}].id")
        if step["id"] in seen:
            raise TaskStateValidationError(f"duplicate step id: {step['id']}")
        seen.add(step["id"])
        step["title"] = _required_text(step.get("title"), f"steps[{index}].title")
        step["kind"] = _enum(step.get("kind"), STEP_KINDS, f"steps[{index}].kind")
        step["status"] = _enum(
            step.get("status"),
            STEP_STATUSES,
            f"steps[{index}].status",
        )
        step["acceptance"] = _optional_text(step.get("acceptance"))
        step["verification_hint"] = _optional_text(step.get("verification_hint"))
        step["summary"] = _optional_text(step.get("summary"))
        step["evidence_refs"] = _string_list(
            step.get("evidence_refs"),
            f"steps[{index}].evidence_refs",
        )
        if (
            not isinstance(step.get("failure_count"), int)
            or isinstance(step.get("failure_count"), bool)
            or step["failure_count"] < 0
        ):
            raise TaskStateValidationError(
                f"steps[{index}].failure_count must be a non-negative integer"
            )
        steps.append(step)
    return steps


def _validate_goal(value: object) -> None:
    if not isinstance(value, Mapping):
        raise TaskStateValidationError("goal must be an object")
    _required_text(value.get("value"), "goal.value")
    _required_text(value.get("source"), "goal.source")
    _required_text(value.get("confidence"), "goal.confidence")


def _validate_constraints(value: object) -> None:
    if not isinstance(value, list):
        raise TaskStateValidationError("user_constraints must be a list")
    for index, item in enumerate(value):
        if isinstance(item, str):
            continue
        if not isinstance(item, Mapping):
            raise TaskStateValidationError(
                f"user_constraints[{index}] must be a string or object"
            )
        _required_text(item.get("content"), f"user_constraints[{index}].content")


def _validate_plan_or_none(value: object, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise TaskStateValidationError(f"{field_name} must be an object or null")
    steps = value.get("steps")
    if steps is not None:
        if field_name == "proposed_plan":
            _validate_proposed_plan_steps(steps, field_name)
        else:
            _string_list(steps, f"{field_name}.steps")


def _validate_plan_step_refs(
    plan: object,
    step_ids: set[str],
    field_name: str,
) -> None:
    if plan is None or not isinstance(plan, Mapping):
        return
    steps = plan.get("steps")
    if not isinstance(steps, list):
        return
    if any(not isinstance(step_id, str) for step_id in steps):
        return
    missing = [step_id for step_id in steps if step_id not in step_ids]
    if missing and step_ids:
        raise TaskStateValidationError(
            f"{field_name}.steps references unknown steps: " + ", ".join(missing)
        )


def _validate_proposed_plan_steps(value: object, field_name: str) -> None:
    if not isinstance(value, list):
        raise TaskStateValidationError(f"{field_name}.steps must be a list")
    for index, item in enumerate(value):
        if isinstance(item, str) and item.strip():
            continue
        if not isinstance(item, Mapping):
            raise TaskStateValidationError(
                f"{field_name}.steps[{index}] must be a string or object"
            )
        _required_text(item.get("title") or item.get("id"), f"{field_name}.steps[{index}]")


def _copy_plan(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TaskStateValidationError("plan must be an object")
    return deepcopy(dict(value))


def _first_open_step_id(steps: object) -> str | None:
    if not isinstance(steps, list):
        return None
    for step in steps:
        if isinstance(step, Mapping) and step.get("status") in {"pending", "in_progress"}:
            return _optional_text(step.get("id"))
    return None


def _task_mode(value: object) -> str:
    return _enum(value, TASK_MODES, "current_mode")


def _enum(value: object, allowed: frozenset[str], field_name: str) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if text not in allowed:
        raise TaskStateValidationError(f"{field_name} must be one of {sorted(allowed)}")
    return text


def _string_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise TaskStateValidationError(f"{field_name} must be a list")
    result: list[str] = []
    for index, item in enumerate(value):
        text = _required_text(item, f"{field_name}[{index}]")
        result.append(text)
    return result


def _required_text(value: object, field_name: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise TaskStateValidationError(f"{field_name} is required")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    return text or None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "TASK_STATE_SCHEMA_VERSION",
    "TaskStateStore",
    "TaskStateValidationError",
    "new_task_state",
    "validate_task_state_payload",
]
