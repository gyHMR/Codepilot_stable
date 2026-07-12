from __future__ import annotations

# 新手导读：events.py 把原始 Agent 事件归一化成更适合审计和统计的形态。
# 关注点：它不改变执行，只整理已经发生的事实。

"""Stable, slim run event contract.

The agent may still emit richer internal events.  Persistence normalizes those
events into this contract so observability remains passive and does not affect
agent execution.
"""

from dataclasses import asdict, is_dataclass
from typing import Any


RUN_EVENT_TYPES = {
    "run_started",
    "run_finished",
    "model_call_started",
    "model_call_finished",
    "context_built",
    "memory_retrieved",
    "memory_written",
    "user_memory_requested",
    "user_correction_observed",
    "task_completed_with_reusable_experience",
    "memory_record_created",
    "memory_candidate_created",
    "memory_record_approved",
    "memory_record_edited",
    "memory_record_disabled",
    "memory_record_deleted",
    "memory_record_superseded",
    "tool_call_started",
    "tool_call_finished",
    "plan_proposed",
    "plan_approval_required",
    "plan_approved",
    "plan_rejected",
    "plan_updated",
    "plan_completed",
    "plan_abandoned",
    "plan_state_warning",
    "run_guard_checked",
    "file_changed",
    "error",
}

_LOW_VALUE_EVENTS = {
    "turn_start",
    "turn_end",
    "message_update",
}

_PLAN_EVENTS = {
    "plan_proposed",
    "plan_approval_required",
    "plan_approved",
    "plan_rejected",
    "plan_updated",
    "plan_completed",
    "plan_abandoned",
}


def normalize_event_value(value: Any) -> Any:
    """Return a JSON-serializable representation of arbitrary event values."""

    if is_dataclass(value):
        return normalize_event_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): normalize_event_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [normalize_event_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def event_to_record(event: dict[str, Any]) -> dict[str, Any]:
    """Normalize an internal event into the public run-event contract.

    Low-value internal progress events return ``{}`` so recorders can skip them
    without interrupting the running agent.
    """

    raw = normalize_event_value(event)
    if not isinstance(raw, dict):
        raise TypeError("event must normalize to a dictionary")
    event_type = str(raw.get("type", ""))
    if event_type in _LOW_VALUE_EVENTS:
        return {}
    if event_type == "agent_start":
        return {**_base(raw, "run_started")}
    if event_type == "agent_end":
        return {
            **_base(raw, "run_finished"),
            "status": str(raw.get("status", "")),
            "stop_reason": str(raw.get("stopReason") or raw.get("stop_reason") or ""),
        }
    if event_type == "message_start" and _message_role(raw) == "assistant":
        return {**_base(raw, "model_call_started")}
    if event_type == "message_end" and _message_role(raw) == "assistant":
        return _model_finished(raw)
    if event_type in {"context_projected", "context_built"}:
        return _context_built(raw)
    if event_type == "memory_retrieved":
        return _memory_retrieved(raw)
    if event_type in {
        "memory_written",
    }:
        return _memory_written(raw, event_type)
    if event_type == "tool_started":
        _require_internal_tool_event(raw, finished=False)
        return {
            **_base(raw, "tool_call_started"),
            "tool_call_id": str(raw.get("toolCallId") or ""),
            "tool_name": str(raw.get("toolName") or ""),
            "args": _slim_args(_dict(raw.get("args"))),
        }
    if event_type in {"tool_completed", "tool_failed", "tool_interrupted"}:
        _require_internal_tool_event(raw, finished=True)
        return _tool_finished(raw)
    if event_type in {"tool_call_started", "tool_call_finished"}:
        _require_canonical_tool_record(raw)
        return _canonical_existing(raw)
    if event_type in _PLAN_EVENTS:
        return _plan_event(raw, event_type)
    if event_type == "run_guard_checked":
        return _run_guard_checked(raw)
    if event_type == "file_diff":
        return {
            **_base(raw, "file_changed"),
            "path": str(raw.get("path", "")),
            "status": str(raw.get("status", "")),
        }
    if event_type == "error":
        return {
            **_base(raw, "error"),
            **{
                key: item
                for key, item in raw.items()
                if key not in _COMMON_EVENT_FIELDS
            },
        }
    if event_type in RUN_EVENT_TYPES:
        return _canonical_existing(raw)
    return {}


def validate_run_event(event: dict[str, Any]) -> list[str]:
    """Return validation errors for the public event contract."""

    errors: list[str] = []
    if event.get("type") not in RUN_EVENT_TYPES:
        errors.append(f"unknown event type: {event.get('type')!r}")
    for field in (
        "schema_version",
        "event_id",
        "run_id",
        "session_id",
        "turn",
        "type",
        "timestamp_ms",
    ):
        if field not in event:
            errors.append(f"missing field: {field}")
    return errors


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for event in events:
        event_type = str(event.get("type", "unknown"))
        counts[event_type] = counts.get(event_type, 0) + 1
    return {"total_events": len(events), "event_counts": counts}


_COMMON_EVENT_FIELDS = {
    "schema_version",
    "event_id",
    "eventId",
    "run_id",
    "runId",
    "session_id",
    "sessionId",
    "turn",
    "turnId",
    "type",
    "timestamp",
    "timestamp_ms",
}


def _canonical_existing(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        **_base(raw, str(raw["type"])),
        **{
            key: item
            for key, item in raw.items()
            if key
            not in {
                *(_COMMON_EVENT_FIELDS),
                "schema_version",
            }
        },
    }


def _base(raw: dict[str, Any], event_type: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event_id": str(raw.get("event_id") or raw.get("eventId") or ""),
        "run_id": str(raw.get("run_id") or raw.get("runId") or ""),
        "session_id": raw.get("session_id") or raw.get("sessionId"),
        "turn": _int(raw.get("turn", raw.get("turnId"))),
        "type": event_type,
        "timestamp_ms": _int(raw.get("timestamp_ms", raw.get("timestamp"))),
    }


def _message_role(raw: dict[str, Any]) -> str:
    return str(_dict(raw.get("message")).get("role", ""))


def _model_finished(raw: dict[str, Any]) -> dict[str, Any]:
    message = _dict(raw.get("message"))
    usage = _dict(message.get("usage"))
    cost = _dict(usage.get("cost"))
    return {
        **_base(raw, "model_call_finished"),
        "provider": str(message.get("provider", "")),
        "model": str(message.get("model", "")),
        "api": str(message.get("api", "")),
        "stop_reason": str(message.get("stop_reason", "")),
        "input_tokens": _int(usage.get("input")),
        "output_tokens": _int(usage.get("output")),
        "total_tokens": _int(usage.get("total_tokens")),
        "total_cost": _float(cost.get("total")),
    }


def _context_built(raw: dict[str, Any]) -> dict[str, Any]:
    report = _dict(raw.get("report")) or raw
    return {
        **_base(raw, "context_built"),
        "context_id": str(report.get("context_id", "")),
        "mode": str(report.get("context_mode") or report.get("mode") or "normal"),
        "budget_tokens": _int(report.get("total_budget_tokens") or report.get("budget_tokens")),
        "tokens_before": _int(report.get("estimated_tokens_before") or report.get("tokens_before")),
        "tokens_after": _int(report.get("estimated_tokens_after") or report.get("tokens_after")),
        "selected_items": [_selected_item(item) for item in _list_of_dicts(report.get("selected_items"))],
        "sections": [_context_section(item) for item in _list_of_dicts(report.get("sections"))],
        "stale_items": [str(item) for item in _list(report.get("stale_items"))],
        "dropped_counts": _dropped_counts(report),
        "tokens_by_layer": {
            str(key): _int(value)
            for key, value in _dict(report.get("tokens_by_layer")).items()
        },
        "memory_ids": [
            str(item)
            for item in _list(report.get("retrieved_memory_ids") or report.get("memory_ids"))
        ],
    }


def _selected_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(item.get("id", "")),
        "kind": str(item.get("kind", "file")),
        "path": str(item.get("path", "")),
        "source": str(item.get("source", "")),
        "tokens": _int(item.get("tokens", item.get("estimated_tokens"))),
        "freshness": str(item.get("freshness", "unknown")),
        "reason": _reason(item),
    }


def _context_section(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(item.get("name", "")),
        "budget_tokens": _int(item.get("budget_tokens")),
        "candidate_items": _int(item.get("candidate_items")),
        "selected_items": _int(item.get("selected_items")),
        "estimated_tokens_before": _int(item.get("estimated_tokens_before")),
        "estimated_tokens_after": _int(item.get("estimated_tokens_after")),
        "reduction_policy": str(item.get("reduction_policy", "")),
    }


def _reason(item: dict[str, Any]) -> str:
    reasons = _list(item.get("reason_tags"))
    if reasons:
        return str(reasons[0])
    reason = item.get("reason")
    return str(reason) if reason else "task_related"


def _dropped_counts(report: dict[str, Any]) -> dict[str, int]:
    counts = {
        str(key): _int(value)
        for key, value in _dict(report.get("dropped_counts")).items()
    }
    for item in _list_of_dicts(report.get("dropped_items")):
        reason = str(item.get("reason", "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _tool_finished(raw: dict[str, Any]) -> dict[str, Any]:
    result = _dict(raw.get("result"))
    permission = raw.get("permission")
    if isinstance(permission, dict):
        permission = permission.get("decision") or permission.get("action")
    verification = _dict(result.get("verification"))
    affected = raw.get("affectedPaths")
    is_error = bool(raw.get("isError", False))
    status = str(raw.get("status") or "")
    if not status:
        status = "error" if is_error else "success"
    return {
        **_base(raw, "tool_call_finished"),
        "tool_call_id": str(raw.get("toolCallId") or ""),
        "tool_name": str(raw.get("toolName") or ""),
        "status": status,
        "is_error": is_error,
        "error_reason": raw.get("errorReason"),
        "approved": bool(raw.get("approved", True)),
        "permission": permission,
        "duration_ms": _optional_int(raw.get("durationMs")),
        "affected_paths": [str(path) for path in affected or [] if isinstance(path, str)],
        "workspace_changed": _optional_bool(raw.get("workspaceChanged")),
        "verification_status": str(verification.get("status") or "none"),
        "output_truncated": bool(raw.get("outputTruncated", False)),
    }


def _require_internal_tool_event(raw: dict[str, Any], *, finished: bool) -> None:
    for field_name in ("toolCallId", "toolName"):
        if not isinstance(raw.get(field_name), str) or not raw[field_name]:
            raise ValueError(f"Internal tool event requires {field_name}")
    if finished:
        if "status" not in raw or "isError" not in raw:
            raise ValueError("Finished internal tool event requires status and isError")
    elif not isinstance(raw.get("args"), dict):
        raise ValueError("tool_started event requires args")


def _require_canonical_tool_record(raw: dict[str, Any]) -> None:
    if raw.get("schema_version") != 1:
        raise ValueError("Canonical tool record requires schema_version=1")
    for field_name in ("tool_call_id", "tool_name"):
        if not isinstance(raw.get(field_name), str) or not raw[field_name]:
            raise ValueError(f"Canonical tool record requires {field_name}")


def _memory_retrieved(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        **_base(raw, "memory_retrieved"),
        "memory_ids": [
            str(item)
            for item in _list(raw.get("memory_ids") or raw.get("memoryIds"))
        ],
        "reasons": _dict(raw.get("reasons")),
    }


def _memory_written(raw: dict[str, Any], action: str) -> dict[str, Any]:
    memory_ids = [
        str(item)
        for item in _list(raw.get("memory_ids") or raw.get("memoryIds"))
    ]
    single = raw.get("memoryId") or raw.get("memory_id")
    if single and str(single) not in memory_ids:
        memory_ids.append(str(single))
    return {
        **_base(raw, "memory_written"),
        "memory_ids": memory_ids,
        "action": str(raw.get("action") or action),
        "memory_type": str(raw.get("memoryType") or raw.get("memory_type") or ""),
        "status": str(raw.get("status") or ""),
    }


def _plan_event(raw: dict[str, Any], event_type: str) -> dict[str, Any]:
    plan = _dict(raw.get("plan"))
    return {
        **_base(raw, event_type),
        "plan_id": str(plan.get("plan_id") or raw.get("plan_id") or ""),
        "status": str(plan.get("status") or raw.get("status") or ""),
        "origin_mode": str(plan.get("origin_mode") or raw.get("origin_mode") or ""),
        "raw_user_request": str(plan.get("raw_user_request") or raw.get("raw_user_request") or ""),
        "interpreted_goal": str(plan.get("interpreted_goal") or raw.get("interpreted_goal") or ""),
        "items": _list_of_dicts(plan.get("items")),
    }


def _run_guard_checked(raw: dict[str, Any]) -> dict[str, Any]:
    decision = _dict(raw.get("decision"))
    signals = _dict(raw.get("signals"))
    return {
        **_base(raw, "run_guard_checked"),
        "action": str(decision.get("action") or raw.get("action") or ""),
        "reason": str(decision.get("reason") or raw.get("reason") or ""),
        "verification_status": str(signals.get("verification_status") or ""),
        "workspace_changed": bool(signals.get("workspace_changed", False)),
    }


def _slim_args(args: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in args.items()
        if key in {"path", "file_path", "target", "command", "cmd"}
        and isinstance(value, (str, int, float, bool))
    }


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in _list(value) if isinstance(item, dict)]


def _int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _float(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


__all__ = [
    "RUN_EVENT_TYPES",
    "event_to_record",
    "normalize_event_value",
    "summarize_events",
    "validate_run_event",
]
