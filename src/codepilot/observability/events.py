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
    "tool_call_started",
    "tool_call_finished",
    "task_plan_created",
    "task_step_updated",
    "task_decision_made",
    "completion_checked",
    "file_changed",
    "error",
}

_LOW_VALUE_EVENTS = {
    "turn_start",
    "turn_end",
    "message_update",
    "tool_execution_update",
    "tool_execution_grace",
    "task_recovery_updated",
    "task_recovery_warning",
    "planning_discovery_started",
    "planning_discovery_step",
    "planning_discovery_completed",
    "planning_synthesis_started",
    "planning_synthesis_completed",
    "tool_approval_required",
    "tool_approval_resolved",
    "tool_approval_decision",
    "tool_approval_result_replaced",
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

    Low-value legacy events return ``{}`` so recorders can skip them without
    interrupting the running agent.
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
    if event_type in {"context_prepared", "context_built"}:
        return _context_built(raw)
    if event_type == "memory_retrieved":
        return _memory_retrieved(raw)
    if event_type in {
        "memory_written",
        "memory_updated",
        "memory_created",
        "memory_promoted",
    }:
        return _memory_written(raw, event_type)
    if event_type in {"tool_call_started", "tool_execution_start"}:
        return {
            **_base(raw, "tool_call_started"),
            "tool_call_id": str(raw.get("toolCallId") or raw.get("tool_call_id") or ""),
            "tool_name": str(raw.get("toolName") or raw.get("tool_name") or ""),
            "args": _slim_args(_dict(raw.get("args"))),
        }
    if event_type in {"tool_call_finished", "tool_execution_end"}:
        return _tool_finished(raw)
    if event_type in {"task_decision", "task_decision_made"}:
        return _task_decision(raw)
    if event_type in {"task_plan_created", "task_step_updated", "completion_checked"}:
        return _task_event(raw, event_type)
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
                if key not in _COMMON_LEGACY_FIELDS
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


_COMMON_LEGACY_FIELDS = {
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
                *(_COMMON_LEGACY_FIELDS),
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
    if not isinstance(affected, list):
        affected = raw.get("affected_paths")
    if not isinstance(affected, list):
        affected = result.get("affected_paths")
    is_error = bool(
        raw.get("isError", raw.get("is_error", result.get("is_error", False)))
    )
    status = str(raw.get("status") or result.get("status") or "")
    if not status:
        status = "error" if is_error else "success"
    return {
        **_base(raw, "tool_call_finished"),
        "tool_call_id": str(
            raw.get("toolCallId")
            or raw.get("tool_call_id")
            or result.get("tool_call_id")
            or ""
        ),
        "tool_name": str(
            raw.get("toolName")
            or raw.get("tool_name")
            or result.get("tool_name")
            or ""
        ),
        "status": status,
        "is_error": is_error,
        "error_reason": raw.get("errorReason")
        or result.get("error_code")
        or result.get("error_reason"),
        "approved": bool(raw.get("approved", result.get("approved", True))),
        "permission": permission,
        "duration_ms": _optional_int(raw.get("durationMs", raw.get("duration_ms"))),
        "affected_paths": [str(path) for path in affected or [] if isinstance(path, str)],
        "workspace_changed": _optional_bool(
            raw.get("workspaceChanged", result.get("workspace_changed"))
        ),
        "verification_status": str(verification.get("status") or "none"),
        "output_truncated": bool(raw.get("outputTruncated", raw.get("output_truncated", False))),
    }


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
    return {
        **_base(raw, "memory_written"),
        "memory_ids": [
            str(item)
            for item in _list(raw.get("memory_ids") or raw.get("memoryIds"))
        ],
        "action": str(raw.get("action") or action),
    }


def _task_decision(raw: dict[str, Any]) -> dict[str, Any]:
    decision = _dict(raw.get("decision"))
    return {
        **_base(raw, "task_decision_made"),
        "task_id": _task_field(raw, "task_id"),
        "mode": _task_field(raw, "mode"),
        "phase": _task_field(raw, "phase"),
        "decision": str(
            decision.get("action") or raw.get("decision") or raw.get("action") or ""
        ),
        "reason": str(decision.get("reason") or raw.get("reason") or ""),
    }


def _task_event(raw: dict[str, Any], legacy_type: str) -> dict[str, Any]:
    task = _dict(raw.get("task"))
    step = _dict(raw.get("step")) or _current_step(task)
    completion = _dict(raw.get("completion"))
    if legacy_type == "completion_checked":
        event_type = "completion_checked"
    else:
        event_type = legacy_type
    return {
        **_base(raw, event_type),
        "task_id": str(task.get("task_id") or raw.get("task_id") or ""),
        "mode": str(task.get("mode") or raw.get("mode") or ""),
        "phase": str(task.get("phase") or raw.get("phase") or ""),
        "step_id": str(raw.get("step_id") or step.get("id", "")),
        "step_title": str(raw.get("step_title") or step.get("title", "")),
        "step_status": str(raw.get("step_status") or step.get("status", "")),
        "evidence_refs": [
            str(item)
            for item in _list(raw.get("evidence_refs") or task.get("evidence_refs"))
        ],
        "completion_satisfied": raw.get(
            "completion_satisfied",
            completion.get("satisfied", task.get("completion_satisfied")),
        ),
        "completion_reason": raw.get(
            "completion_reason",
            completion.get("reason", task.get("completion_reason")),
        ),
    }


def _current_step(task: dict[str, Any]) -> dict[str, Any]:
    for step in _list_of_dicts(task.get("steps")):
        if step.get("status") in {"in_progress", "pending"}:
            return step
    steps = _list_of_dicts(task.get("steps"))
    return steps[-1] if steps else {}


def _task_field(raw: dict[str, Any], field: str) -> str:
    task = _dict(raw.get("task"))
    return str(task.get(field) or raw.get(field) or "")


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
