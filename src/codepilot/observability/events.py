from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from codepilot.protocols import (
    AssistantMessage,
    Message,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)

AGENT_EVENT_TYPES = {
    "agent_start",
    "turn_start",
    "message_start",
    "message_update",
    "message_end",
    "model_retry_start",
    "tool_execution_start",
    "tool_execution_update",
    "tool_execution_end",
    "task_plan_created",
    "task_step_updated",
    "task_decision",
    "completion_checked",
    "turn_end",
    "agent_end",
    "error",
}

SESSION_EVENT_TYPES = {
    "auto_retry_start",
    "context_compacted",
    "context_freshness_checked",
    "session_forked",
    "session_switch_entry",
}

_COMMON_AGENT_EVENT_FIELDS = {"type", "runId", "turnId", "eventId", "timestamp", "sessionId"}


def normalize_event_value(value: Any) -> Any:
    if isinstance(value, (AssistantMessage, ToolResultMessage, UserMessage)):
        return normalize_event_value(asdict(value))
    if isinstance(value, ToolResult):
        return {
            "tool_call_id": value.tool_call_id,
            "tool_name": value.tool_name,
            "content": [normalize_event_value(block) for block in value.content],
            "details": normalize_event_value(value.details),
            "is_error": value.is_error,
            "status": value.status,
            "approved": value.approved,
            "approval_id": value.approval_id,
            "error_code": value.error_code,
            "exit_code": value.exit_code,
            "affected_paths": list(value.affected_paths),
            "workspace_changed": value.workspace_changed,
            "diff_summary": value.diff_summary,
            "verification": normalize_event_value(value.verification),
        }
    if isinstance(value, dict):
        return {str(k): normalize_event_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_event_value(item) for item in value]
    if is_dataclass(value):
        return normalize_event_value(asdict(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def event_to_record(event: dict[str, Any]) -> dict[str, Any]:
    record = normalize_event_value(event)
    if not isinstance(record, dict):
        raise TypeError("event must normalize to a dictionary")
    record.setdefault("schema_version", 1)
    return record


def validate_agent_event(event: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    event_type = event.get("type")
    if event_type not in AGENT_EVENT_TYPES:
        errors.append(f"unknown event type: {event_type!r}")
        return errors
    missing = sorted(field for field in _COMMON_AGENT_EVENT_FIELDS if field not in event)
    if missing:
        errors.append(f"missing common fields: {', '.join(missing)}")
    return errors


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    run_ids: set[str] = set()
    session_ids: set[str] = set()
    tool_calls = 0
    tool_errors = 0
    errors = 0
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_cost": 0.0,
    }

    for event in events:
        event_type = str(event.get("type", "unknown"))
        counts[event_type] = counts.get(event_type, 0) + 1
        if isinstance(event.get("runId"), str):
            run_ids.add(str(event["runId"]))
        if isinstance(event.get("sessionId"), str):
            session_ids.add(str(event["sessionId"]))
        if event_type == "tool_execution_start":
            tool_calls += 1
        if event_type == "tool_execution_end" and event.get("isError"):
            tool_errors += 1
        if event_type == "error":
            errors += 1

        message = event.get("message")
        if isinstance(message, dict) and message.get("role") == "assistant":
            raw_usage = message.get("usage")
            if isinstance(raw_usage, dict):
                usage["input_tokens"] += int(raw_usage.get("input", 0) or 0)
                usage["output_tokens"] += int(raw_usage.get("output", 0) or 0)
                usage["total_tokens"] += int(raw_usage.get("total_tokens", 0) or 0)
                cost = raw_usage.get("cost")
                if isinstance(cost, dict):
                    usage["total_cost"] += float(cost.get("total", 0.0) or 0.0)

    return {
        "total_events": len(events),
        "event_counts": counts,
        "run_count": len(run_ids),
        "session_count": len(session_ids),
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "errors": errors,
        "usage": usage,
    }
