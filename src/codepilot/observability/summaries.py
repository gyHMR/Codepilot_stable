from __future__ import annotations

"""Eval-ready summaries derived from events and AgentRunResult."""

from dataclasses import dataclass, field
from typing import Any

from codepilot.protocols import AgentRunResult

from .events import normalize_event_value, summarize_events
from .metrics import (
    build_model_call_records,
    build_run_metrics,
    build_tool_call_records,
)


@dataclass(frozen=True)
class EvalRunSummary:
    total_events: int
    run_count: int
    session_count: int
    tool_calls: int
    tool_errors: int
    errors: int
    usage: dict[str, Any] = field(default_factory=dict)
    event_counts: dict[str, int] = field(default_factory=dict)

    @property
    def tool_error_rate(self) -> float:
        if self.tool_calls <= 0:
            return 0.0
        return self.tool_errors / self.tool_calls

    @property
    def has_errors(self) -> bool:
        return self.errors > 0 or self.tool_errors > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_events": self.total_events,
            "run_count": self.run_count,
            "session_count": self.session_count,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "tool_error_rate": self.tool_error_rate,
            "errors": self.errors,
            "has_errors": self.has_errors,
            "usage": self.usage,
            "event_counts": self.event_counts,
        }


def build_eval_summary(events: list[dict[str, Any]]) -> EvalRunSummary:
    raw = summarize_events(events)
    return EvalRunSummary(
        total_events=int(raw.get("total_events", 0)),
        run_count=int(raw.get("run_count", 0)),
        session_count=int(raw.get("session_count", 0)),
        tool_calls=int(raw.get("tool_calls", 0)),
        tool_errors=int(raw.get("tool_errors", 0)),
        errors=int(raw.get("errors", 0)),
        usage=dict(raw.get("usage", {})),
        event_counts=dict(raw.get("event_counts", {})),
    )


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    session_id: str | None
    status: str
    stop_reason: str
    model_attempts: int
    tool_iterations: int
    tool_calls: int
    affected_paths: list[str] = field(default_factory=list)
    workspace_changed: bool = False
    verification_count: int = 0
    verification_passed: int = 0
    approval_count: int = 0
    denied_count: int = 0
    token_usage: dict[str, int] = field(default_factory=dict)
    cost: dict[str, float] = field(default_factory=dict)
    duration_ms: int | None = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "model_attempts": self.model_attempts,
            "tool_iterations": self.tool_iterations,
            "tool_calls": self.tool_calls,
            "affected_paths": list(self.affected_paths),
            "workspace_changed": self.workspace_changed,
            "verification_count": self.verification_count,
            "verification_passed": self.verification_passed,
            "approval_count": self.approval_count,
            "denied_count": self.denied_count,
            "token_usage": dict(self.token_usage),
            "cost": dict(self.cost),
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


def build_run_summary(
    result: AgentRunResult | dict[str, Any],
    *,
    events: list[dict[str, Any]] | None = None,
) -> RunSummary:
    record = _run_record(result)
    metrics = build_run_metrics(record, events=events)
    model_calls = build_model_call_records(record, events=events)
    cost_input = sum(item.cost.get("input", 0.0) for item in model_calls)
    cost_output = sum(item.cost.get("output", 0.0) for item in model_calls)

    return RunSummary(
        run_id=str(record.get("run_id", "")),
        session_id=_optional_str(record.get("session_id")),
        status=str(record.get("status", "")),
        stop_reason=str(record.get("stop_reason", "")),
        model_attempts=metrics.model_attempts,
        tool_iterations=metrics.tool_iterations,
        tool_calls=metrics.tool_calls,
        affected_paths=[str(path) for path in record.get("affected_paths", []) if isinstance(path, str)],
        workspace_changed=bool(record.get("workspace_changed", False)),
        verification_count=metrics.verification_count,
        verification_passed=metrics.verification_passed,
        approval_count=metrics.approval_count,
        denied_count=metrics.denied_count,
        token_usage={
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "total_tokens": metrics.total_tokens,
        },
        cost={
            "input": cost_input,
            "output": cost_output,
            "total": metrics.total_cost,
        },
        duration_ms=metrics.duration_ms,
        error=_dict_or_none(record.get("error")),
    )


def build_run_report(
    result: AgentRunResult | dict[str, Any],
    *,
    events: list[dict[str, Any]] | None = None,
    task: str | None = None,
) -> dict[str, Any]:
    record = _run_record(result)
    summary = build_run_summary(record, events=events)
    metrics = build_run_metrics(record, events=events)
    model_calls = build_model_call_records(record, events=events)
    tool_calls = build_tool_call_records(record, events=events)
    event_counts = summarize_events(events or []).get("event_counts", {})
    verification = _list_of_dicts(record.get("verification"))
    return {
        "run_id": summary.run_id,
        "session_id": summary.session_id,
        "task": task or "",
        "summary": summary.to_dict(),
        "metrics": metrics.to_dict(),
        "model_calls": [item.to_dict() for item in model_calls],
        "tool_calls": [item.to_dict() for item in tool_calls],
        "final_text": _final_text(record),
        "affected_paths": list(summary.affected_paths),
        "verification": verification,
        "error": summary.error,
        "event_count": len(events or []),
        "event_counts": dict(event_counts) if isinstance(event_counts, dict) else {},
    }


def _run_record(result: AgentRunResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    record = normalize_event_value(result)
    return record if isinstance(record, dict) else {}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dict_or_none(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _final_text(record: dict[str, Any]) -> str:
    final_message = _dict(record.get("final_message"))
    content = final_message.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts).strip()
