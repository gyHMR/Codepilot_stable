from __future__ import annotations

"""Eval-ready summaries derived from normalized RuntimeEvent logs."""

from dataclasses import dataclass, field
from typing import Any

from .events import summarize_events


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
