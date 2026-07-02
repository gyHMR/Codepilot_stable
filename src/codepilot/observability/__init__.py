from __future__ import annotations

"""Passive observability primitives for Codepilot runs."""

from .events import (
    RUN_EVENT_TYPES,
    event_to_record,
    normalize_event_value,
    summarize_events,
    validate_run_event,
)
from .recorder import EventRecorder
from .redact import redact_artifact
from .summary import RunSummary, build_run_report, build_run_summary
from .trace import (
    AuditBundle,
    ContextTrace,
    MemoryTrace,
    ModelCallTrace,
    RunTrace,
    TaskTrace,
    ToolCallTrace,
    build_run_trace,
    load_audit_bundle,
    load_run_trace,
    write_run_trace,
)

__all__ = [
    "AuditBundle",
    "ContextTrace",
    "EventRecorder",
    "MemoryTrace",
    "ModelCallTrace",
    "RUN_EVENT_TYPES",
    "RunSummary",
    "RunTrace",
    "TaskTrace",
    "ToolCallTrace",
    "build_run_report",
    "build_run_summary",
    "build_run_trace",
    "event_to_record",
    "load_audit_bundle",
    "load_run_trace",
    "normalize_event_value",
    "redact_artifact",
    "summarize_events",
    "validate_run_event",
    "write_run_trace",
]
