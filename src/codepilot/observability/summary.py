from __future__ import annotations

# 新手导读：summary.py 根据事件和指标生成可读运行摘要。
# 关注点：它服务于学习展示和调试复盘。

"""Run summary/report projection built from :mod:`observability.trace`."""

from dataclasses import asdict, dataclass, field
from typing import Any

from .trace import RunTrace, build_run_trace


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    session_id: str | None
    status: str = ""
    stop_reason: str = ""
    model_calls: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    affected_paths: list[str] = field(default_factory=list)
    workspace_changed: bool = False
    duration_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_run_summary(trace: RunTrace) -> RunSummary:
    duration = (
        max(0, trace.finished_at_ms - trace.started_at_ms)
        if trace.started_at_ms is not None
        and trace.finished_at_ms is not None
        else None
    )
    return RunSummary(
        run_id=trace.run_id,
        session_id=trace.session_id,
        status=trace.status,
        stop_reason=trace.stop_reason,
        model_calls=len(trace.model_calls),
        tool_calls=len(trace.tool_calls),
        tool_errors=sum(item.is_error for item in trace.tool_calls),
        input_tokens=sum(item.input_tokens for item in trace.model_calls),
        output_tokens=sum(item.output_tokens for item in trace.model_calls),
        total_tokens=sum(item.total_tokens for item in trace.model_calls),
        total_cost=sum(item.total_cost for item in trace.model_calls),
        affected_paths=list(trace.affected_paths),
        workspace_changed=trace.workspace_changed,
        duration_ms=duration,
    )


def build_run_report(
    result: dict[str, Any],
    *,
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    trace = build_run_trace(events or [], result=result)
    summary = build_run_summary(trace)
    return {
        "schema_version": 1,
        "run_id": trace.run_id,
        "session_id": trace.session_id,
        "summary": summary.to_dict(),
        "model_calls": [asdict(item) for item in trace.model_calls],
        "tool_calls": [asdict(item) for item in trace.tool_calls],
        "contexts": [asdict(item) for item in trace.contexts],
        "tasks": [asdict(item) for item in trace.tasks],
        "memories": [asdict(item) for item in trace.memories],
        "errors": list(trace.errors),
    }


__all__ = ["RunSummary", "build_run_report", "build_run_summary"]
