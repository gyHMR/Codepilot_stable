from __future__ import annotations

import json
from pathlib import Path

from codepilot.observability import EventRecorder, build_run_summary, build_run_trace
from codepilot.observability.events import validate_run_event


def test_event_recorder_writes_slim_canonical_events(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    recorder = EventRecorder(path)

    recorder.append(
        {
            "type": "context_prepared",
            "runId": "run-1",
            "sessionId": "session-1",
            "turnId": 1,
            "eventId": "legacy-1",
            "timestamp": 100,
            "report": {
                "context_id": "ctx-1",
                "context_mode": "repair",
                "total_budget_tokens": 1200,
                "estimated_tokens_before": 2000,
                "estimated_tokens_after": 900,
                "selected_items": [
                    {
                        "id": "file:src/app.py",
                        "kind": "file",
                        "path": "src/app.py",
                        "source": "src/app.py",
                        "estimated_tokens": 120,
                        "freshness": "fresh",
                        "trust": "current",
                        "content": "must not be persisted",
                    }
                ],
                "stale_items": ["docs/legacy.md"],
                "dropped_items": [{"reason": "budget"}],
                "retrieved_memory_ids": ["mem-1"],
                "tokens_by_layer": {"evidence": 120},
                "context_view": {"large": "debug payload"},
                "repository_delta": {"unused": True},
            },
        }
    )

    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

    assert record["type"] == "context_built"
    assert validate_run_event(record) == []
    assert record["event_id"] == "legacy-1"
    assert record["run_id"] == "run-1"
    assert record["turn"] == 1
    assert record["mode"] == "repair"
    assert record["tokens_after"] == 900
    assert record["selected_items"] == [
        {
            "id": "file:src/app.py",
            "kind": "file",
            "path": "src/app.py",
            "source": "src/app.py",
            "tokens": 120,
            "freshness": "fresh",
            "reason": "task_related",
        }
    ]
    assert "context_view" not in record
    assert "repository_delta" not in record
    assert "content" not in json.dumps(record, ensure_ascii=False)


def test_event_recorder_redacts_secrets_and_skips_low_value_events(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    recorder = EventRecorder(path)

    skipped = recorder.append(
        {
            "type": "turn_start",
            "runId": "run-1",
            "sessionId": "session-1",
            "turnId": 1,
            "eventId": "legacy-turn",
            "timestamp": 100,
        }
    )
    written = recorder.append(
        {
            "type": "error",
            "runId": "run-1",
            "sessionId": "session-1",
            "turnId": 1,
            "eventId": "legacy-error",
            "timestamp": 101,
            "message": "api_key=abc123",
            "api_key": "abc123",
        }
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert skipped == {}
    assert len(lines) == 1
    assert written["type"] == "error"
    assert written["api_key"] == "<redacted>"
    assert "abc123" not in lines[0]


def test_run_trace_and_summary_are_built_from_canonical_events() -> None:
    events = [
        {
            "schema_version": 1,
            "event_id": "e1",
            "run_id": "run-1",
            "session_id": "session-1",
            "turn": 1,
            "type": "run_started",
            "timestamp_ms": 100,
        },
        {
            "schema_version": 1,
            "event_id": "e2",
            "run_id": "run-1",
            "session_id": "session-1",
            "turn": 1,
            "type": "model_call_finished",
            "timestamp_ms": 200,
            "provider": "test",
            "model": "mock",
            "stop_reason": "tool_calls",
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "total_cost": 0.01,
        },
        {
            "schema_version": 1,
            "event_id": "e3",
            "run_id": "run-1",
            "session_id": "session-1",
            "turn": 1,
            "type": "tool_call_finished",
            "timestamp_ms": 250,
            "tool_call_id": "tool-1",
            "tool_name": "read",
            "status": "success",
            "is_error": False,
            "affected_paths": ["src/app.py"],
            "workspace_changed": False,
        },
        {
            "schema_version": 1,
            "event_id": "e4",
            "run_id": "run-1",
            "session_id": "session-1",
            "turn": 1,
            "type": "run_finished",
            "timestamp_ms": 300,
            "status": "completed",
            "stop_reason": "done",
        },
    ]

    trace = build_run_trace(events, result={"workspace_changed": False})
    summary = build_run_summary(trace)

    assert trace.run_id == "run-1"
    assert trace.model_calls[0].total_tokens == 15
    assert trace.tool_calls[0].tool_name == "read"
    assert summary.run_id == "run-1"
    assert summary.status == "completed"
    assert summary.tool_calls == 1
    assert summary.total_tokens == 15
    assert summary.total_cost == 0.01
