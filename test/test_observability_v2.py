from __future__ import annotations

import json
from pathlib import Path

from codepilot.evaluation.evidence import evidence_from_traces
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
            "eventId": "raw-1",
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
                "stale_items": ["docs/v1-note.md"],
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
    assert record["event_id"] == "raw-1"
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
            "eventId": "raw-turn",
            "timestamp": 100,
        }
    )
    written = recorder.append(
        {
            "type": "error",
            "runId": "run-1",
            "sessionId": "session-1",
            "turnId": 1,
            "eventId": "raw-error",
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


def test_trace_normalizes_raw_event_shapes_even_when_event_names_are_canonical() -> None:
    events = [
        {
            "type": "memory_retrieved",
            "runId": "run-1",
            "sessionId": "session-1",
            "turnId": 1,
            "eventId": "mem-evt",
            "timestamp": 100,
            "memoryIds": ["mem_api_contract_v2"],
            "reasons": {"mem_api_contract_v2": ["path_match"]},
        },
        {
            "type": "task_plan_created",
            "runId": "run-1",
            "sessionId": "session-1",
            "turnId": 1,
            "eventId": "task-evt",
            "timestamp": 110,
            "task": {
                "task_id": "task-1",
                "mode": "repair",
                "phase": "acting",
                "steps": [
                    {
                        "id": "step-1",
                        "title": "Repair failing pytest",
                        "status": "completed",
                    }
                ],
            },
            "evidence_refs": ["pytest.log"],
        },
    ]

    trace = build_run_trace(events)
    evidence = evidence_from_traces(
        case_id="case-1",
        module="memory",
        traces=[trace],
        expected={"memory_ids": ["mem_api_contract_v2"]},
        task_passed=True,
    )

    assert evidence.memory_ids == ["mem_api_contract_v2"]
    assert evidence.steps[0].step_id == "step-1"
    assert evidence.steps[0].status == "completed"
    assert evidence.steps[0].evidence_refs == ["pytest.log"]


def test_trace_joins_tool_start_args_into_finished_tool_evidence() -> None:
    events = [
        {
            "schema_version": 1,
            "event_id": "tool-start",
            "run_id": "run-1",
            "session_id": "session-1",
            "turn": 1,
            "type": "tool_call_started",
            "timestamp_ms": 100,
            "tool_call_id": "tool-1",
            "tool_name": "read",
            "args": {"path": "docs/api-contract-v1.md"},
        },
        {
            "schema_version": 1,
            "event_id": "tool-end",
            "run_id": "run-1",
            "session_id": "session-1",
            "turn": 1,
            "type": "tool_call_finished",
            "timestamp_ms": 120,
            "tool_call_id": "tool-1",
            "tool_name": "read",
            "status": "success",
            "affected_paths": [],
            "workspace_changed": False,
        },
    ]

    trace = build_run_trace(events)
    evidence = evidence_from_traces(
        case_id="case-1",
        module="memory",
        traces=[trace],
        expected={"failed_attempt_paths": ["docs/api-contract-v1.md"]},
        task_passed=True,
    )

    assert evidence.tools[0].args == {"path": "docs/api-contract-v1.md"}
    assert evidence.tools[0].affected_paths == ["docs/api-contract-v1.md"]


def test_v2_core_tool_events_normalize_to_observability_records() -> None:
    import asyncio

    async def run_case() -> None:
        from codepilot.core.contracts import (
            AgentLoopInput,
            AgentLoopLimits,
            AgentLoopPorts,
            RunCorrelation,
        )
        from codepilot.core.loop import run_agent_loop
        from codepilot.llm.ports import LLMCompleted, ModelDescriptor
        from codepilot.observability import event_to_record, summarize_events
        from codepilot.protocols import AssistantMessage, TextContent, ToolCall, ToolResultMessage
        from codepilot.tools.ports import ToolObservation

        class FakeModel:
            def __init__(self) -> None:
                self.calls = 0

            async def stream(self, request):
                self.calls += 1
                if self.calls == 1:
                    yield LLMCompleted(
                        message=AssistantMessage(
                            content=[
                                ToolCall(
                                    id="call-read",
                                    name="read",
                                    arguments={"path": "README.md"},
                                )
                            ]
                        )
                    )
                    return
                assert isinstance(request.messages[-1], ToolResultMessage)
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text="done")])
                )

        class FakeTools:
            def catalog(self):
                return {"tools": ["read"]}

            async def execute(self, invocation):
                return ToolObservation(
                    tool_call_id=invocation.tool_call_id,
                    name=invocation.name,
                    status="success",
                    content=(TextContent(text="body"),),
                    affected_paths=("README.md",),
                    workspace_changed=False,
                )

        outcome = await run_agent_loop(
            AgentLoopInput(
                run_id="run-v2",
                correlation=RunCorrelation(session_id="session-v2"),
                user_prompt="read file",
                model=ModelDescriptor(provider="fake", model_id="unit"),
                limits=AgentLoopLimits(max_model_turns=3),
            ),
            AgentLoopPorts(model=FakeModel(), tools=FakeTools()),
        )

        records = [record for event in outcome.events if (record := event_to_record(event))]
        summary = summarize_events(records)

        assert summary["event_counts"]["tool_call_started"] == 1
        assert summary["event_counts"]["tool_call_finished"] == 1
        tool_started = next(record for record in records if record["type"] == "tool_call_started")
        tool_finished = next(record for record in records if record["type"] == "tool_call_finished")
        assert tool_started["args"] == {"path": "README.md"}
        assert tool_finished["tool_call_id"] == "call-read"
        assert tool_finished["tool_name"] == "read"
        assert tool_finished["status"] == "success"
        assert tool_finished["approved"] is True

    asyncio.run(run_case())
