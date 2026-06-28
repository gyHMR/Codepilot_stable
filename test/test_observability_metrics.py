from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_run_metrics_and_report_capture_calls_without_driving_the_run() -> None:
    from codepilot.observability import (
        build_model_call_records,
        build_run_metrics,
        build_run_report,
        build_tool_call_records,
    )
    from codepilot.protocols import (
        AgentRunCounters,
        AgentRunResult,
        AssistantMessage,
        Cost,
        RunVerification,
        TextContent,
        ToolResultMessage,
        Usage,
    )

    planning_message = AssistantMessage(
        api="openai-compatible",
        provider="openai",
        model="gpt-demo",
        stop_reason="toolUse",
        usage=Usage(
            input=20,
            output=10,
            total_tokens=30,
            cost=Cost(input=0.5, output=0.5, total=1.0),
        ),
        timestamp=160,
    )
    final_message = AssistantMessage(
        api="openai-compatible",
        provider="openai",
        model="gpt-demo",
        content=[TextContent(text="done")],
        stop_reason="stop",
        usage=Usage(
            input=7,
            output=5,
            total_tokens=12,
            cost=Cost(input=1.0, output=1.0, total=2.0),
        ),
        timestamp=245,
    )
    result = AgentRunResult(
        run_id="run_observe",
        session_id="session_observe",
        status="completed",
        stop_reason="final_answer",
        counters=AgentRunCounters(model_attempts=3, tool_iterations=1, tool_calls=2),
        messages=[
            planning_message,
            ToolResultMessage(
                tool_call_id="tool_1",
                tool_name="edit",
                status="success",
                affected_paths=["src/a.py"],
                workspace_changed=True,
            ),
            final_message,
        ],
        final_message=final_message,
        affected_paths=["src/a.py"],
        workspace_changed=True,
        verification=[
            RunVerification(
                tool_call_id="tool_1",
                tool_name="pytest",
                status="passed",
                command="python -m pytest test -q",
                exit_code=0,
                summary="ok",
            )
        ],
    )
    events = [
        {"type": "agent_start", "timestamp": 100},
        {"type": "message_start", "timestamp": 110, "message": {"role": "assistant"}},
        {"type": "message_end", "timestamp": 160, "message": planning_message},
        {
            "type": "tool_execution_end",
            "timestamp": 190,
            "toolCallId": "tool_1",
            "toolName": "edit",
            "status": "error",
            "isError": True,
            "approved": False,
            "approvalId": "approval_1",
            "errorReason": "command_failed",
            "durationMs": 25,
            "affectedPaths": ["src/a.py"],
            "workspaceChanged": True,
            "outputTruncated": True,
            "result": {
                "tool_call_id": "tool_1",
                "tool_name": "edit",
                "status": "error",
                "is_error": True,
                "verification": {"status": "failed"},
            },
        },
        {"type": "message_start", "timestamp": 200, "message": {"role": "assistant"}},
        {"type": "message_end", "timestamp": 245, "message": final_message},
        {"type": "agent_end", "timestamp": 260},
    ]

    model_calls = build_model_call_records(result, events=events)
    tool_calls = build_tool_call_records(result, events=events)
    metrics = build_run_metrics(result, events=events)
    report = build_run_report(result, events=events, task="observe a run")

    assert [item.latency_ms for item in model_calls] == [50, 45]
    assert model_calls[0].provider == "openai"
    assert model_calls[0].token_usage["total_tokens"] == 30
    assert model_calls[1].cost["total"] == 2.0

    assert len(tool_calls) == 1
    assert tool_calls[0].tool_call_id == "tool_1"
    assert tool_calls[0].status == "error"
    assert tool_calls[0].is_error
    assert not tool_calls[0].approved
    assert tool_calls[0].duration_ms == 25
    assert tool_calls[0].affected_paths == ["src/a.py"]
    assert tool_calls[0].workspace_changed is True
    assert tool_calls[0].verification_status == "failed"
    assert tool_calls[0].output_truncated

    assert metrics.duration_ms == 160
    assert metrics.model_attempts == 3
    assert metrics.model_calls == 2
    assert metrics.tool_calls == 2
    assert metrics.tool_errors == 1
    assert metrics.total_tokens == 42
    assert metrics.total_cost == 3.0
    assert metrics.verification_count == 1
    assert metrics.verification_passed == 1

    assert report["task"] == "observe a run"
    assert report["summary"]["tool_calls"] == 2
    assert report["metrics"]["total_tokens"] == 42
    assert report["model_calls"][0]["latency_ms"] == 50
    assert report["tool_calls"][0]["error_reason"] == "command_failed"
    assert report["event_counts"]["message_end"] == 2


def test_audit_report_counts_nested_task_decisions_and_completion_reasons() -> None:
    from codepilot.observability.audit import build_audit_report

    report = build_audit_report(
        {
            "run_id": "run_task",
            "session_id": "session_task",
            "status": "waiting_user",
            "stop_reason": "task_incomplete",
            "counters": {},
            "messages": [],
            "task": {
                "completed_steps": [],
                "pending_steps": ["修复失败测试"],
                "blocked_steps": [],
                "completion_satisfied": False,
                "completion_reason": "incomplete_steps",
            },
        },
        events=[
            {
                "type": "task_decision",
                "decision": {
                    "action": "repair",
                    "reason": "verification_failed",
                },
            },
            {
                "type": "task_decision",
                "action": "replan",
                "reason": "repeated_step_failure",
            },
            {
                "type": "completion_checked",
                "completion": {
                    "satisfied": False,
                    "reason": "incomplete_steps",
                },
            },
        ],
    )

    assert report["task"]["decision_counts"] == {"repair": 1, "replan": 1}
    assert report["task"]["completion_reason_counts"] == {"incomplete_steps": 1}
