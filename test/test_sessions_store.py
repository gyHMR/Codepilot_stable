from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_session_store_persists_messages_forks_and_summarizes_events(tmp_path: Path) -> None:
    from codepilot.core import AgentToolResult
    from codepilot.protocols import AssistantMessage, Cost, TextContent, Usage, UserMessage
    from codepilot.sessions.store import SessionStore

    store = SessionStore(tmp_path, "session_test")
    store.ensure_initialized(model_id="m", provider="p", system_prompt="sys")

    user = UserMessage(content="hello")
    assistant = AssistantMessage(
        content=[TextContent(text="world")],
        usage=Usage(input=3, output=4, total_tokens=7, cost=Cost(total=0.01)),
    )
    store.append_context_message(user)
    store.append_context_message(assistant)
    store.append_event(
        {
            "type": "tool_execution_end",
            "runId": "run_1",
            "turnId": 1,
            "eventId": "run_1:1",
            "timestamp": 1,
            "sessionId": "session_test",
            "toolCallId": "call_1",
            "toolName": "echo",
            "result": AgentToolResult(content=[TextContent(text="ok")], details={"ok": True}),
            "isError": False,
        }
    )
    store.append_event(
        {
            "type": "message_end",
            "runId": "run_1",
            "turnId": 1,
            "eventId": "run_1:2",
            "timestamp": 2,
            "sessionId": "session_test",
            "message": assistant,
        }
    )

    restored = store.load_session_messages()
    assert len(restored) == 2
    restored_assistant = restored[-1]
    assert isinstance(restored_assistant, AssistantMessage)
    assert restored_assistant.usage.total_tokens == 7

    forked = store.fork_to("session_fork")
    assert forked.read_meta()["parent_session_id"] == "session_test"
    assert len(forked.load_session_messages()) == 2

    events = store.load_events()
    assert events[0]["result"]["content"][0]["text"] == "ok"
    summary = store.summarize_events()
    assert summary["total_events"] == 2
    assert summary["tool_errors"] == 0
    assert summary["usage"]["total_tokens"] == 7


def test_event_recorder_builds_eval_summary(tmp_path: Path) -> None:
    from codepilot.observability import EventRecorder, build_eval_summary

    recorder = EventRecorder(tmp_path / "events.jsonl")
    recorder.append(
        {
            "type": "tool_execution_start",
            "runId": "run_1",
            "turnId": 1,
            "eventId": "run_1:1",
            "timestamp": 1,
            "sessionId": "session_1",
            "toolCallId": "tool_1",
            "toolName": "read",
            "args": {},
        }
    )
    recorder.append(
        {
            "type": "tool_execution_end",
            "runId": "run_1",
            "turnId": 1,
            "eventId": "run_1:2",
            "timestamp": 2,
            "sessionId": "session_1",
            "toolCallId": "tool_1",
            "toolName": "read",
            "result": {"content": [], "details": {}},
            "isError": True,
        }
    )

    summary = build_eval_summary(recorder.load())

    assert summary.tool_calls == 1
    assert summary.tool_errors == 1
    assert summary.tool_error_rate == 1.0
    assert summary.has_errors


def test_session_store_persists_run_results(tmp_path: Path) -> None:
    from codepilot.protocols import (
        AgentRunCounters,
        AgentRunResult,
        AssistantMessage,
        TextContent,
        ToolResultMessage,
    )
    from codepilot.sessions.run_store import RunStore
    from codepilot.sessions.store import SessionStore

    store = SessionStore(tmp_path, "session_run")
    store.ensure_initialized(model_id="m", provider="p", system_prompt="")
    tracked = tmp_path / "a.py"
    tracked.write_text("print('old')\n", encoding="utf-8")
    file_state = RunStore.file_state_for_path(tmp_path, "a.py")
    final = AssistantMessage(content=[TextContent(text="done")])
    tool_result = ToolResultMessage(
        tool_call_id="tool_1",
        tool_name="read",
        content=[TextContent(text="print('old')")],
        affected_paths=["a.py"],
        metadata={"file_state": file_state},
    )
    result = AgentRunResult(
        run_id="run_1",
        session_id="session_run",
        status="completed",
        stop_reason="final_answer",
        counters=AgentRunCounters(model_attempts=2, tool_iterations=1, tool_calls=1),
        messages=[tool_result, final],
        final_message=final,
        affected_paths=["a.py"],
        workspace_changed=True,
    )

    store.append_run_result(result)
    loaded = store.load_run_results()

    assert loaded[0]["run_id"] == "run_1"
    assert loaded[0]["counters"]["model_attempts"] == 2
    assert loaded[0]["affected_paths"] == ["a.py"]
    assert (tmp_path / ".codepilot" / "runs" / "run_1" / "result.json").exists()

    run_store = RunStore(tmp_path, "session_run")
    assert run_store.load_run_result("run_1")["run_id"] == "run_1"
    assert run_store.evaluate_freshness().status == "valid"

    tracked.write_text("print('new')\n", encoding="utf-8")
    freshness = run_store.evaluate_freshness()
    assert freshness.status == "stale"
    assert freshness.changed_paths == ["a.py"]


def test_observability_builds_run_summary_and_report_from_run_result() -> None:
    from codepilot.observability import build_run_report, build_run_summary
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

    final = AssistantMessage(
        content=[TextContent(text="fixed the bug")],
        usage=Usage(input=10, output=5, total_tokens=15, cost=Cost(total=0.02)),
    )
    result = AgentRunResult(
        run_id="run_report",
        session_id="session_report",
        status="completed",
        stop_reason="final_answer",
        counters=AgentRunCounters(model_attempts=2, tool_iterations=1, tool_calls=2),
        messages=[
            ToolResultMessage(tool_name="edit", affected_paths=["src/a.py"], workspace_changed=True),
            ToolResultMessage(tool_name="bash", verification={"status": "passed"}),
            final,
        ],
        final_message=final,
        affected_paths=["src/a.py"],
        workspace_changed=True,
        verification=[
            RunVerification(
                tool_call_id="tool_1",
                tool_name="bash",
                status="passed",
                command="python -m pytest test -q",
                exit_code=0,
                summary="passed",
            )
        ],
    )
    events = [
        {"type": "agent_start", "timestamp": 100},
        {"type": "agent_end", "timestamp": 260},
    ]

    summary = build_run_summary(result, events=events)
    report = build_run_report(result, events=events, task="fix test")

    assert summary.status == "completed"
    assert summary.duration_ms == 160
    assert summary.verification_count == 1
    assert summary.verification_passed == 1
    assert summary.token_usage["total_tokens"] == 15
    assert summary.cost["total"] == 0.02
    assert report["task"] == "fix test"
    assert report["summary"]["tool_calls"] == 2
    assert report["verification"][0]["command"] == "python -m pytest test -q"
