from __future__ import annotations

from codepilot.protocols import (
    AgentRunCounters,
    AgentRunResult,
    AssistantMessage,
    RunVerification,
    TextContent,
    ToolCall,
    ToolResultMessage,
)
from codepilot.sessions.run_reconciliation import (
    merge_approved_tool_result,
    replace_pending_tool_result,
)


def _approved_tool_result() -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id="tool_1",
        tool_name="custom_mutate",
        content=[TextContent(text="mutated")],
        status="success",
        approved=True,
        approval_id="approval_1",
        affected_paths=["src/a.py", ""],
        workspace_changed=True,
        verification={
            "status": "passed",
            "command": "pytest test/test_a.py",
            "exit_code": 0,
            "summary": "targeted tests passed",
        },
    )


def test_merge_approved_tool_result_records_approval_time_evidence() -> None:
    result = AgentRunResult(
        run_id="run_1",
        session_id="session_1",
        status="completed",
        stop_reason="final_answer",
        counters=AgentRunCounters(model_attempts=1, tool_calls=2),
        affected_paths=["README.md"],
    )

    merged = merge_approved_tool_result(result, _approved_tool_result())

    assert merged is not result
    assert merged.counters.tool_calls == 3
    assert merged.messages[0].tool_call_id == "tool_1"
    assert merged.affected_paths == ["README.md", "src/a.py"]
    assert merged.workspace_changed is True
    assert merged.verification == [
        RunVerification(
            tool_call_id="tool_1",
            tool_name="custom_mutate",
            status="passed",
            command="pytest test/test_a.py",
            exit_code=0,
            summary="targeted tests passed",
        )
    ]


def test_merge_approved_tool_result_is_idempotent_for_same_approval() -> None:
    approved_tool_result = _approved_tool_result()
    result = AgentRunResult(
        run_id="run_1",
        session_id="session_1",
        status="completed",
        stop_reason="final_answer",
        counters=AgentRunCounters(tool_calls=1),
        messages=[approved_tool_result],
        affected_paths=["src/a.py"],
        workspace_changed=True,
        verification=[
            RunVerification(
                tool_call_id="tool_1",
                tool_name="custom_mutate",
                status="passed",
                command="pytest test/test_a.py",
            )
        ],
    )

    merged = merge_approved_tool_result(result, approved_tool_result)

    assert merged.counters.tool_calls == 1
    assert merged.messages == [approved_tool_result]
    assert merged.affected_paths == ["src/a.py"]
    assert len(merged.verification) == 1


def test_replace_pending_tool_result_matches_by_approval_id_first() -> None:
    assistant = AssistantMessage(
        content=[ToolCall(id="tool_1", name="custom_mutate", arguments={})]
    )
    pending = ToolResultMessage(
        tool_call_id="tool_1",
        tool_name="custom_mutate",
        content=[TextContent(text="approval needed")],
        status="approval_required",
        approval_id="approval_1",
        approved=False,
    )
    replacement = ToolResultMessage(
        tool_call_id="tool_1",
        tool_name="custom_mutate",
        content=[TextContent(text="mutated")],
        status="success",
        approval_id="approval_1",
        approved=True,
    )

    updated, replaced = replace_pending_tool_result(
        [assistant, pending],
        replacement,
        approval_id="approval_1",
    )

    assert replaced is True
    assert updated == [assistant, replacement]


def test_replace_pending_tool_result_falls_back_to_pending_call_id() -> None:
    pending = ToolResultMessage(
        tool_call_id="tool_1",
        tool_name="custom_mutate",
        content=[TextContent(text="approval needed")],
        status="approval_required",
        approval_id=None,
        approved=False,
    )
    unrelated_completed = ToolResultMessage(
        tool_call_id="tool_2",
        tool_name="custom_mutate",
        content=[TextContent(text="already done")],
        status="success",
    )
    replacement = ToolResultMessage(
        tool_call_id="tool_1",
        tool_name="custom_mutate",
        content=[TextContent(text="mutated")],
        status="success",
        approved=True,
    )

    updated, replaced = replace_pending_tool_result(
        [unrelated_completed, pending],
        replacement,
    )

    assert replaced is True
    assert updated == [unrelated_completed, replacement]


def test_replace_pending_tool_result_ignores_resolved_approval_result() -> None:
    resolved = ToolResultMessage(
        tool_call_id="tool_1",
        tool_name="custom_mutate",
        content=[TextContent(text="already mutated")],
        status="success",
        approval_id="approval_1",
        approved=True,
    )
    replacement = ToolResultMessage(
        tool_call_id="tool_1",
        tool_name="custom_mutate",
        content=[TextContent(text="new result")],
        status="success",
        approval_id="approval_1",
        approved=True,
    )

    updated, replaced = replace_pending_tool_result(
        [resolved],
        replacement,
        approval_id="approval_1",
    )

    assert replaced is False
    assert updated == [resolved]
