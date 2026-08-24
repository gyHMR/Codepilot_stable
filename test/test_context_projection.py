from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from codepilot.core import transcript
from codepilot.protocols import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from codepilot.sessions.context.projection import ContextProjector
from codepilot.sessions.context.state import ContextState


def test_parallel_tool_batches_are_grouped_and_closed_by_call_id() -> None:
    assert hasattr(transcript, "message_groups")
    assert hasattr(transcript, "tool_batch_call_ids")
    assert hasattr(transcript, "is_closed_tool_batch")
    assistant = AssistantMessage(
        content=[
            ToolCall(id="call_a", name="read", arguments={"path": "a.py"}),
            ToolCall(id="call_b", name="read", arguments={"path": "b.py"}),
        ]
    )
    result_a = ToolResultMessage(tool_call_id="call_a", tool_name="read")
    result_b = ToolResultMessage(tool_call_id="call_b", tool_name="read")

    groups = transcript.message_groups((assistant, result_b, result_a))

    assert groups == ((assistant, result_b, result_a),)
    assert transcript.tool_batch_call_ids(groups[0]) == frozenset({"call_a", "call_b"})
    assert transcript.is_closed_tool_batch(groups[0]) is True
    assert transcript.is_closed_tool_batch((assistant, result_a)) is False


def test_unconsumed_batch_requires_no_later_assistant_response() -> None:
    assert hasattr(transcript, "latest_unconsumed_tool_batch")
    assistant = AssistantMessage(
        content=[ToolCall(id="call_1", name="read", arguments={"path": "a.py"})]
    )
    result = ToolResultMessage(tool_call_id="call_1", tool_name="read")
    user = UserMessage(content="additional instruction")

    assert transcript.latest_unconsumed_tool_batch(
        transcript.message_groups((assistant, result, user))
    ) == 0
    assert transcript.latest_unconsumed_tool_batch(
        transcript.message_groups(
            (
                assistant,
                result,
                AssistantMessage(content=[TextContent(text="consumed")]),
            )
        )
    ) is None


def test_unconsumed_parallel_read_batch_keeps_full_results_without_mutation(
    tmp_path: Path,
) -> None:
    assistant = AssistantMessage(
        content=[
            ToolCall(id="call_a", name="read", arguments={"path": "a.py"}),
            ToolCall(id="call_b", name="read", arguments={"path": "b.py"}),
        ]
    )
    result_a = _read_result("call_a", "a.py", "a" * 5_000, sha256="hash-a")
    result_b = _read_result("call_b", "b.py", "b" * 5_000, sha256="hash-b")
    messages = (UserMessage(content="inspect"), assistant, result_b, result_a)
    original = deepcopy(messages)
    state = _observed_state(tmp_path, messages)

    plan = ContextProjector(
        workspace_dir=tmp_path,
        session_id="session_1",
    ).build(
        messages=messages,
        state=state,
        run_id="run_1",
        pressure="tight",
    )

    projected_results = {
        item.message.tool_call_id: item
        for item in plan.messages
        if isinstance(item.message, ToolResultMessage)
    }
    assert _text(projected_results["call_a"].message) == "a" * 5_000
    assert _text(projected_results["call_b"].message) == "b" * 5_000
    assert projected_results["call_a"].action == "keep_full"
    assert projected_results["call_b"].action == "keep_full"
    assert messages == original


def test_consumed_exact_read_duplicate_omits_the_older_closed_batch(
    tmp_path: Path,
) -> None:
    old_assistant, old_result = _read_batch(
        "old", "src/app.py", "old duplicate body", sha256="same-hash"
    )
    new_assistant, new_result = _read_batch(
        "new", "src/app.py", "new canonical body", sha256="same-hash"
    )
    messages = (
        old_assistant,
        old_result,
        new_assistant,
        new_result,
        AssistantMessage(content=[TextContent(text="consumed")]),
    )

    plan = ContextProjector(
        workspace_dir=tmp_path,
        session_id="session_1",
    ).build(
        messages=messages,
        state=_observed_state(tmp_path, messages),
        run_id="run_1",
        pressure="normal",
    )

    old_items = [item for item in plan.messages if item.group_id == "group:0"]
    provider_results = [
        message
        for message in plan.model_messages
        if isinstance(message, ToolResultMessage)
    ]
    assert {item.action for item in old_items} == {"covered_by_newer_read"}
    assert [message.tool_call_id for message in provider_results] == ["call_new"]
    assert _text(provider_results[0]) == "new canonical body"


def test_same_read_version_with_different_ranges_keeps_both_bodies(
    tmp_path: Path,
) -> None:
    first_assistant, first_result = _read_batch(
        "first",
        "src/app.py",
        "lines 1-10",
        sha256="same-hash",
        offset=1,
        returned_lines=10,
    )
    second_assistant, second_result = _read_batch(
        "second",
        "src/app.py",
        "lines 11-20",
        sha256="same-hash",
        offset=11,
        returned_lines=10,
    )
    messages = (
        first_assistant,
        first_result,
        second_assistant,
        second_result,
        AssistantMessage(content=[TextContent(text="consumed")]),
    )

    plan = ContextProjector(
        workspace_dir=tmp_path,
        session_id="session_1",
    ).build(
        messages=messages,
        state=_observed_state(tmp_path, messages),
        run_id="run_1",
        pressure="normal",
    )

    provider_results = [
        message
        for message in plan.model_messages
        if isinstance(message, ToolResultMessage)
    ]
    assert [_text(message) for message in provider_results] == [
        "lines 1-10",
        "lines 11-20",
    ]


def test_session_regression_keeps_all_unique_ranges_from_one_large_file(
    tmp_path: Path,
) -> None:
    messages = []
    expected = []
    for index, offset in enumerate((1, 200, 400, 600, 800)):
        text = f"register.py lines {offset}-{offset + 199}"
        expected.append(text)
        messages.extend(
            _read_batch(
                f"segment_{index}",
                "agent-test/register.py",
                text,
                sha256="same-file-version",
                offset=offset,
                returned_lines=200,
            )
        )
    messages.append(AssistantMessage(content=[TextContent(text="consumed")]))

    plan = ContextProjector(
        workspace_dir=tmp_path,
        session_id="session_83a27f93be18",
    ).build(
        messages=tuple(messages),
        state=_observed_state(tmp_path, messages),
        pressure="normal",
    )

    visible_results = [
        message
        for message in plan.model_messages
        if isinstance(message, ToolResultMessage)
    ]
    assert [_text(message) for message in visible_results] == expected
    assert transcript.unsettled_tool_calls(plan.model_messages) == ()


def test_new_file_hash_does_not_hide_an_unreplaced_historical_read(
    tmp_path: Path,
) -> None:
    old_assistant = AssistantMessage(
        content=[
            ToolCall(id="call_old", name="read", arguments={"path": "src/app.py"}),
            ToolCall(id="call_shell", name="shell", arguments={"command": "git status"}),
        ]
    )
    old_read = _read_result(
        "call_old", "src/app.py", "obsolete source", sha256="old-hash"
    )
    shell_result = ToolResultMessage(
        tool_call_id="call_shell",
        tool_name="shell",
        content=[TextContent(text="working tree clean")],
    )
    new_assistant, new_read = _read_batch(
        "new", "src/app.py", "current source", sha256="new-hash"
    )
    messages = (
        old_assistant,
        old_read,
        shell_result,
        new_assistant,
        new_read,
        AssistantMessage(content=[TextContent(text="consumed")]),
    )

    plan = ContextProjector(
        workspace_dir=tmp_path,
        session_id="session_1",
    ).build(
        messages=messages,
        state=_observed_state(tmp_path, messages),
        run_id="run_1",
        pressure="normal",
    )
    projected = {
        message.tool_call_id: message
        for message in plan.model_messages
        if isinstance(message, ToolResultMessage)
    }

    assert _text(projected["call_old"]) == "obsolete source"
    assert _text(projected["call_shell"]) == "working tree clean"
    assert _text(projected["call_new"]) == "current source"
    assert transcript.unsettled_tool_calls(plan.model_messages) == ()


def test_consumed_large_tool_result_remains_full_in_lossless_projection(
    tmp_path: Path,
) -> None:
    assistant = AssistantMessage(
        content=[ToolCall(id="call_1", name="shell", arguments={"command": "pytest"})],
        metadata={"session_message_id": "msg_assistant"},
    )
    result = ToolResultMessage(
        tool_call_id="call_1",
        tool_name="shell",
        content=[TextContent(text="failure output\n" * 800)],
        status="error",
        exit_code=1,
        affected_paths=["test/test_app.py"],
        metadata={"session_message_id": "msg_result"},
    )
    consumed = AssistantMessage(content=[TextContent(text="continue")])
    state = ContextState(workspace_dir=tmp_path)
    state.observe_messages(
        (assistant, result, consumed), repository_fingerprint="workspace_1"
    )
    projector = ContextProjector(
        workspace_dir=tmp_path,
        session_id="session_1",
    )

    plan = projector.build(
        messages=(UserMessage(content="Fix tests."), assistant, result, consumed),
        state=state,
        run_id="run_1",
        pressure="tight",
    )

    projected_result = next(
        item for item in plan.messages if isinstance(item.message, ToolResultMessage)
    )

    assert projected_result.action == "keep_full"
    assert _text(projected_result.message) == "failure output\n" * 800
    assert "artifact_ref" not in projected_result.message.metadata
    assert not hasattr(plan, "evidence")
    assert [type(item.message) for item in plan.messages[-3:-1]] == [
        AssistantMessage,
        ToolResultMessage,
    ]


def test_stale_read_keeps_its_body_and_adds_a_historical_warning(
    tmp_path: Path,
) -> None:
    assistant, result = _read_batch(
        "old",
        "src/app.py",
        "def login():\n    return True",
        sha256="old-hash",
        returned_lines=2,
    )
    state = _observed_state(tmp_path, (assistant, result))
    state.invalidate_paths(["src/app.py"])

    plan = ContextProjector(
        workspace_dir=tmp_path,
        session_id="session_1",
    ).build(
        messages=(assistant, result, AssistantMessage(content=[TextContent(text="used")])),
        state=state,
        pressure="normal",
    )

    projected = next(
        message
        for message in plan.model_messages
        if isinstance(message, ToolResultMessage)
    )
    text = _text(projected)
    assert "freshness=stale" in text
    assert "def login():\n    return True" in text


def test_orphan_tool_result_is_removed_from_the_provider_message_chain(
    tmp_path: Path,
) -> None:
    result = ToolResultMessage(
        tool_call_id="missing_call",
        tool_name="shell",
        content=[TextContent(text="orphan")],
        metadata={"session_message_id": "msg_orphan"},
    )
    state = ContextState(workspace_dir=tmp_path)
    state.observe_messages((result,), repository_fingerprint="workspace_1")

    plan = ContextProjector(
        workspace_dir=tmp_path,
        session_id="session_1",
    ).build(
        messages=(result,),
        state=state,
        run_id="run_1",
        pressure="normal",
    )

    assert plan.messages[0].action == "discard_orphan"
    assert plan.model_messages == ()


def _read_batch(
    suffix: str,
    path: str,
    text: str,
    *,
    sha256: str,
    offset: int = 1,
    returned_lines: int = 1,
) -> tuple[AssistantMessage, ToolResultMessage]:
    call_id = f"call_{suffix}"
    return (
        AssistantMessage(
            content=[ToolCall(id=call_id, name="read", arguments={"path": path})]
        ),
        _read_result(
            call_id,
            path,
            text,
            sha256=sha256,
            offset=offset,
            returned_lines=returned_lines,
        ),
    )


def _read_result(
    call_id: str,
    path: str,
    text: str,
    *,
    sha256: str,
    offset: int = 1,
    returned_lines: int = 1,
) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=call_id,
        tool_name="read",
        content=[TextContent(text=text)],
        details={
            "path": path,
            "sha256": sha256,
            "offset": offset,
            "returned_lines": returned_lines,
        },
    )


def _observed_state(tmp_path: Path, messages) -> ContextState:
    state = ContextState(workspace_dir=tmp_path)
    state.observe_messages(tuple(messages), repository_fingerprint="workspace_1")
    return state


def _text(message: ToolResultMessage) -> str:
    return "\n".join(
        block.text for block in message.content if isinstance(block, TextContent)
    )
