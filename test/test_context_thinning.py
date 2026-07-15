from __future__ import annotations

from pathlib import Path

from codepilot.core.transcript import unsettled_tool_calls
from codepilot.protocols import (
    AssistantMessage,
    Message,
    TextContent,
    ToolCall,
    ToolResultMessage,
)
from codepilot.sessions.context.state import ContextState
from codepilot.sessions.context.thinning import ContextThinner


def test_tight_thinning_archives_old_recoverable_tool_output(tmp_path: Path) -> None:
    old = _tool_batch("old", "shell", "old output\n" * 20_000)
    recent = _tool_batch("recent", "shell", "recent output\n" * 20_000)
    messages = (*old, *recent, AssistantMessage(content=[TextContent(text="used")]))
    state = _state(tmp_path, messages)

    result = ContextThinner(workspace_dir=tmp_path).thin(
        messages,
        state=state,
        run_id="run_1",
        target_tokens=0,
        estimate_tokens=_estimate,
    )

    old_result = _result(result.messages, "call_old")
    recent_result = _result(result.messages, "call_recent")
    assert result.actions == ("old_tool_output:call_old",)
    assert "Older tool output omitted" in _text(old_result)
    assert _text(recent_result) == "recent output\n" * 20_000
    artifact = tmp_path / str(old_result.metadata["artifact_ref"])
    assert artifact.read_text(encoding="utf-8") == "old output\n" * 20_000
    assert unsettled_tool_calls(result.messages) == ()


def test_unique_read_body_is_not_deterministically_thinned(tmp_path: Path) -> None:
    read = _read_batch("read", "src/app.py", "source code\n" * 200, "hash-a")
    recent = _tool_batch("recent", "shell", "recent output\n" * 20_000)
    messages = (*read, *recent, AssistantMessage(content=[TextContent(text="used")]))

    result = ContextThinner(workspace_dir=tmp_path).thin(
        messages,
        state=_state(tmp_path, messages),
        run_id="run_1",
        target_tokens=0,
        estimate_tokens=_estimate,
    )

    assert result.actions == ()
    assert _text(_result(result.messages, "call_read")) == "source code\n" * 200


def test_stale_read_is_thinned_only_after_fresh_coverage_exists(
    tmp_path: Path,
) -> None:
    old = _read_batch("old", "src/app.py", "old source", "old-hash")
    new = _read_batch("new", "src/app.py", "new source", "new-hash")
    recent = _tool_batch("recent", "shell", "recent output\n" * 20_000)
    messages = (*old, *new, *recent, AssistantMessage(content=[TextContent(text="used")]))
    state = ContextState(workspace_dir=tmp_path)
    state.observe_messages(old, repository_fingerprint="workspace-old")
    state.invalidate_paths(["src/app.py"])
    state.observe_messages(new, repository_fingerprint="workspace-new")
    state.observe_messages(recent, repository_fingerprint="workspace-new")

    result = ContextThinner(workspace_dir=tmp_path).thin(
        messages,
        state=state,
        run_id="run_1",
        target_tokens=0,
        estimate_tokens=_estimate,
    )

    assert result.actions == ("read_covered:call_old",)
    assert "reason=read_covered" in _text(_result(result.messages, "call_old"))
    assert _text(_result(result.messages, "call_new")) == "new source"


def test_recent_tool_protection_scales_for_a_small_context_window(
    tmp_path: Path,
) -> None:
    old = _tool_batch("old", "shell", "old output\n" * 2_000)
    middle = _tool_batch("middle", "shell", "middle output\n" * 2_000)
    recent = _tool_batch("recent", "shell", "recent output\n" * 2_000)
    messages = (
        *old,
        *middle,
        *recent,
        AssistantMessage(content=[TextContent(text="used")]),
    )

    result = ContextThinner(workspace_dir=tmp_path).thin(
        messages,
        state=_state(tmp_path, messages),
        run_id="run_1",
        target_tokens=0,
        recent_tool_tokens=4_000,
        estimate_tokens=_estimate,
    )

    assert result.actions == (
        "old_tool_output:call_old",
        "old_tool_output:call_middle",
    )
    assert _text(_result(result.messages, "call_recent")) == "recent output\n" * 2_000


def test_recent_parallel_tool_batch_is_protected_as_one_unit(tmp_path: Path) -> None:
    old = _tool_batch("old", "shell", "old output\n" * 2_000)
    assistant = AssistantMessage(
        content=[
            ToolCall(id="call_a", name="shell", arguments={}),
            ToolCall(id="call_b", name="shell", arguments={}),
        ]
    )
    result_a = ToolResultMessage(
        tool_call_id="call_a",
        tool_name="shell",
        content=[TextContent(text="a output\n" * 2_000)],
    )
    result_b = ToolResultMessage(
        tool_call_id="call_b",
        tool_name="shell",
        content=[TextContent(text="b output\n" * 2_000)],
    )
    messages = (
        *old,
        assistant,
        result_a,
        result_b,
        AssistantMessage(content=[TextContent(text="used")]),
    )

    result = ContextThinner(workspace_dir=tmp_path).thin(
        messages,
        state=_state(tmp_path, messages),
        run_id="run_1",
        target_tokens=0,
        recent_tool_tokens=1,
        estimate_tokens=_estimate,
    )

    assert result.actions == ("old_tool_output:call_old",)
    assert _text(_result(result.messages, "call_a")) == "a output\n" * 2_000
    assert _text(_result(result.messages, "call_b")) == "b output\n" * 2_000
    assert unsettled_tool_calls(result.messages) == ()


def _tool_batch(
    suffix: str,
    name: str,
    text: str,
) -> tuple[AssistantMessage, ToolResultMessage]:
    call_id = f"call_{suffix}"
    return (
        AssistantMessage(content=[ToolCall(id=call_id, name=name, arguments={})]),
        ToolResultMessage(
            tool_call_id=call_id,
            tool_name=name,
            content=[TextContent(text=text)],
        ),
    )


def _read_batch(
    suffix: str,
    path: str,
    text: str,
    sha256: str,
) -> tuple[AssistantMessage, ToolResultMessage]:
    assistant, result = _tool_batch(suffix, "read", text)
    result.details = {
        "path": path,
        "sha256": sha256,
        "offset": 1,
        "returned_lines": max(1, len(text.splitlines())),
    }
    return assistant, result


def _state(tmp_path: Path, messages: tuple[Message, ...]) -> ContextState:
    state = ContextState(workspace_dir=tmp_path)
    state.observe_messages(messages, repository_fingerprint="workspace")
    return state


def _estimate(messages: tuple[Message, ...]) -> int:
    return sum(
        len(_text(message)) // 4
        for message in messages
        if isinstance(message, ToolResultMessage)
    )


def _result(messages: tuple[Message, ...], call_id: str) -> ToolResultMessage:
    return next(
        message
        for message in messages
        if isinstance(message, ToolResultMessage) and message.tool_call_id == call_id
    )


def _text(message: ToolResultMessage) -> str:
    return "\n".join(
        block.text for block in message.content if isinstance(block, TextContent)
    )
