from __future__ import annotations

from pathlib import Path

from codepilot.protocols import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from codepilot.sessions.context.projection import ContextProjector
from codepilot.sessions.context.state import ContextState


def test_l2_and_l4_share_one_source_ref_for_projected_tool_results(
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
    state = ContextState(workspace_dir=tmp_path)
    state.observe_messages((assistant, result), repository_fingerprint="workspace_1")
    projector = ContextProjector(
        workspace_dir=tmp_path,
        session_id="session_1",
    )

    plan = projector.build(
        messages=(UserMessage(content="Fix tests."), assistant, result),
        state=state,
        run_id="run_1",
        pressure="tight",
    )

    projected_result = next(
        item for item in plan.messages if isinstance(item.message, ToolResultMessage)
    )
    evidence = next(item for item in plan.evidence if item.source_ref == projected_result.source_ref)
    artifact_ref = projected_result.message.metadata["artifact_ref"]

    assert projected_result.action == "replace_with_artifact_ref"
    assert evidence.action == "show_projected_status"
    assert str(artifact_ref).startswith(".codepilot/runs/run_1/artifacts/tool_outputs/")
    assert (tmp_path / str(artifact_ref)).is_file()
    assert [type(item.message) for item in plan.messages[-2:]] == [
        AssistantMessage,
        ToolResultMessage,
    ]


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
