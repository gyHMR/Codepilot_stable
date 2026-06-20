from __future__ import annotations

import asyncio
from pathlib import Path


def test_repository_tracker_detects_external_dirty_file_changes(tmp_path: Path) -> None:
    from codepilot.runtime.repository_tracker import RepositoryTracker

    tracked = tmp_path / "app.py"
    tracked.write_text("value = 1\n", encoding="utf-8", newline="\n")
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "add", "app.py")
    _git(tmp_path, "commit", "-m", "initial")

    tracker = RepositoryTracker(tmp_path)
    first = tracker.snapshot()
    tracked.write_text("value = 2\n", encoding="utf-8", newline="\n")
    second, first_delta = tracker.refresh(first)
    tracked.write_text("value = 3\n", encoding="utf-8", newline="\n")
    third, second_delta = tracker.refresh(second)

    assert first.fingerprint != second.fingerprint
    assert second.fingerprint != third.fingerprint
    assert "app.py" in first_delta.modified_paths
    assert "app.py" in second_delta.modified_paths


def test_context_compiler_preserves_current_request_and_reports_budget(tmp_path: Path) -> None:
    asyncio.run(_compile_context_case(tmp_path))


async def _compile_context_case(tmp_path: Path) -> None:
    from codepilot.core import AgentContext, ContextPreparationRequest
    from codepilot.protocols import TextContent, ToolResultMessage, UserMessage
    from codepilot.runtime.context_compiler import ContextCompiler, ContextPolicy
    from codepilot.sessions.context_state import SessionContextState

    target = tmp_path / "service.py"
    target.write_text("def run():\n    return 1\n", encoding="utf-8", newline="\n")
    messages = [
        UserMessage(content=f"old message {index} " + "x" * 300)
        for index in range(12)
    ]
    messages.append(UserMessage(content="CURRENT REQUEST MUST SURVIVE"))
    messages.append(
        ToolResultMessage(
            tool_call_id="read_1",
            tool_name="read",
            content=[TextContent(text="def run(): return 1")],
            details={
                "file_state": {
                    "path": "service.py",
                    "sha256": "source-hash",
                    "exists": True,
                }
            },
        )
    )
    state = SessionContextState(workspace_dir=tmp_path)
    compiler = ContextCompiler(
        workspace=str(tmp_path),
        state=state,
        policy=ContextPolicy(minimum_input_budget=200, safety_margin_tokens=0),
    )

    prepared = await compiler.compile(
        AgentContext(
            system_prompt="rules\n\n## Repository Context\n- stale snapshot\n\n当前日期：2026-01-01",
            messages=messages,
        ),
        ContextPreparationRequest(
            session_id="session_1",
            model_context_window=500,
            model_max_output_tokens=100,
        ),
    )

    assert any(
        isinstance(message, UserMessage)
        and message.content == "CURRENT REQUEST MUST SURVIVE"
        for message in prepared.messages
    )
    assert "stale snapshot" not in prepared.system_prompt
    assert "Repository fingerprint:" in prepared.system_prompt
    assert "service.py" in prepared.system_prompt
    assert prepared.report.dropped_items
    assert prepared.report.estimated_tokens_after <= prepared.report.estimated_tokens_before


def test_context_compiler_marks_hash_bound_summary_stale(tmp_path: Path) -> None:
    asyncio.run(_stale_summary_case(tmp_path))


async def _stale_summary_case(tmp_path: Path) -> None:
    from codepilot.core import AgentContext, ContextPreparationRequest
    from codepilot.protocols import UserMessage
    from codepilot.runtime.context_compiler import ContextCompiler
    from codepilot.sessions.context_state import FileSummary, SessionContextState
    from codepilot.tools.sandbox import file_state_for_path

    target = tmp_path / "service.py"
    target.write_text("value = 1\n", encoding="utf-8", newline="\n")
    source_hash = file_state_for_path(tmp_path, "service.py")["sha256"]
    state = SessionContextState(workspace_dir=tmp_path)
    state.file_summaries["service.py"] = FileSummary(
        path="service.py",
        summary="service returns one",
        source_hash=source_hash,
    )
    target.write_text("value = 2\n", encoding="utf-8", newline="\n")

    prepared = await ContextCompiler(
        workspace=str(tmp_path),
        state=state,
    ).compile(
        AgentContext(system_prompt="rules", messages=[UserMessage(content="inspect")]),
        ContextPreparationRequest(
            session_id="session_1",
            model_context_window=4000,
            model_max_output_tokens=500,
        ),
    )

    assert state.file_summaries["service.py"].freshness == "stale"
    assert any(item.startswith("file_summary:service.py:stale") for item in prepared.report.stale_items)
    assert "service returns one" not in prepared.system_prompt


def test_edit_rejects_stale_expected_file_hash(tmp_path: Path) -> None:
    asyncio.run(_stale_edit_case(tmp_path))


async def _stale_edit_case(tmp_path: Path) -> None:
    from codepilot.tools.builtin.file_tools import create_file_tools
    from codepilot.tools.sandbox import WorkspaceSandbox, file_state_for_path

    target = tmp_path / "app.py"
    target.write_text("value = 1\n", encoding="utf-8", newline="\n")
    stale_hash = file_state_for_path(tmp_path, "app.py")["sha256"]
    target.write_text("value = 2\n", encoding="utf-8", newline="\n")
    tools = create_file_tools(WorkspaceSandbox(tmp_path), allow=lambda _name: True)
    edit = next(tool for tool in tools if tool.name == "edit")

    result = await edit.execute(
        "edit_1",
        {
            "path": "app.py",
            "old_text": "value = 2",
            "new_text": "value = 3",
            "expected_file_hash": stale_hash,
        },
    )

    assert result.is_error
    assert result.error_code == "stale_file"
    assert target.read_text(encoding="utf-8") == "value = 2\n"


def test_agent_loop_compiles_context_before_each_model_call(tmp_path: Path) -> None:
    asyncio.run(_per_model_call_compile_case(tmp_path))


async def _per_model_call_compile_case(tmp_path: Path) -> None:
    from codepilot.core import AgentContext, AgentLoopConfig, run_agent_loop
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import AssistantMessage, Model, TextContent, ToolCall, ToolResultMessage, UserMessage
    from codepilot.runtime.context_compiler import ContextCompiler
    from codepilot.sessions.context_state import SessionContextState
    from codepilot.tools import AgentTool, AgentToolResult

    target = tmp_path / "generated.py"

    async def fake_stream(_model, context, _options):
        stream = AssistantMessageEventStream()
        if any(isinstance(message, ToolResultMessage) for message in context.messages):
            stream.end(AssistantMessage(content=[TextContent(text="done")]))
        else:
            stream.end(
                AssistantMessage(
                    content=[ToolCall(id="write_1", name="write_test", arguments={})],
                    stop_reason="toolUse",
                )
            )
        return stream

    async def write_tool(*_args):
        target.write_text("created = True\n", encoding="utf-8", newline="\n")
        return AgentToolResult(
            content=[TextContent(text="created generated.py")],
            affected_paths=["generated.py"],
            workspace_changed=True,
        )

    compiler = ContextCompiler(
        workspace=str(tmp_path),
        state=SessionContextState(workspace_dir=tmp_path),
    )
    events = []
    await run_agent_loop(
        prompts=[UserMessage(content="create a file")],
        context=AgentContext(
            system_prompt="rules",
            messages=[],
            tools=[
                AgentTool(
                    name="write_test",
                    label="write",
                    description="write",
                    parameters={},
                    execute=write_tool,
                )
            ],
        ),
        config=AgentLoopConfig(
            model=Model(
                id="context-test",
                name="Context Test",
                api="unit-test",
                provider="unit-test",
                base_url="",
                reasoning=False,
                input=["text"],
                context_window=4000,
                max_tokens=500,
            ),
            convert_to_llm=lambda items: items,
            prepare_context=compiler.compile,
            allow_unmanaged_tools=True,
        ),
        emit=events.append,
        stream_fn=fake_stream,
    )

    reports = [event["report"] for event in events if event.get("type") == "context_prepared"]
    # The task completion gate adds one extra model turn after a workspace
    # change without fresh verification, so context is compiled for:
    # initial action, post-tool response, and verification steering.
    assert len(reports) == 3
    assert reports[0]["repository_fingerprint"] != reports[1]["repository_fingerprint"]
    assert reports[1]["repository_delta"]["added_paths"]


def _git(root: Path, *args: str) -> None:
    import subprocess

    subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
