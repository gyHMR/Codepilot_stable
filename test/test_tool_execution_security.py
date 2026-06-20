from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


def _metadata(
    name: str,
    *,
    read_only: bool,
    exclusive: bool,
):
    from codepilot.protocols import ToolMetadata

    return ToolMetadata(
        name=name,
        category="test",
        read_only=read_only,
        concurrency_safe=read_only,
        exclusive=exclusive,
        requires_approval=False,
        risk_level="low" if read_only else "medium",
        resource_scope=("workspace",),
        extra={
            "capabilities": [
                "filesystem.read" if read_only else "filesystem.write"
            ]
        },
    )


def test_permission_policy_uses_mode_and_shell_classification() -> None:
    from codepilot.tools.permissions import PermissionPolicy, ToolRequest

    bash = _metadata("bash", read_only=False, exclusive=True)
    read = _metadata("read", read_only=True, exclusive=False)

    assert PermissionPolicy(mode="read-only").decide(
        ToolRequest(name="read", metadata=read)
    ).allowed
    assert PermissionPolicy(mode="read-only").decide(
        ToolRequest(name="bash", params={"command": "pytest"}, metadata=bash)
    ).reason == "read_only_mode"
    assert PermissionPolicy(mode="workspace-write").decide(
        ToolRequest(
            name="bash",
            params={"command": "python -m pytest -q"},
            metadata=bash,
        )
    ).allowed
    unknown = PermissionPolicy(mode="workspace-write").decide(
        ToolRequest(name="bash", params={"command": "python script.py"}, metadata=bash)
    )
    assert unknown.requires_approval
    assert unknown.reason == "unknown_shell_command"
    assert PermissionPolicy(mode="ask").decide(
        ToolRequest(name="write", metadata=_metadata("write", read_only=False, exclusive=True))
    ).requires_approval


def test_model_cannot_authorize_dangerous_shell() -> None:
    from codepilot.tools.permissions import PermissionPolicy, ToolRequest

    decision = PermissionPolicy().decide(
        ToolRequest(
            name="bash",
            params={
                "command": "rm -rf .",
                "allow_dangerous": True,
            },
            metadata=_metadata("bash", read_only=False, exclusive=True),
        )
    )

    assert decision.denied
    assert decision.reason == "model_authorization_forbidden"


def test_invalid_permission_regex_is_reported() -> None:
    from codepilot.tools.permissions import PermissionPolicy

    with pytest.raises(ValueError, match="Invalid permission regex"):
        PermissionPolicy(bash_block_patterns=["["])


def test_approval_executes_original_request_once() -> None:
    asyncio.run(_approval_case())


async def _approval_case() -> None:
    from codepilot.protocols import TextContent
    from codepilot.tools import AgentTool, AgentToolResult, ToolRegistry, ToolRuntime
    from codepilot.tools.approval import ApprovalDecision
    from codepilot.tools.permissions import PermissionPolicy
    from codepilot.tools.types import ToolRuntimeRequest

    calls = []
    approved_request = None

    async def execute(tool_call_id, params, signal=None, on_update=None):
        _ = signal, on_update
        calls.append((tool_call_id, dict(params)))
        return AgentToolResult(content=[TextContent(text="done")])

    class Approve:
        async def request_approval(self, request, metadata, decision):
            nonlocal approved_request
            _ = metadata, decision
            approved_request = request
            return ApprovalDecision(
                approved=True,
                reason="user_approved",
                approval_id="approval_1",
            )

    registry = ToolRegistry()
    registry.register(
        AgentTool(
            name="write",
            label="write",
            description="write",
            parameters={},
            execute=execute,
        ),
        metadata=_metadata("write", read_only=False, exclusive=True),
    )
    runtime = ToolRuntime(
        registry,
        permission_policy=PermissionPolicy(mode="ask"),
        approval_provider=Approve(),
    )
    params = {"path": "a.txt", "content": "original"}
    result = await runtime.execute(
        ToolRuntimeRequest(
            tool_call_id="call_1",
            name="write",
            params=params,
        )
    )

    assert approved_request.params == params
    assert calls == [("call_1", params)]
    assert result.status == "success"
    assert result.approval_id == "approval_1"


def test_shell_policy_filters_environment_and_truncates_output(monkeypatch) -> None:
    from codepilot.tools.shell_policy import build_shell_environment, truncate_output

    monkeypatch.setenv("PATH", "safe-path")
    monkeypatch.setenv("CODEPILOT_TEST_SECRET", "must-not-leak")
    monkeypatch.setenv("CUSTOM_BUILD_MODE", "debug")

    env = build_shell_environment(("CUSTOM_BUILD_MODE", "CODEPILOT_TEST_SECRET"))
    output = truncate_output("A" * 100 + "TAIL", 40)

    assert env["PATH"] == "safe-path"
    assert env["CUSTOM_BUILD_MODE"] == "debug"
    assert "CODEPILOT_TEST_SECRET" not in env
    assert output.truncated
    assert output.text.endswith("TAIL")
    assert output.original_chars == 104


def test_shell_tool_rejects_invalid_timeout_and_hides_dangerous_parameter(tmp_path: Path) -> None:
    asyncio.run(_shell_limits_case(tmp_path))


async def _shell_limits_case(tmp_path: Path) -> None:
    from codepilot.tools.builtin import create_builtin_tools

    tools = {tool.name: tool for tool in create_builtin_tools(tmp_path)}
    bash = tools["bash"]
    result = await bash.execute(
        "bash_1",
        {"command": "python -m pytest", "timeout_seconds": 0},
    )

    assert result.error_code == "invalid_timeout"
    assert "allow_dangerous" not in bash.parameters["properties"]


def test_shell_tool_applies_env_allowlist_and_output_limits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    asyncio.run(_shell_runtime_limits_case(tmp_path, monkeypatch))


async def _shell_runtime_limits_case(tmp_path: Path, monkeypatch) -> None:
    from codepilot.tools.builtin.shell_tools import create_shell_tools
    from codepilot.tools.sandbox import WorkspaceSandbox
    from codepilot.tools.shell_policy import ShellExecutionPolicy

    captured = {}

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"A" * 100 + b"TAIL", b"E" * 80 + b"END"

    async def fake_subprocess(*_args, **kwargs):
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setenv("PATH", "safe-path")
    monkeypatch.setenv("CODEPILOT_TEST_SECRET", "must-not-leak")
    monkeypatch.setattr(asyncio, "create_subprocess_shell", fake_subprocess)
    tool = create_shell_tools(
        WorkspaceSandbox(tmp_path),
        allow=lambda name: name == "bash",
        policy=ShellExecutionPolicy(
            stdout_limit=40,
            stderr_limit=30,
            allowed_env=("CODEPILOT_TEST_SECRET",),
        ),
    )[0]

    result = await tool.execute(
        "bash_1",
        {"command": "python -m pytest -q"},
    )

    assert captured["env"]["PATH"] == "safe-path"
    assert "CODEPILOT_TEST_SECRET" not in captured["env"]
    assert result.metadata["stdout_truncated"] is True
    assert result.metadata["stderr_truncated"] is True
    assert result.metadata["stdout_original_chars"] == 104
    assert result.content[0].text.endswith("END")


def test_workspace_status_returns_structured_git_evidence(tmp_path: Path) -> None:
    asyncio.run(_workspace_status_case(tmp_path))


async def _workspace_status_case(tmp_path: Path) -> None:
    import subprocess

    from codepilot.tools.builtin import create_builtin_tools

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
    )
    tracked = tmp_path / "a.py"
    tracked.write_text("value = 1\n", encoding="utf-8", newline="\n")
    subprocess.run(["git", "add", "a.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True)
    tracked.write_text("value = 2\n", encoding="utf-8", newline="\n")

    tool = next(
        tool
        for tool in create_builtin_tools(tmp_path)
        if tool.name == "workspace_status"
    )
    result = await tool.execute("status_1", {})

    assert result.details["is_git_repo"] is True
    assert result.details["dirty"] is True
    assert result.details["changed_paths"][0]["path"] == "a.py"


def test_coordinator_parallelizes_reads_and_serializes_exclusive_tools() -> None:
    asyncio.run(_coordinator_scheduling_case())


async def _coordinator_scheduling_case() -> None:
    from codepilot.core.events import AgentEventEmitter
    from codepilot.core.tool_coordinator import ToolCallCoordinator
    from codepilot.core.types import AgentContext, AgentLoopConfig
    from codepilot.protocols import AssistantMessage, Model, TextContent, ToolCall
    from codepilot.tools import AgentTool, AgentToolResult

    active_reads = 0
    max_active_reads = 0
    reads_finished = asyncio.Event()
    read_count = 0
    write_started_after_reads = False

    async def read_execute(*_args):
        nonlocal active_reads, max_active_reads, read_count
        active_reads += 1
        max_active_reads = max(max_active_reads, active_reads)
        await asyncio.sleep(0.03)
        active_reads -= 1
        read_count += 1
        if read_count == 2:
            reads_finished.set()
        return AgentToolResult(content=[TextContent(text="read")])

    async def write_execute(*_args):
        nonlocal write_started_after_reads
        write_started_after_reads = reads_finished.is_set()
        return AgentToolResult(content=[TextContent(text="write")])

    read_tool = AgentTool(
        name="read",
        label="read",
        description="read",
        parameters={},
        execute=read_execute,
        runtime_managed=True,
        metadata=_metadata("read", read_only=True, exclusive=False),
    )
    write_tool = AgentTool(
        name="write",
        label="write",
        description="write",
        parameters={},
        execute=write_execute,
        runtime_managed=True,
        metadata=_metadata("write", read_only=False, exclusive=True),
    )
    model = Model(
        id="test",
        name="test",
        api="test",
        provider="test",
        base_url="",
        reasoning=False,
        input=["text"],
        context_window=1000,
        max_tokens=100,
    )
    events = []
    coordinator = ToolCallCoordinator(
        config=AgentLoopConfig(
            model=model,
            convert_to_llm=lambda messages: messages,
            tool_execution="parallel",
        ),
        emitter=AgentEventEmitter(events.append, run_id="run_1", session_id="session_1"),
    )
    assistant = AssistantMessage(
        content=[
            ToolCall(id="read_1", name="read"),
            ToolCall(id="read_2", name="read"),
            ToolCall(id="write_1", name="write"),
        ],
        stop_reason="toolUse",
    )

    results = await coordinator.execute_batch(
        AgentContext(
            system_prompt="",
            messages=[],
            tools=[read_tool, write_tool],
        ),
        assistant,
    )

    assert len(results) == 3
    assert max_active_reads == 2
    assert write_started_after_reads is True


def test_read_supports_line_ranges_and_search_skips_ignored_dirs(tmp_path: Path) -> None:
    asyncio.run(_bounded_read_search_case(tmp_path))


async def _bounded_read_search_case(tmp_path: Path) -> None:
    from codepilot.tools.builtin import create_builtin_tools

    target = tmp_path / "app.py"
    target.write_text(
        "line one\nline two\nline three\nline four\n",
        encoding="utf-8",
        newline="\n",
    )
    ignored = tmp_path / ".git"
    ignored.mkdir()
    (ignored / "hidden.py").write_text(
        "secret_pattern\n",
        encoding="utf-8",
        newline="\n",
    )
    tools = {tool.name: tool for tool in create_builtin_tools(tmp_path)}

    read = await tools["read"].execute(
        "read_1",
        {"path": "app.py", "offset": 2, "limit": 2},
    )
    grep = await tools["grep"].execute(
        "grep_1",
        {"pattern": "secret_pattern"},
    )

    assert read.content[0].text == "2\tline two\n3\tline three"
    assert read.metadata["start_line"] == 2
    assert read.metadata["end_line"] == 3
    assert read.metadata["truncated"] is True
    assert grep.content[0].text == "(no matches)"


def test_shell_failure_preserves_workspace_side_effects(tmp_path: Path) -> None:
    asyncio.run(_shell_side_effect_case(tmp_path))


async def _shell_side_effect_case(tmp_path: Path) -> None:
    import subprocess
    import sys

    from codepilot.tools.builtin import create_builtin_tools

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    seed = tmp_path / "seed.txt"
    seed.write_text("seed\n", encoding="utf-8", newline="\n")
    subprocess.run(["git", "add", "seed.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True)

    bash = next(tool for tool in create_builtin_tools(tmp_path) if tool.name == "bash")
    command = (
        f'"{sys.executable}" -c "from pathlib import Path; '
        "Path('generated.txt').write_text('created'); raise SystemExit(1)\""
    )
    result = await bash.execute("bash_1", {"command": command})

    assert result.status == "error"
    assert result.exit_code == 1
    assert result.workspace_changed is True
    assert "generated.txt" in result.affected_paths


def test_cli_approval_provider_supports_allow_and_deny() -> None:
    asyncio.run(_cli_approval_case())


async def _cli_approval_case() -> None:
    from codepilot.interfaces.cli.approval import CliApprovalProvider
    from codepilot.tools.permissions import ToolDecision
    from codepilot.tools.types import ToolRuntimeRequest

    outputs = []
    provider = CliApprovalProvider(
        input_fn=lambda _prompt: "y",
        output_fn=outputs.append,
    )
    decision = await provider.request_approval(
        ToolRuntimeRequest(
            tool_call_id="call_1",
            name="bash",
            params={"command": "python script.py"},
        ),
        _metadata("bash", read_only=False, exclusive=True),
        ToolDecision(
            "approval_required",
            "unknown_shell_command",
            {"capabilities": ["process.execute"], "risk_level": "medium"},
        ),
    )

    assert decision.approved is True
    assert decision.approval_id
    assert any("python script.py" in line for line in outputs)


def test_tool_result_message_preserves_approval_evidence() -> None:
    from codepilot.protocols import TextContent, ToolResultMessage
    from codepilot.sessions.serde import message_from_dict, message_to_dict

    message = ToolResultMessage(
        tool_call_id="call_1",
        tool_name="write",
        content=[TextContent(text="done")],
        approved=True,
        approval_id="approval_1",
    )
    restored = message_from_dict(message_to_dict(message))

    assert isinstance(restored, ToolResultMessage)
    assert restored.approved is True
    assert restored.approval_id == "approval_1"
