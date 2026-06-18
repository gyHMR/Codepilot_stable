from __future__ import annotations

import sys
import asyncio
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_workspace_sandbox_blocks_path_escape(tmp_path: Path) -> None:
    from codepilot.tools.sandbox import WorkspaceSandbox

    sandbox = WorkspaceSandbox(tmp_path)

    with pytest.raises(ValueError, match="workspace boundary"):
        sandbox.resolve_path("../outside.txt")


def test_permission_policy_blocks_dangerous_bash() -> None:
    from codepilot.tools.permissions import PermissionPolicy, ToolRequest

    decision = PermissionPolicy().decide(ToolRequest(name="bash", params={"command": "rm -rf ."}))

    assert decision.denied
    assert decision.reason == "dangerous_command"


def test_permission_policy_supports_future_approval_flow() -> None:
    from codepilot.tools.permissions import PermissionPolicy, ToolRequest

    decision = PermissionPolicy(require_approval_for_mutations=True).decide(
        ToolRequest(name="write", params={"path": "a.txt", "content": "x"})
    )

    assert decision.requires_approval
    assert decision.reason == "mutation_requires_approval"


def test_permission_policy_uses_tool_metadata_for_read_only() -> None:
    from codepilot.tools.permissions import PermissionPolicy, ToolRequest
    from codepilot.tools.types import ToolMetadata

    metadata = ToolMetadata(
        name="database.query",
        category="database",
        read_only=True,
        concurrency_safe=True,
        exclusive=False,
        requires_approval=False,
        risk_level="low",
        resource_scope=("database",),
    )

    decision = PermissionPolicy(read_only=True).decide(
        ToolRequest(name="database.query", params={"sql": "select 1"}, metadata=metadata)
    )

    assert decision.allowed


def test_permission_policy_requires_approval_for_high_risk_metadata() -> None:
    from codepilot.tools.permissions import PermissionPolicy, ToolRequest
    from codepilot.tools.types import ToolMetadata

    metadata = ToolMetadata(
        name="email.send",
        category="email",
        read_only=False,
        concurrency_safe=False,
        exclusive=True,
        requires_approval=False,
        risk_level="high",
        resource_scope=("email",),
        network_access=True,
        credential_required=True,
    )

    decision = PermissionPolicy().decide(
        ToolRequest(name="email.send", params={"to": "a@example.com"}, metadata=metadata)
    )

    assert decision.requires_approval
    assert decision.reason == "high_risk_tool_requires_approval"
    assert decision.details["resource_scope"] == ["email"]


def test_tool_runtime_blocks_dangerous_bash_before_execution() -> None:
    asyncio.run(_run_tool_runtime_blocks_dangerous_bash_before_execution())


def test_tool_runtime_requires_approval_for_high_risk_tool() -> None:
    asyncio.run(_run_tool_runtime_requires_approval_for_high_risk_tool())


def test_tool_runtime_agent_tool_adapter_preserves_denied_status() -> None:
    asyncio.run(_run_tool_runtime_agent_tool_adapter_preserves_denied_status())


async def _run_tool_runtime_blocks_dangerous_bash_before_execution() -> None:
    from codepilot.protocols import TextContent
    from codepilot.tools import AgentTool, AgentToolResult
    from codepilot.tools.registry import ToolRegistry
    from codepilot.tools.runtime import ToolRuntime
    from codepilot.tools.types import ToolRuntimeRequest

    executed = False

    async def execute(tool_call_id, params, signal=None, on_update=None):
        nonlocal executed
        _ = tool_call_id, params, signal, on_update
        executed = True
        return AgentToolResult(content=[TextContent(text="executed")], details={})

    registry = ToolRegistry()
    registry.register(
        AgentTool(
            name="bash",
            label="Bash",
            description="Run shell command",
            parameters={},
            execute=execute,
        )
    )
    runtime = ToolRuntime(registry)
    result = await runtime.execute(
        ToolRuntimeRequest(
            tool_call_id="tool_1",
            name="bash",
            params={"command": "rm -rf ."},
        )
    )

    assert result.is_error
    assert result.status == "denied"
    assert not executed
    assert result.result.details["reason"] == "dangerous_command"
    assert result.result.details["status"] == "denied"


async def _run_tool_runtime_requires_approval_for_high_risk_tool() -> None:
    from codepilot.protocols import TextContent
    from codepilot.tools import AgentTool, AgentToolResult
    from codepilot.tools.registry import ToolRegistry
    from codepilot.tools.runtime import ToolRuntime
    from codepilot.tools.types import ToolMetadata, ToolRuntimeRequest

    executed = False

    async def execute(tool_call_id, params, signal=None, on_update=None):
        nonlocal executed
        _ = tool_call_id, params, signal, on_update
        executed = True
        return AgentToolResult(content=[TextContent(text="sent")], details={})

    registry = ToolRegistry()
    registry.register(
        AgentTool(
            name="email.send",
            label="Send email",
            description="Send an email",
            parameters={},
            execute=execute,
        ),
        metadata=ToolMetadata(
            name="email.send",
            category="email",
            read_only=False,
            concurrency_safe=False,
            exclusive=True,
            requires_approval=False,
            risk_level="high",
            resource_scope=("email",),
            network_access=True,
            credential_required=True,
        ),
    )
    runtime = ToolRuntime(registry)
    result = await runtime.execute(
        ToolRuntimeRequest(
            tool_call_id="tool_1",
            name="email.send",
            params={"to": "a@example.com"},
        )
    )

    assert result.is_error
    assert result.status == "approval_required"
    assert not result.approved
    assert not executed
    assert result.result.details["reason"] == "high_risk_tool_requires_approval"
    assert result.result.details["status"] == "approval_required"


async def _run_tool_runtime_agent_tool_adapter_preserves_denied_status() -> None:
    from codepilot.protocols import TextContent
    from codepilot.tools import AgentTool, AgentToolResult
    from codepilot.tools.registry import ToolRegistry
    from codepilot.tools.runtime import ToolRuntime

    executed = False

    async def execute(tool_call_id, params, signal=None, on_update=None):
        nonlocal executed
        _ = tool_call_id, params, signal, on_update
        executed = True
        return AgentToolResult(content=[TextContent(text="executed")], details={})

    registry = ToolRegistry()
    registry.register(
        AgentTool(
            name="bash",
            label="Bash",
            description="Run shell command",
            parameters={},
            execute=execute,
        )
    )
    agent_tool = ToolRuntime(registry).as_agent_tools()[0]

    assert getattr(agent_tool, "runtime_managed") is True

    result = await agent_tool.execute(
        "tool_1",
        {"command": "rm -rf ."},
    )

    assert result.is_error
    assert result.status == "denied"
    assert not result.approved
    assert not executed
    assert result.details["reason"] == "dangerous_command"
    assert result.details["status"] == "denied"


def test_unknown_external_tool_metadata_is_conservative() -> None:
    from codepilot.protocols import TextContent
    from codepilot.tools import AgentTool, AgentToolResult
    from codepilot.tools.registry import ToolRegistry

    async def execute(tool_call_id, params, signal=None, on_update=None):
        _ = tool_call_id, params, signal, on_update
        return AgentToolResult(content=[TextContent(text="ok")], details={})

    registry = ToolRegistry()
    registry.register(
        AgentTool(
            name="calendar.create_event",
            label="Calendar",
            description="Create event",
            parameters={},
            execute=execute,
        )
    )

    metadata = registry.metadata_for("calendar.create_event")

    assert metadata is not None
    assert metadata.category == "extension"
    assert not metadata.read_only
    assert metadata.risk_level == "medium"
    assert metadata.requires_approval


def test_read_only_tool_assembly_filters_by_metadata(tmp_path: Path) -> None:
    from codepilot.protocols import TextContent
    from codepilot.tools import AgentTool, AgentToolResult
    from codepilot.runtime.config_loader import RuntimeConfig
    from codepilot.runtime.tool_assembler import assemble_tools
    from codepilot.runtime.types import CreateAgentSessionOptions

    async def execute(tool_call_id, params, signal=None, on_update=None):
        _ = tool_call_id, params, signal, on_update
        return AgentToolResult(content=[TextContent(text="ok")], details={})

    external_tool = AgentTool(
        name="calendar.create_event",
        label="Calendar",
        description="Create event",
        parameters={},
        execute=execute,
    )
    config = RuntimeConfig(
        system_prompt="",
        thinking_level="off",
        tool_execution="parallel",
        max_context_messages=None,
        retain_recent_messages=24,
        max_context_tokens=None,
        retry_enabled=True,
        max_retries=2,
        retry_base_delay_ms=1200,
        read_only_mode=True,
        block_dangerous_bash=True,
        bash_allow_patterns=None,
        bash_block_patterns=None,
        edit_require_unique_match=True,
        extension_paths=[],
        skill_paths=[],
        mcp_servers=[],
        prompt_guidelines=None,
        append_system_prompt=None,
        prompt_debug_sources=False,
        tool_snippets=None,
        enabled_builtin_tools=None,
    )
    assembled = assemble_tools(
        tmp_path,
        CreateAgentSessionOptions(workspace_dir=tmp_path, tools=[external_tool]),
        config,
    )

    names = {tool.name for tool in assembled.tools}

    assert "read" in names
    assert "grep" in names
    assert "calendar.create_event" not in names
