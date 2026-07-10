from __future__ import annotations

import asyncio
import json
from pathlib import Path


def test_runtime_tools_catalog_is_filtered_by_current_mode(tmp_path: Path) -> None:
    from codepilot.runtime import SessionOpenIntent
    from codepilot.runtime.config import load_runtime_config
    from codepilot.runtime.tools import build_runtime_tools

    config = load_runtime_config(
        SessionOpenIntent(workspace_dir=tmp_path, current_mode="read")
    )
    loaded = build_runtime_tools(tmp_path, SessionOpenIntent(workspace_dir=tmp_path), config)

    names = {tool.name for tool in loaded.specs}
    assert {"ls", "read", "grep", "find", "workspace_status", "update_plan"} <= names
    assert "write" not in names
    assert "bash" not in names


def test_runtime_tools_registers_caller_tools_without_reserved_name_compat(tmp_path: Path) -> None:
    from codepilot.protocols import TextContent
    from codepilot.runtime import SessionOpenIntent
    from codepilot.runtime.config import load_runtime_config
    from codepilot.runtime.tools import build_runtime_tools
    from codepilot.tools.contracts import ToolDefinition, ToolMetadata, ToolResult

    async def execute(request, signal=None, on_update=None):
        _ = request, signal, on_update
        return ToolResult(content=[TextContent(text="ok")])

    custom = ToolDefinition(
        name="custom_echo",
        label="Custom Echo",
        description="Echo from caller",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        metadata=ToolMetadata(
            name="custom_echo",
            category="extension",
            read_only=True,
            concurrency_safe=True,
            exclusive=False,
            requires_approval=False,
            risk_level="low",
            scopes=("read", "plan", "build"),
        ),
        execute=execute,
    )
    reserved = ToolDefinition(
        name="read",
        label="Reserved",
        description="Reserved name should be ignored",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        metadata=ToolMetadata(
            name="read",
            category="extension",
            read_only=True,
            concurrency_safe=True,
            exclusive=False,
            requires_approval=False,
            risk_level="low",
            scopes=("read", "plan", "build"),
        ),
        execute=execute,
    )

    intent = SessionOpenIntent(workspace_dir=tmp_path, tools=[custom, reserved])
    loaded = build_runtime_tools(tmp_path, intent, load_runtime_config(intent))

    assert loaded.registry.get("custom_echo") is custom
    assert loaded.registry.get("read") is not reserved
    assert any("reserved builtin name" in warning for warning in loaded.warnings)


def test_skill_loader_tool_loads_discovered_skill_content(tmp_path: Path) -> None:
    from codepilot.runtime import SessionOpenIntent
    from codepilot.runtime.config import load_runtime_config
    from codepilot.runtime.tools import build_runtime_tools
    from codepilot.tools.contracts import ToolInvocation
    from codepilot.tools.permissions import PermissionPolicy
    from codepilot.tools.runtime import ToolRuntime

    skill_dir = tmp_path / ".codepilot" / "skills"
    skill_dir.mkdir(parents=True)
    (skill_dir / "demo.md").write_text(
        "\n".join(
            [
                "---",
                "name: Demo Skill",
                "command: demo",
                "description: Demo workflow",
                "---",
                "# Demo",
                "Use this workflow.",
            ]
        ),
        encoding="utf-8",
    )

    intent = SessionOpenIntent(workspace_dir=tmp_path)
    loaded = build_runtime_tools(tmp_path, intent, load_runtime_config(intent))
    runtime = ToolRuntime(loaded.registry, permission_policy=PermissionPolicy())

    observation = asyncio.run(
        runtime.execute(
            ToolInvocation(
                run_id="run1",
                tool_call_id="skill1",
                name="load_skill",
                arguments={"name": "demo"},
                current_mode="build",
            )
        )
    )

    assert observation.status == "success"
    assert "Use this workflow." in observation.content[0].text
    assert observation.metadata["details"]["command"] == "demo"


def test_grep_searches_when_path_is_specific_file(tmp_path: Path) -> None:
    from codepilot.tools.builtins.search import create_search_tools
    from codepilot.tools.contracts import ToolCallRequest
    from codepilot.tools.sandbox import WorkspaceSandbox

    source = tmp_path / "src" / "register.py"
    source.parent.mkdir()
    source.write_text(
        "class UserRegister:\n    pass\n",
        encoding="utf-8",
        newline="\n",
    )
    tools = {
        tool.name: tool
        for tool in create_search_tools(
            WorkspaceSandbox(tmp_path),
            allow=lambda name: name in {"grep", "find"},
        )
    }

    result = asyncio.run(
        tools["grep"].execute(
            ToolCallRequest(
                run_id="run1",
                tool_call_id="grep1",
                name="grep",
                arguments={
                    "path": "src/register.py",
                    "pattern": "class UserRegister",
                },
                current_mode="build",
            )
        )
    )

    assert result.status == "success"
    assert "src/register.py:1:class UserRegister:" in result.content[0].text
    assert result.details["scanned_files"] == 1
    assert result.metadata["matches"] == 1


def test_tools_json_limits_enabled_builtin_tools(tmp_path: Path) -> None:
    from codepilot.runtime import SessionOpenIntent
    from codepilot.runtime.config import load_runtime_config
    from codepilot.runtime.tools import build_runtime_tools

    config_dir = tmp_path / ".codepilot"
    config_dir.mkdir()
    (config_dir / "tools.json").write_text(
        json.dumps({"enabled": ["read", "workspace_status", "update_plan"]}),
        encoding="utf-8",
    )

    intent = SessionOpenIntent(workspace_dir=tmp_path)
    loaded = build_runtime_tools(tmp_path, intent, load_runtime_config(intent))

    assert {tool.name for tool in loaded.specs} == {
        "read",
        "workspace_status",
        "update_plan",
    }
