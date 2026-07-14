from __future__ import annotations

import asyncio
import json
from pathlib import Path

from tool_runtime_testkit import execute_tool


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "docs" / "examples" / "extensions"


def test_demo_skill_loads_as_command_and_compact_index() -> None:
    from codepilot.extensions import CommandOutcome, SessionCommandContext, load_skills

    loaded = load_skills(ROOT, configured_paths=[str(EXAMPLES / "demo-review")])

    assert loaded.errors == []
    assert loaded.skills[0].manifest.name == "demo-review"
    assert loaded.skills[0].manifest.version == "1.0.0"
    assert "demo-review" in loaded.commands
    assert len(loaded.append_prompts) == 1
    assert "Available Skills" in loaded.append_prompts[0]
    assert "/demo-review" in loaded.append_prompts[0]
    assert "Use a compact review checklist" in loaded.append_prompts[0]
    assert "Goal:" not in loaded.append_prompts[0]
    assert "Verification:" not in loaded.append_prompts[0]

    command = loaded.commands["demo-review"]
    rendered = command.handler(
        SessionCommandContext(
            name="demo-review",
            args=[],
            raw_text="/demo-review check this change",
        )
    )

    assert isinstance(rendered, CommandOutcome)
    assert rendered.output is None
    assert rendered.prompt is not None
    assert "Call load_skill" in rendered.prompt
    assert "check this change" in rendered.prompt
    assert "Goal:" not in rendered.prompt


def test_demo_skill_registers_load_skill_tool_for_on_demand_content() -> None:
    from codepilot.extensions import load_skills

    loaded = load_skills(ROOT, configured_paths=[str(EXAMPLES / "demo-review")])

    assert [tool.spec.name for tool in loaded.tools] == ["load_skill", "read_skill_resource"]

    tool = loaded.tools[0]
    result = _execute_registration(tool, {"name": "demo-review"}, mode="plan")

    assert result.status == "success"
    assert result.content[0].text.startswith("Loaded skill demo-review v1.0.0")
    assert "Goal:" in result.content[0].text
    assert "Verification:" in result.content[0].text
    assert result.data["skill"] == "demo-review"
    assert result.data["command"] == "demo-review"
    assert result.data["resources"] == ("references/checklist.md",)

    resource = _execute_registration(
        loaded.tools[1],
        {"skill": "demo-review", "path": "references/checklist.md"},
        mode="plan",
    )
    assert resource.status == "success"
    assert "Check behavior" in resource.content[0].text
    assert resource.data["path"] == "references/checklist.md"


def test_demo_extension_registers_command_tool_and_prompt() -> None:
    from codepilot.extensions import SessionCommandContext, load_extensions

    loaded = load_extensions(ROOT, configured_paths=[str(EXAMPLES / "demo_extension.py")])

    assert loaded.errors == []
    assert "demo-extension" in loaded.commands
    assert [tool.spec.name for tool in loaded.tools] == ["demo_echo"]
    assert loaded.prompt_guidelines
    assert loaded.append_prompts

    command_output = loaded.commands["demo-extension"].handler(
        SessionCommandContext(
            name="demo-extension",
            args=[],
            raw_text="/demo-extension",
        )
    )
    assert command_output == "Demo extension is loaded."

    tool = loaded.tools[0]
    result = _execute_registration(tool, {"text": "hello"})
    assert result.content[0].text == "hello"
    assert result.data["demo_extension"] is True

def test_demo_mcp_config_creates_canonical_registration() -> None:
    from codepilot.extensions.mcp import (
        MCPRemoteTool,
        create_mcp_registrations,
        parse_mcp_server_configs,
    )

    raw = json.loads((EXAMPLES / "demo_mcp_config.json").read_text(encoding="utf-8"))
    configs = parse_mcp_server_configs(raw["mcp_servers"])

    assert len(configs) == 1
    assert configs[0].name == "demo"
    assert configs[0].allow_tools == frozenset({"echo"})

    class FakeMCPClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, dict[str, object]]] = []

        async def call_tool(
            self,
            server: str,
            tool: str,
            arguments: dict[str, object],
        ) -> object:
            self.calls.append((server, tool, arguments))
            return {"content": [{"type": "text", "text": arguments.get("text", "")}]}

    client = FakeMCPClient()
    remote = MCPRemoteTool(
        name="echo",
        description="Demo MCP echo tool.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        annotations={"readOnlyHint": True},
    )
    tools = create_mcp_registrations(configs[0], (remote,), client=client)

    assert [tool.spec.name for tool in tools] == ["mcp__demo__echo"]
    result = _execute_registration(tools[0], {"text": "hello"})

    assert client.calls == [("demo", "echo", {"text": "hello"})]
    assert "hello" in result.content[0].text
    assert result.data["result"]["content"][0]["text"] == "hello"
    assert result.content_trust == "untrusted"


def _execute_registration(registration, arguments, *, mode="execute"):
    from codepilot.tools import ToolExecutionRequest, ToolRegistry, ToolRuntime

    registry = ToolRegistry()
    registration_id = registry.register(registration)
    runtime = ToolRuntime(registry)
    return asyncio.run(
        execute_tool(
            runtime,
            ToolExecutionRequest(
                run_id="run-extension-example",
                session_id="session-extension-example",
                tool_call_id=f"call-{registration.spec.name}",
                tool_name=registration.spec.name,
                arguments=arguments,
                mode=mode,
                registration_id=registration_id,
            )
        )
    )
