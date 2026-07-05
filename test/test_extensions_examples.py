from __future__ import annotations

import asyncio
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "docs" / "examples" / "extensions"


def test_demo_skill_loads_as_command_and_compact_index() -> None:
    from codepilot.extensions import SessionCommandContext, load_skills

    loaded = load_skills(ROOT, configured_paths=[str(EXAMPLES / "demo_skill.md")])

    assert loaded.errors == []
    assert loaded.skills[0].name == "Demo Review Checklist"
    assert loaded.skills[0].command_name == "demo-review"
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

    assert isinstance(rendered, str)
    assert "Applied skill Demo Review Checklist" in rendered
    assert "Goal:" in rendered
    assert "Verification:" in rendered


def test_demo_skill_registers_load_skill_tool_for_on_demand_content() -> None:
    from codepilot.extensions import load_skills

    loaded = load_skills(ROOT, configured_paths=[str(EXAMPLES / "demo_skill.md")])

    assert [tool.name for tool in loaded.tools] == ["load_skill"]

    tool = loaded.tools[0]
    result = asyncio.run(tool.execute("call_load_skill", {"name": "demo-review"}))

    assert not result.is_error
    assert result.content[0].text.startswith("Loaded skill Demo Review Checklist")
    assert "Goal:" in result.content[0].text
    assert "Verification:" in result.content[0].text
    assert result.details["skill"] == "Demo Review Checklist"
    assert result.details["command"] == "demo-review"


def test_demo_extension_registers_command_tool_prompt_and_hook() -> None:
    from codepilot.extensions import AfterToolCallContext, SessionCommandContext, load_extensions
    from codepilot.protocols import AssistantMessage, ToolCall
    from codepilot.protocols.tool_hooks import ToolHookContextSnapshot

    loaded = load_extensions(ROOT, configured_paths=[str(EXAMPLES / "demo_extension.py")])

    assert loaded.errors == []
    assert "demo-extension" in loaded.commands
    assert [tool.name for tool in loaded.tools] == ["demo_echo"]
    assert loaded.prompt_guidelines
    assert loaded.append_prompts
    assert len(loaded.after_tool_hooks) == 1

    command_output = loaded.commands["demo-extension"].handler(
        SessionCommandContext(
            name="demo-extension",
            args=[],
            raw_text="/demo-extension",
        )
    )
    assert command_output == "Demo extension is loaded."

    tool = loaded.tools[0]
    result = asyncio.run(tool.execute("call_demo", {"text": "hello"}))
    assert result.content[0].text == "hello"
    assert result.details == {"demo_extension": True}

    hook_result = loaded.after_tool_hooks[0](
        AfterToolCallContext(
            assistant_message=AssistantMessage(content=[]),
            tool_call=ToolCall(id="call_demo", name="demo_echo", arguments={}),
            args={"text": "hello"},
            result=result,
            is_error=False,
            context=ToolHookContextSnapshot(system_prompt="", messages=()),
        ),
        None,
    )
    assert hook_result is not None
    assert hook_result.details["demo_after_hook_seen"] is True


def test_demo_mcp_config_creates_proxy_tool() -> None:
    from codepilot.extensions.mcp import create_mcp_proxy_tools, parse_mcp_tool_configs

    raw = json.loads((EXAMPLES / "demo_mcp_config.json").read_text(encoding="utf-8"))
    configs = parse_mcp_tool_configs(raw["mcp_servers"])

    assert len(configs) == 1
    assert configs[0].name == "mcp_demo_echo"
    assert configs[0].server == "demo"
    assert configs[0].tool == "echo"

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
            return {"echo": arguments.get("text", "")}

    client = FakeMCPClient()
    tools = create_mcp_proxy_tools(configs, client=client)

    assert [tool.name for tool in tools] == ["mcp_demo_echo"]
    result = asyncio.run(tools[0].execute("call_mcp", {"text": "hello"}))

    assert client.calls == [("demo", "echo", {"text": "hello"})]
    assert "hello" in result.content[0].text
    assert result.details == {"server": "demo", "tool": "echo"}
