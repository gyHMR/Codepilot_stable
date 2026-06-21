from __future__ import annotations

from codepilot.core import AfterToolCallResult
from codepilot.protocols import TextContent
from codepilot.tools import AgentTool, AgentToolResult


def register(api):
    api.add_prompt_guideline(
        "Demo extension loaded: keep extension behavior visible and minimal."
    )
    api.append_system_prompt(
        "## Demo Extension\n"
        "This section is added by docs/examples/extensions/demo_extension.py."
    )
    api.register_command(
        "demo-extension",
        _demo_command,
        description="Show that a Python extension command is available.",
    )
    api.register_tool(
        AgentTool(
            name="demo_echo",
            label="Demo Echo",
            description="Return the provided text. This demonstrates extension tools.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            execute=_demo_echo,
        )
    )
    api.on_after_tool_call(_mark_demo_tool_result)


def _demo_command(ctx):
    _ = ctx
    return "Demo extension is loaded."


async def _demo_echo(tool_call_id, params, signal=None, on_update=None):
    _ = tool_call_id, signal, on_update
    text = str(params.get("text", ""))
    return AgentToolResult(
        content=[TextContent(text=text)],
        details={"demo_extension": True},
    )


def _mark_demo_tool_result(ctx, signal=None):
    _ = signal
    if ctx.tool_call.name != "demo_echo":
        return None
    details = dict(ctx.result.details)
    details["demo_after_hook_seen"] = True
    return AfterToolCallResult(details=details)
