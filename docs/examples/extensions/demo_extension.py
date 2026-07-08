from __future__ import annotations

from codepilot.extensions import AfterToolCallResult
from codepilot.protocols import TextContent
from codepilot.tools import ToolCallRequest, ToolDefinition, ToolMetadata, ToolResult


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
        ToolDefinition(
            name="demo_echo",
            label="Demo Echo",
            description="Return the provided text. This demonstrates extension tools.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            metadata=ToolMetadata(
                name="demo_echo",
                category="extension",
                read_only=True,
                concurrency_safe=True,
                exclusive=False,
                requires_approval=False,
                risk_level="low",
                scopes=("read", "plan", "build"),
                extra={"capabilities": ["demo.echo"]},
            ),
            execute=_demo_echo,
        )
    )
    api.on_after_tool_call(_mark_demo_tool_result)


def _demo_command(ctx):
    _ = ctx
    return "Demo extension is loaded."


async def _demo_echo(request: ToolCallRequest, signal=None, on_update=None):
    _ = signal, on_update
    text = str(request.arguments.get("text", ""))
    return ToolResult(
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
