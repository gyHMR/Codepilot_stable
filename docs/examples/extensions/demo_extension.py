from __future__ import annotations

from dataclasses import dataclass

from codepilot.tools import (
    ConcurrencyPolicy,
    DataclassCodec,
    OutputLimits,
    OutputTrustPolicy,
    TextContent,
    TimeoutPolicy,
    ToolAccessRequest,
    ToolAccessResolution,
    ToolPolicy,
    ToolRegistration,
    ToolSpec,
)


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
    api.register_tool(_demo_registration())


def _demo_command(ctx):
    _ = ctx
    return "Demo extension is loaded."


@dataclass(frozen=True)
class DemoInput:
    text: str


@dataclass(frozen=True)
class DemoOutput:
    text: str
    demo_extension: bool = True


def _demo_registration() -> ToolRegistration:
    input_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }
    output_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "demo_extension": {"type": "boolean"},
        },
        "required": ["text", "demo_extension"],
        "additionalProperties": False,
    }

    class Resolver:
        def resolve(self, input, request):
            _ = request
            return ToolAccessResolution(
                input=input,
                access=ToolAccessRequest(
                    actions=("demo.echo",),
                    resources=(),
                    effects=frozenset(),
                    risk="low",
                    reason="Return the provided demo text",
                ),
            )

    async def handler(input: DemoInput, context) -> DemoOutput:
        context.cancellation.raise_if_cancelled()
        return DemoOutput(text=input.text)

    class Renderer:
        def render(self, data):
            return (TextContent(text=data["text"]),)

    input_codec = DataclassCodec(DemoInput, input_schema)
    output_codec = DataclassCodec(DemoOutput, output_schema)
    return ToolRegistration(
        version="1.0.0",
        implementation_version="1",
        spec=ToolSpec(
            "demo_echo",
            "Return the provided text for the extension demonstration.",
            input_schema,
            output_schema,
        ),
        category="external",
        source="extension",
        owner="extension:demo",
        policy=ToolPolicy(
            allowed_modes=frozenset({"plan", "execute"}),
            declared_effects=frozenset(),
            required_permissions=frozenset(),
            base_risk="low",
            approval="never",
            timeout=TimeoutPolicy(5_000, 5_000),
            concurrency=ConcurrencyPolicy(mode="parallel"),
            output_limits=OutputLimits(),
            output_trust=OutputTrustPolicy(),
        ),
        input_codec=input_codec,
        output_codec=output_codec,
        handler=handler,
        renderer=Renderer(),
        access_resolver=Resolver(),
    )
