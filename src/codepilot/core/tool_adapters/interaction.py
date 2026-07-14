"""将用户交互工具适配为 Tools 注册定义。"""

from __future__ import annotations

"""Core-owned user interaction tool registration."""

import json
from dataclasses import dataclass

from codepilot.tools.codecs import DataclassCodec, JsonObjectCodec
from codepilot.tools.contracts import ToolRegistration, ToolSpec
from codepilot.tools.results import TextContent
from codepilot.tools.security import (
    ConcurrencyPolicy,
    OutputLimits,
    OutputTrustPolicy,
    TimeoutPolicy,
    ToolAccessRequest,
    ToolAccessResolution,
    ToolPolicy,
)
from codepilot.tools.state import InteractionRequest, build_interaction_request


REQUEST_USER_INPUT_TOOL = "request_user_input"


@dataclass(frozen=True)
class RequestUserInput:
    """模型请求用户补充信息的结构化输入。"""
    prompt: str
    options: tuple[str, ...] = ()
    allow_free_text: bool = True

    def __post_init__(self) -> None:
        prompt = str(self.prompt).strip()
        if not prompt:
            raise ValueError("prompt cannot be empty")
        options = tuple(str(value).strip() for value in self.options)
        if any(not value for value in options):
            raise ValueError("options cannot contain empty values")
        if len(options) != len(set(options)):
            raise ValueError("options must be unique")
        if options and len(options) < 2:
            raise ValueError("options must contain at least two choices")
        if not isinstance(self.allow_free_text, bool):
            raise TypeError("allow_free_text must be bool")
        if not self.allow_free_text and not options:
            raise ValueError("options are required when free text is disabled")
        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "options", options)


def create_interaction_registration() -> ToolRegistration:
    """创建受 Core 等待状态约束的用户交互工具注册。"""
    input_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "minLength": 1},
            "options": {
                "type": "array",
                "minItems": 2,
                "maxItems": 8,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "allow_free_text": {"type": "boolean"},
        },
        "required": ["prompt"],
        "additionalProperties": False,
    }
    output_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "answers": {
                "type": "object",
                "properties": {"answer": {}},
                "required": ["answer"],
                "additionalProperties": False,
            }
        },
        "required": ["answers"],
        "additionalProperties": False,
    }

    class Resolver:
        def resolve(self, input: RequestUserInput, request):
            interaction = build_interaction_request(
                request,
                prompt=input.prompt,
                options=input.options,
                allow_free_text=input.allow_free_text,
            )
            return ToolAccessResolution(
                input=interaction,
                access=ToolAccessRequest(
                    actions=("request_user_input",),
                    resources=(),
                    effects=frozenset(),
                    risk="low",
                    reason="Pause this tool call until structured user input is provided",
                ),
            )

    async def handler(input: InteractionRequest, context) -> InteractionRequest:
        context.cancellation.raise_if_cancelled()
        return input

    class Renderer:
        def render(self, data):
            return (
                TextContent(
                    text="User input received: "
                    + json.dumps(data["answers"], ensure_ascii=False, sort_keys=True)
                ),
            )

    return ToolRegistration(
        version="1.0.0",
        implementation_version="1",
        spec=ToolSpec(
            REQUEST_USER_INPUT_TOOL,
            (
                "Pause the current tool call only when a required user choice or missing value cannot "
                "be inferred safely. Provide a concise prompt and, when applicable, two to eight "
                "mutually exclusive options. This is user input collection, not tool security approval "
                "and not Task Plan approval."
            ),
            input_schema,
            output_schema,
        ),
        category="interaction",
        source="builtin",
        owner="core.interaction",
        policy=ToolPolicy(
            allowed_modes=frozenset({"plan", "execute"}),
            declared_effects=frozenset(),
            required_permissions=frozenset(),
            base_risk="low",
            approval="never",
            timeout=TimeoutPolicy(5_000, 5_000),
            concurrency=ConcurrencyPolicy(mode="serial", group="user_interaction"),
            output_limits=OutputLimits(),
            output_trust=OutputTrustPolicy(),
        ),
        input_codec=DataclassCodec(RequestUserInput, input_schema),
        output_codec=JsonObjectCodec(output_schema),
        handler=handler,
        renderer=Renderer(),
        access_resolver=Resolver(),
    )


__all__ = ["REQUEST_USER_INPUT_TOOL", "RequestUserInput", "create_interaction_registration"]
