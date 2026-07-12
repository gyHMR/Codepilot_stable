from __future__ import annotations

"""Canonical workspace status tool."""

from dataclasses import dataclass
from typing import Any

from ..codecs import DataclassCodec
from ..contracts import ToolExecutionContext, ToolRegistration, ToolSpec
from ..results import TextContent
from ..sandbox import WorkspaceSandbox
from ..security import (
    ConcurrencyPolicy,
    OutputLimits,
    OutputTrustPolicy,
    TimeoutPolicy,
    ToolAccessRequest,
    ToolAccessResolution,
    ToolEffect,
    ToolPolicy,
    ToolResource,
)

_DRAFT = "https://json-schema.org/draft/2020-12/schema"


@dataclass(frozen=True)
class WorkspaceStatusInput:
    include_hidden: bool = False


@dataclass(frozen=True)
class WorkspaceStatusOutput:
    text: str
    details: dict[str, Any]
    metadata: dict[str, Any]


def create_workspace_status_registration(sandbox: WorkspaceSandbox) -> ToolRegistration:
    input_schema = {
        "$schema": _DRAFT,
        "type": "object",
        "properties": {"include_hidden": {"type": "boolean", "default": False}},
        "additionalProperties": False,
    }
    output_schema = {
        "$schema": _DRAFT,
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "details": {"type": "object"},
            "metadata": {"type": "object"},
        },
        "required": ["text", "details", "metadata"],
        "additionalProperties": False,
    }

    class Resolver:
        def resolve(self, input, request):
            _ = request
            return ToolAccessResolution(
                input=input,
                access=ToolAccessRequest(
                    actions=("workspace_status",),
                    resources=(ToolResource("workspace:///"),),
                    effects=frozenset({"filesystem_read"}),
                    risk="low",
                    reason="Inspect workspace root status",
                ),
            )

    async def handler(input: WorkspaceStatusInput, context: ToolExecutionContext):
        context.cancellation.raise_if_cancelled()
        entries = sorted(
            item.name
            for item in sandbox.root.iterdir()
            if input.include_hidden or not item.name.startswith(".")
        )
        details = {
            "workspace": str(sandbox.root),
            "entry_count": len(entries),
            "entries": entries[:100],
            "is_git_repository": (sandbox.root / ".git").exists(),
        }
        context.effects.report(
            ToolEffect(
                kind="filesystem_read",
                resource=ToolResource("workspace:///"),
                operation="inspect workspace status",
                status="completed",
                certainty="observed",
            )
        )
        return WorkspaceStatusOutput(
            text=(
                f"Workspace: {sandbox.root}\n"
                f"Entries: {len(entries)}\n"
                f"Git repository: {details['is_git_repository']}"
            ),
            details=details,
            metadata={},
        )

    class Renderer:
        def render(self, data):
            return (TextContent(text=data["text"]),)

    return ToolRegistration(
        version="1.0.0",
        implementation_version="2",
        spec=ToolSpec(
            "workspace_status",
            "Summarize the workspace root, visible entries, and Git repository presence.",
            input_schema,
            output_schema,
        ),
        category="filesystem",
        source="builtin",
        owner="codepilot.builtin",
        policy=ToolPolicy(
            allowed_modes=frozenset({"plan", "execute"}),
            declared_effects=frozenset({"filesystem_read"}),
            required_permissions=frozenset({"workspace.read"}),
            base_risk="low",
            approval="never",
            timeout=TimeoutPolicy(5_000, 10_000),
            concurrency=ConcurrencyPolicy(mode="parallel"),
            output_limits=OutputLimits(),
            output_trust=OutputTrustPolicy(),
        ),
        input_codec=DataclassCodec(WorkspaceStatusInput, input_schema),
        output_codec=DataclassCodec(WorkspaceStatusOutput, output_schema),
        handler=handler,
        renderer=Renderer(),
        access_resolver=Resolver(),
    )


__all__ = ["create_workspace_status_registration"]
