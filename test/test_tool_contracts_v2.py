from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest


def _policy():
    from codepilot.tools.security import (
        ConcurrencyPolicy,
        OutputLimits,
        OutputTrustPolicy,
        TimeoutPolicy,
        ToolPolicy,
    )

    return ToolPolicy(
        allowed_modes=frozenset({"execute"}),
        declared_effects=frozenset({"filesystem_read"}),
        required_permissions=frozenset({"workspace.read"}),
        base_risk="low",
        approval="never",
        timeout=TimeoutPolicy(default_execution_ms=5_000, max_execution_ms=10_000),
        concurrency=ConcurrencyPolicy(mode="parallel"),
        output_limits=OutputLimits(max_data_bytes=8_192, max_content_bytes=16_384),
        output_trust=OutputTrustPolicy(default_content_trust="trusted"),
    )


def test_canonical_tool_contracts_are_immutable_and_explicit() -> None:
    from codepilot.tools.contracts import (
        ToolExecutionRequest,
        ToolRegistration,
        ToolSpec,
    )
    from codepilot.tools.security import ToolAccessRequest, ToolAccessResolution, ToolResource

    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    spec = ToolSpec(
        name="read",
        description="Read a UTF-8 text file from the workspace.",
        input_schema=input_schema,
        output_schema={"type": "object"},
    )
    input_schema["properties"] = {}

    assert "path" in spec.input_schema["properties"]
    with pytest.raises(ValueError, match="tool name"):
        ToolSpec(name="bad name", description="Invalid name.", input_schema={})

    arguments = {"path": "src/app.py"}
    request = ToolExecutionRequest(
        run_id="run-1",
        session_id="session-1",
        tool_call_id="call-1",
        tool_name="read",
        arguments=arguments,
        mode="execute",
        registration_id="reg-1",
    )
    arguments["path"] = "changed.py"

    assert request.arguments == {"path": "src/app.py"}
    with pytest.raises(FrozenInstanceError):
        request.tool_name = "write"  # type: ignore[misc]

    class Codec:
        json_schema = {"type": "object"}

        def decode(self, value):
            return value

        def encode(self, value):
            return value

    async def handler(input, context):
        return input

    class Renderer:
        def render(self, data):
            return ()

    class Resolver:
        def resolve(self, input, context):
            return ToolAccessResolution(
                input=input,
                access=ToolAccessRequest(
                    actions=("read",),
                    resources=(ToolResource("workspace:///src/app.py"),),
                    effects=frozenset({"filesystem_read"}),
                    risk="low",
                    reason="Read workspace file",
                ),
            )

    registration = ToolRegistration(
        version="1.0.0",
        implementation_version="1",
        spec=spec,
        category="filesystem",
        source="builtin",
        owner="codepilot.builtin",
        policy=_policy(),
        input_codec=Codec(),
        output_codec=Codec(),
        handler=handler,
        renderer=Renderer(),
        access_resolver=Resolver(),
    )

    assert registration.spec is spec
    assert not hasattr(registration, "registration_id")


def test_security_values_capture_authorized_resources_and_effects() -> None:
    from codepilot.tools.security import (
        ToolAccessRequest,
        ToolAccessResolution,
        ToolEffect,
        ToolResource,
    )

    preview = {"path": "src/app.py"}
    resource = ToolResource("workspace:///src/app.py")
    effect = ToolEffect(
        kind="filesystem_write",
        resource=resource,
        operation="write",
        status="completed",
        certainty="observed",
    )
    access = ToolAccessRequest(
        actions=("write",),
        resources=(resource,),
        effects=frozenset({"filesystem_write"}),
        risk="medium",
        reason="Update workspace file",
        safe_preview=preview,
    )
    preview["path"] = "changed.py"
    resolution = ToolAccessResolution(input={"path": "src/app.py"}, access=access)

    assert effect.resource.uri == "workspace:///src/app.py"
    assert access.safe_preview == {"path": "src/app.py"}
    assert resolution.input == {"path": "src/app.py"}


def test_canonical_result_enforces_invariants_and_projects_one_way() -> None:
    from codepilot.tools.results import (
        TextContent,
        ToolError,
        ToolResult,
        to_tool_result_message,
    )
    from codepilot.tools.security import ToolEffect, ToolResource

    error = ToolError(
        code="tool.execution.timeout",
        kind="execution_timeout",
        message="Tool execution timed out",
        retryable=True,
    )
    result = ToolResult(
        tool_call_id="call-1",
        tool_name="write",
        status="timed_out",
        content=(TextContent(text="partial output"),),
        data={"written_bytes": 4},
        error=error,
        effects=(
            ToolEffect(
                kind="filesystem_write",
                resource=ToolResource("workspace:///src/app.py"),
                operation="write",
                status="partial",
                certainty="observed",
            ),
        ),
        registration_id="reg-1",
    )

    message = to_tool_result_message(result)

    assert message.status == "timed_out"
    assert message.is_error is True
    assert message.error_code == "tool.execution.timeout"
    assert message.content[0].text == "partial output"
    assert message.affected_paths == ["workspace:///src/app.py"]
    assert message.workspace_changed is True
    assert "canonical_status" not in message.metadata

    with pytest.raises(ValueError, match="success result"):
        ToolResult(
            tool_call_id="call-2",
            tool_name="read",
            status="success",
            error=error,
            registration_id="reg-1",
        )

    with pytest.raises(ValueError, match="success result"):
        ToolResult(
            tool_call_id="call-2b",
            tool_name="read",
            status="success",
            approval={"approval_id": "approval-1"},
            registration_id="reg-1",
        )

    with pytest.raises(ValueError, match="user input suspension"):
        to_tool_result_message(
            ToolResult(
                tool_call_id="call-3",
                tool_name="ask",
                status="user_input_required",
                interaction={"interaction_id": "input-1"},
                registration_id="reg-1",
            )
        )
