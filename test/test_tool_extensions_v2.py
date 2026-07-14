from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from tool_runtime_testkit import execute_tool


def test_mcp_config_rejects_removed_legacy_field_aliases() -> None:
    from codepilot.extensions.mcp import parse_mcp_server_configs

    with pytest.raises(ValueError, match="unknown fields: tools"):
        parse_mcp_server_configs(
            [
                {
                    "name": "demo",
                    "url": "https://example.test/mcp",
                    "allow_tools": ["echo"],
                    "tools": [],
                }
            ]
        )

    with pytest.raises(ValueError, match="requires a non-empty allow_tools"):
        parse_mcp_server_configs(
            [
                {
                    "name": "demo",
                    "url": "https://example.test/mcp",
                    "allow_tools": [],
                }
            ]
        )


def test_registry_register_batch_is_atomic_and_owner_scoped() -> None:
    from codepilot.tools.registry import ToolRegistrationConflictError, ToolRegistry

    registry = ToolRegistry()
    first = _registration("extension_first", owner="extension:demo")
    second = _registration("extension_second", owner="extension:demo")

    registration_ids = registry.register_batch(
        (first, second),
        owner="extension:demo",
    )

    assert len(registration_ids) == 2
    assert registry.entry("extension_first") is not None
    assert registry.entry("extension_second") is not None

    with pytest.raises(ToolRegistrationConflictError):
        registry.register_batch(
            (
                _registration("extension_third", owner="extension:other"),
                _registration("extension_first", owner="extension:other"),
            ),
            owner="extension:other",
        )

    assert registry.entry("extension_third") is None
    assert registry.unregister_owner("extension:demo") == 2
    assert registry.entry("extension_first") is None
    assert registry.entry("extension_second") is None


def test_extension_api_accepts_only_canonical_registrations_and_binds_owner() -> None:
    from codepilot.extensions.api import ExtensionAPI

    api = ExtensionAPI(owner="extension:/workspace/demo.py")
    api.register_tool(_registration("demo_echo", owner="caller", source="caller"))
    loaded = api.snapshot()

    assert len(loaded.tools) == 1
    assert loaded.tools[0].source == "extension"
    assert loaded.tools[0].owner == "extension:/workspace/demo.py"

    with pytest.raises(TypeError, match="ToolRegistration"):
        api.register_tool(object())  # type: ignore[arg-type]


def test_skill_loader_produces_canonical_registration_executed_by_runtime() -> None:
    from pathlib import Path

    from codepilot.extensions import load_skills

    root = Path(__file__).resolve().parents[1]
    skill = root / "docs" / "examples" / "extensions" / "demo-review"
    loaded = load_skills(root, configured_paths=[str(skill)])

    assert loaded.errors == []
    assert [item.spec.name for item in loaded.tools] == ["load_skill", "read_skill_resource"]
    registration = loaded.tools[0]
    assert registration.source == "skill"
    assert registration.category == "external"

    runtime, ids = _runtime((registration,))
    result = asyncio.run(
        execute_tool(
            runtime,
            _request(
                "load_skill",
                ids["load_skill"],
                mode="plan",
                arguments={"name": "demo-review"},
            )
        )
    )

    assert result.status == "success"
    assert result.data["skill"] == "demo-review"
    assert "Goal:" in result.content[0].text


def test_mcp_adapter_builds_canonical_registration_with_policy_and_output_validation() -> None:
    from codepilot.extensions.mcp import (
        MCPAuthConfig,
        MCPRemoteTool,
        MCPServerConfig,
        create_mcp_registrations,
    )

    class Client:
        async def call_tool(self, server, tool, arguments):
            assert (server, tool) == ("demo", "echo")
            return {
                "structuredContent": {"echo": arguments["text"]},
                "content": [{"type": "text", "text": arguments["text"]}],
            }

    config = MCPServerConfig(
        name="demo",
        url="https://example.test/mcp",
        auth=MCPAuthConfig("bearer_env", "DEMO_TOKEN", "demo-token"),
        allow_tools=frozenset({"echo"}),
        timeout_ms=250,
        max_parallel=1,
        tool_policies={
            "echo": {"read_only": True, "requires_approval": False},
        },
    )
    remote = MCPRemoteTool(
        name="echo",
        description="Echo one value from the demo MCP server.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"echo": {"type": "string"}},
            "required": ["echo"],
            "additionalProperties": False,
        },
    )
    registrations = create_mcp_registrations(config, (remote,), client=Client())

    assert [item.spec.name for item in registrations] == ["mcp__demo__echo"]
    registration = registrations[0]
    assert registration.source == "mcp"
    assert registration.owner == "mcp:demo"
    assert registration.policy.approval == "never"
    assert registration.policy.timeout.default_execution_ms == 250
    assert registration.policy.concurrency.group == "mcp:demo"
    assert registration.policy.concurrency.max_parallel == 1
    assert registration.policy.output_trust.default_content_trust == "untrusted"
    assert "credential:demo-token" in registration.policy.required_permissions
    assert registration.policy.declared_effects == frozenset(
        {"network_access", "external_state_read", "credential_access"}
    )

    runtime, ids = _runtime(registrations)
    result = asyncio.run(
        execute_tool(
            runtime,
            _request(
                "mcp__demo__echo",
                ids["mcp__demo__echo"],
                mode="execute",
                arguments={"text": "hello"},
            )
        )
    )

    assert result.status == "success"
    assert result.data["structured_output"]["echo"] == "hello"
    assert result.output_validation == "schema_validated"
    assert result.content_trust == "untrusted"
    assert {effect.kind for effect in result.effects} == {
        "network_access",
        "external_state_read",
        "credential_access",
    }


def test_mcp_unverified_output_uses_artifact_renderer_and_server_concurrency_limit() -> None:
    from codepilot.extensions.mcp import MCPRemoteTool, MCPServerConfig, create_mcp_registrations
    from codepilot.tools.results import ArtifactContent

    class Client:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0

        async def call_tool(self, server, tool, arguments):
            _ = server, tool
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.02)
            self.active -= 1
            return {
                "content": [
                    {
                        "type": "artifact",
                        "artifact_id": f"asset-{arguments['id']}",
                        "mime_type": "text/plain",
                        "size_bytes": 12,
                    }
                ]
            }

    client = Client()
    config = MCPServerConfig(
        name="assets",
        url="https://example.test/mcp",
        auth=None,
        allow_tools=frozenset({"fetch"}),
        max_parallel=1,
        tool_policies={"fetch": {"read_only": True, "requires_approval": False}},
    )
    remote = MCPRemoteTool(
        name="fetch",
        description="Fetch one opaque artifact reference.",
        input_schema={
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
            "additionalProperties": False,
        },
    )
    registrations = create_mcp_registrations(config, (remote,), client=client)
    runtime, ids = _runtime(registrations)
    requests = (
        _request(
            "mcp__assets__fetch",
            ids["mcp__assets__fetch"],
            mode="execute",
            arguments={"id": "a"},
            call_id="call-fetch-a",
        ),
        _request(
            "mcp__assets__fetch",
            ids["mcp__assets__fetch"],
            mode="execute",
            arguments={"id": "b"},
            call_id="call-fetch-b",
        ),
    )

    results = asyncio.run(runtime.execute_batch(requests))

    assert client.max_active == 1
    assert all(item.output_validation == "structurally_validated" for item in results)
    assert all(isinstance(item.content[0], ArtifactContent) for item in results)
    assert [item.content[0].artifact.artifact_id for item in results] == ["asset-a", "asset-b"]


def test_mcp_unknown_side_effect_defaults_to_external_write_and_approval() -> None:
    from codepilot.extensions.mcp import MCPRemoteTool, MCPServerConfig, create_mcp_registrations

    config = MCPServerConfig(
        name="remote",
        url="https://example.test/mcp",
        auth=None,
        allow_tools=frozenset({"change_state"}),
    )
    remote = MCPRemoteTool(
        name="change_state",
        description="Call an MCP operation with unspecified side effects.",
        input_schema={"type": "object", "properties": {}},
    )
    registration = create_mcp_registrations(config, (remote,), client=None)[0]

    assert registration.policy.approval == "always"
    assert registration.policy.base_risk == "medium"
    assert registration.policy.declared_effects == frozenset(
        {"network_access", "external_state_write"}
    )
    assert registration.policy.allowed_modes == frozenset({"execute"})


@dataclass(frozen=True)
class _Input:
    text: str


@dataclass(frozen=True)
class _Output:
    text: str


def _registration(name, *, owner, source="extension"):
    from codepilot.tools.codecs import DataclassCodec
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

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }

    class Resolver:
        def resolve(self, input, request):
            _ = request
            return ToolAccessResolution(
                input=input,
                access=ToolAccessRequest(
                    actions=(name,),
                    resources=(),
                    effects=frozenset(),
                    risk="low",
                    reason="extension test",
                ),
            )

    async def handler(input, context):
        _ = context
        return _Output(input.text)

    class Renderer:
        def render(self, data):
            return (TextContent(text=data["text"]),)

    input_codec = DataclassCodec(_Input, schema)
    output_codec = DataclassCodec(_Output, schema)
    return ToolRegistration(
        version="1.0.0",
        implementation_version="1",
        spec=ToolSpec(name, f"Canonical extension tool {name}.", schema, schema),
        category="external",
        source=source,
        owner=owner,
        policy=ToolPolicy(
            allowed_modes=frozenset({"plan", "execute"}),
            declared_effects=frozenset(),
            required_permissions=frozenset(),
            base_risk="low",
            approval="never",
            timeout=TimeoutPolicy(500, 500),
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


def _runtime(registrations):
    from codepilot.tools.registry import ToolRegistry
    from codepilot.tools.runtime import ToolRuntime

    registry = ToolRegistry()
    ids = {item.spec.name: registry.register(item) for item in registrations}
    return ToolRuntime(registry), ids


def _request(name, registration_id, *, mode, arguments, call_id=None):
    from codepilot.tools.contracts import ToolExecutionRequest

    return ToolExecutionRequest(
        run_id="run-extension-v2",
        session_id="session-extension-v2",
        tool_call_id=call_id or f"call-{name}",
        tool_name=name,
        arguments=arguments,
        mode=mode,
        registration_id=registration_id,
    )
