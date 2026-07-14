from __future__ import annotations

import asyncio
import json

import httpx

from tool_runtime_testkit import execute_tool


def test_streamable_http_transport_initializes_discovers_calls_and_closes() -> None:
    from codepilot.extensions.mcp import MCPServerConfig, StreamableHttpTransport

    requests: list[tuple[str, dict[str, object], dict[str, str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        headers = {key.lower(): value for key, value in request.headers.items()}
        if request.method == "DELETE":
            assert headers["mcp-session-id"] == "session-1"
            return httpx.Response(204)
        payload = json.loads(request.content)
        requests.append((payload["method"], payload, headers))
        if payload["method"] == "notifications/initialized":
            return httpx.Response(202)
        if payload["method"] == "initialize":
            result = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "demo", "version": "1"},
            }
        elif payload["method"] == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo text.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                        "annotations": {"readOnlyHint": True},
                    }
                ]
            }
        else:
            result = {"content": [{"type": "text", "text": "hello"}]}
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                "MCP-Session-Id": "session-1",
            },
            json={"jsonrpc": "2.0", "id": payload["id"], "result": result},
        )

    async def run_case():
        config = MCPServerConfig(
            name="demo",
            url="https://example.test/mcp",
            auth=None,
            allow_tools=frozenset({"echo"}),
        )
        transport = StreamableHttpTransport(
            config,
            token="secret-token",
            http_transport=httpx.MockTransport(handler),
        )
        await transport.initialize()
        tools = await transport.list_tools()
        result = await transport.call_tool("echo", {"text": "hello"})
        await transport.aclose()
        return tools, result

    tools, result = asyncio.run(run_case())

    assert [tool.name for tool in tools] == ["echo"]
    assert result["content"][0]["text"] == "hello"
    assert [item[0] for item in requests] == [
        "initialize",
        "notifications/initialized",
        "tools/list",
        "tools/call",
    ]
    assert all(item[2]["authorization"] == "Bearer secret-token" for item in requests)
    assert "secret-token" not in str([item[1] for item in requests])


def test_mcp_manager_discovers_allowlisted_tools_and_registers_them(monkeypatch) -> None:
    from codepilot.extensions.mcp import MCPRemoteTool, create_mcp_manager
    from codepilot.tools.registry import ToolRegistry
    from codepilot.tools.runtime import ToolRuntime

    created: list[tuple[object, str | None]] = []

    class FakeTransport:
        diagnostics = ()

        def __init__(self) -> None:
            self.closed = False

        async def initialize(self) -> None:
            return None

        async def list_tools(self):
            return (
                MCPRemoteTool(
                    name="allowed",
                    description="Read one remote value.",
                    input_schema={"type": "object", "properties": {}},
                    annotations={"readOnlyHint": True},
                ),
                MCPRemoteTool(
                    name="blocked",
                    description="Must not be registered.",
                    input_schema={"type": "object", "properties": {}},
                ),
            )

        async def call_tool(self, tool, arguments):
            assert tool == "allowed"
            assert arguments == {}
            return {"content": [{"type": "text", "text": "remote value"}]}

        async def aclose(self) -> None:
            self.closed = True

    def factory(config, token):
        created.append((config, token))
        return FakeTransport()

    monkeypatch.setenv("DEMO_MCP_TOKEN", "top-secret")
    manager = create_mcp_manager(
        [
            {
                "name": "demo",
                "transport": "streamable_http",
                "url": "https://example.test/mcp",
                "auth": {"type": "bearer_env", "env": "DEMO_MCP_TOKEN"},
                "allow_tools": ["allowed"],
            }
        ],
        transport_factory=factory,
    )
    registry = ToolRegistry()

    async def run_case():
        await manager.ensure_ready(registry)
        entry = registry.entry("mcp__demo__allowed")
        assert entry is not None
        result = await execute_tool(
            ToolRuntime(registry),
            _request("mcp__demo__allowed", entry.registration_id),
        )
        await manager.aclose()
        return result

    result = asyncio.run(run_case())

    assert registry.entry("mcp__demo__blocked") is None
    assert result.status == "success"
    assert result.content[0].text == "remote value"
    assert created[0][1] == "top-secret"
    assert "top-secret" not in str(manager.diagnostics)


def test_mcp_config_requires_https_and_rejects_static_tool_catalog() -> None:
    import pytest

    from codepilot.extensions.mcp import parse_mcp_server_configs

    with pytest.raises(ValueError, match="must use HTTPS"):
        parse_mcp_server_configs(
            [{"name": "bad", "url": "http://example.com/mcp", "allow_tools": ["x"]}]
        )
    with pytest.raises(ValueError, match="cannot contain credentials"):
        parse_mcp_server_configs(
            [
                {
                    "name": "bad",
                    "url": "https://token@example.com/mcp",
                    "allow_tools": ["x"],
                }
            ]
        )
    with pytest.raises(ValueError, match="unknown fields: tools"):
        parse_mcp_server_configs(
            [
                {
                    "name": "old",
                    "url": "https://example.test/mcp",
                    "allow_tools": ["x"],
                    "tools": [],
                }
            ]
        )


def test_runtime_discovers_mcp_tools_before_first_model_request_and_closes_transport(
    tmp_path,
) -> None:
    from codepilot.llm.ports import LLMCompleted
    from codepilot.protocols import AssistantMessage, Model, TextContent
    from codepilot.runtime import SessionOpenIntent
    from codepilot.runtime.actions import PromptSubmitted
    from codepilot.runtime.gateway import RuntimeGateway

    transports = []
    seen_tools: list[str] = []

    class FakeTransport:
        diagnostics = ()

        def __init__(self) -> None:
            self.closed = False

        async def initialize(self) -> None:
            return None

        async def list_tools(self):
            from codepilot.extensions.mcp import MCPRemoteTool

            return (
                MCPRemoteTool(
                    name="repository_info",
                    description="Read repository information.",
                    input_schema={"type": "object", "properties": {}},
                    annotations={"readOnlyHint": True},
                ),
            )

        async def call_tool(self, tool, arguments):
            raise AssertionError("model should not call the discovered tool in this test")

        async def aclose(self) -> None:
            self.closed = True

    def factory(config, token):
        _ = config, token
        transport = FakeTransport()
        transports.append(transport)
        return transport

    class ModelPort:
        async def stream(self, request):
            seen_tools.extend(tool.name for tool in request.tools)
            yield LLMCompleted(
                message=AssistantMessage(content=[TextContent(text="done")])
            )

    model = Model(
        id="mcp-runtime-test",
        name="MCP Runtime Test",
        api="unit-test",
        provider="unit-test",
        base_url="",
        reasoning=False,
        input=["text"],
        context_window=32_000,
        max_tokens=500,
    )

    async def run_case():
        gateway = RuntimeGateway(model_port=ModelPort())
        opened = gateway.open_session(
            SessionOpenIntent(
                workspace_dir=tmp_path,
                model=model,
                memory_enabled=False,
                mcp_servers=[
                    {
                        "name": "github",
                        "url": "https://example.test/mcp",
                        "allow_tools": ["repository_info"],
                    }
                ],
                mcp_transport_factory=factory,
            )
        )
        _ = [
            frame
            async for frame in gateway.dispatch(
                opened.session_id,
                PromptSubmitted(text="inspect repository"),
            )
        ]
        await gateway.close_all()

    asyncio.run(run_case())

    assert "mcp__github__repository_info" in seen_tools
    assert transports and transports[0].closed is True


def _request(name: str, registration_id: str):
    from codepilot.tools.contracts import ToolExecutionRequest

    return ToolExecutionRequest(
        run_id="run-mcp",
        session_id="session-mcp",
        tool_call_id="call-mcp",
        tool_name=name,
        arguments={},
        mode="execute",
        registration_id=registration_id,
    )
