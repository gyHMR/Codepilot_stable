from __future__ import annotations

"""MCP server configuration, lazy discovery and client lifecycle."""

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from codepilot.tools.registry import ToolRegistry

from .transport import (
    MCPAuthConfig,
    MCPRemoteTool,
    MCPServerConfig,
    MCPTransport,
    MCPTransportError,
    StreamableHttpTransport,
    validate_server_url,
)


TransportFactory = Callable[[MCPServerConfig, str | None], MCPTransport]
_SERVER_KEYS = frozenset(
    {
        "name",
        "transport",
        "url",
        "auth",
        "allow_tools",
        "timeout_ms",
        "max_parallel",
        "max_output_bytes",
        "max_image_bytes",
        "tool_policies",
    }
)
_TOOL_POLICY_KEYS = frozenset(
    {"read_only", "risk_level", "requires_approval", "output_trust"}
)


@dataclass
class _ServerRuntime:
    config: MCPServerConfig
    state: str = "disconnected"
    transport: MCPTransport | None = None
    tools: tuple[MCPRemoteTool, ...] = ()


class MCPManager:
    """Initialize configured MCP servers once and expose discovered tool calls."""

    def __init__(
        self,
        configs: tuple[MCPServerConfig, ...],
        *,
        transport_factory: TransportFactory | None = None,
    ) -> None:
        self._servers = {config.name: _ServerRuntime(config) for config in configs}
        self._transport_factory = transport_factory or _default_transport_factory
        self._lock = asyncio.Lock()
        self._ready = False
        self._diagnostics: list[str] = []

    @property
    def diagnostics(self) -> tuple[str, ...]:
        return tuple(self._diagnostics)

    @property
    def configured(self) -> bool:
        return bool(self._servers)

    async def ensure_ready(self, registry: ToolRegistry) -> None:
        if self._ready:
            return
        async with self._lock:
            if self._ready:
                return
            for server in self._servers.values():
                await self._initialize_server(server, registry)
            self._ready = True

    async def call_tool(
        self,
        server: str,
        tool: str,
        arguments: dict[str, Any],
    ) -> object:
        runtime = self._servers.get(server)
        if runtime is None or runtime.state != "ready" or runtime.transport is None:
            raise MCPTransportError(
                "mcp.client.server_unavailable",
                f"MCP server '{server}' is unavailable",
            )
        return await runtime.transport.call_tool(tool, arguments)

    async def aclose(self) -> None:
        transports = [
            server.transport
            for server in self._servers.values()
            if server.transport is not None
        ]
        if transports:
            await asyncio.gather(
                *(transport.aclose() for transport in transports),
                return_exceptions=True,
            )
        for server in self._servers.values():
            server.transport = None
            server.state = "closed"
        self._ready = False

    async def _initialize_server(
        self,
        server: _ServerRuntime,
        registry: ToolRegistry,
    ) -> None:
        from .adapter import create_mcp_registrations

        token = _credential(server.config)
        if server.config.auth is not None and token is None:
            server.state = "failed"
            self._diagnostics.append(
                f"mcp:{server.config.name}: credential environment variable "
                f"'{server.config.auth.env}' is not set"
            )
            return
        server.state = "connecting"
        try:
            transport, tools = await self._connect_and_discover(server.config, token)
        except MCPTransportError as exc:
            server.state = "failed"
            self._diagnostics.append(f"mcp:{server.config.name}: {exc.code}: {exc.message}")
            return
        server.transport = transport
        server.tools = tuple(
            tool for tool in tools if tool.name in server.config.allow_tools
        )
        server.state = "ready"
        for tool in server.tools:
            try:
                registration = create_mcp_registrations(
                    server.config,
                    (tool,),
                    client=self,
                )[0]
                registry.register(registration)
            except Exception as exc:
                self._diagnostics.append(
                    f"mcp:{server.config.name}: tool '{tool.name}' "
                    f"registration failed: {exc}"
                )
        for diagnostic in getattr(transport, "diagnostics", ()):
            self._diagnostics.append(f"mcp:{server.config.name}: {diagnostic}")

    async def _connect_and_discover(
        self,
        config: MCPServerConfig,
        token: str | None,
    ) -> tuple[MCPTransport, tuple[MCPRemoteTool, ...]]:
        last_error: MCPTransportError | None = None
        for attempt in range(2):
            transport = self._transport_factory(config, token)
            try:
                await transport.initialize()
                return transport, await transport.list_tools()
            except MCPTransportError as exc:
                last_error = exc
                await transport.aclose()
                if not exc.retryable or attempt == 1:
                    raise
                await asyncio.sleep(0.15)
        assert last_error is not None
        raise last_error


def parse_mcp_server_configs(
    raw_servers: list[dict[str, Any]] | None,
) -> tuple[MCPServerConfig, ...]:
    if not raw_servers:
        return ()
    configs: list[MCPServerConfig] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_servers):
        if not isinstance(raw, Mapping):
            raise ValueError(f"mcp_servers[{index}] must be an object")
        unknown = sorted(set(raw) - _SERVER_KEYS)
        if unknown:
            raise ValueError(
                f"mcp_servers[{index}] has unknown fields: {', '.join(unknown)}"
            )
        name = _required_text(raw.get("name"), f"mcp_servers[{index}].name")
        if name in names:
            raise ValueError(f"Duplicate MCP server name: {name}")
        names.add(name)
        transport = raw.get("transport", "streamable_http")
        if transport != "streamable_http":
            raise ValueError("Only MCP streamable_http transport is supported")
        url = validate_server_url(
            _required_text(raw.get("url"), f"mcp_servers[{index}].url")
        )
        allow_tools = _string_set(raw.get("allow_tools"), "allow_tools")
        if not allow_tools:
            raise ValueError(f"MCP server '{name}' requires a non-empty allow_tools list")
        auth = _auth_config(raw.get("auth"), server=name)
        policies = _tool_policies(raw.get("tool_policies"), server=name)
        unknown_policies = sorted(set(policies) - allow_tools)
        if unknown_policies:
            raise ValueError(
                f"MCP server '{name}' has policies for non-allowlisted tools: "
                + ", ".join(unknown_policies)
            )
        configs.append(
            MCPServerConfig(
                name=name,
                url=url,
                auth=auth,
                allow_tools=frozenset(allow_tools),
                timeout_ms=_bounded_int(raw.get("timeout_ms"), 30_000, 100, 120_000),
                max_parallel=_bounded_int(raw.get("max_parallel"), 4, 1, 16),
                max_output_bytes=_bounded_int(
                    raw.get("max_output_bytes"), 1_000_000, 1_024, 8_000_000
                ),
                max_image_bytes=_bounded_int(
                    raw.get("max_image_bytes"), 4_000_000, 1_024, 16_000_000
                ),
                tool_policies=policies,
            )
        )
    return tuple(configs)


def create_mcp_manager(
    raw_servers: list[dict[str, Any]] | None,
    *,
    transport_factory: TransportFactory | None = None,
) -> MCPManager:
    return MCPManager(
        parse_mcp_server_configs(raw_servers),
        transport_factory=transport_factory,
    )


def _default_transport_factory(
    config: MCPServerConfig,
    token: str | None,
) -> MCPTransport:
    return StreamableHttpTransport(config, token=token)


def _credential(config: MCPServerConfig) -> str | None:
    if config.auth is None:
        return None
    value = os.getenv(config.auth.env)
    return value if value else None


def _auth_config(value: object, *, server: str) -> MCPAuthConfig | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"MCP server '{server}' auth must be an object")
    if set(value) != {"type", "env"} and set(value) != {"type", "env", "binding"}:
        raise ValueError(f"MCP server '{server}' auth has unknown or missing fields")
    if value.get("type") != "bearer_env":
        raise ValueError("Only MCP bearer_env authentication is supported")
    env = _required_text(value.get("env"), f"MCP server '{server}' auth.env")
    binding = (
        _required_text(value.get("binding"), "auth.binding")
        if value.get("binding") is not None
        else f"mcp.{server}"
    )
    return MCPAuthConfig(type="bearer_env", env=env, binding=binding)


def _tool_policies(
    value: object,
    *,
    server: str,
) -> dict[str, Mapping[str, object]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"MCP server '{server}' tool_policies must be an object")
    result: dict[str, Mapping[str, object]] = {}
    for name, policy in value.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(policy, Mapping):
            raise ValueError(f"MCP server '{server}' has an invalid tool policy")
        unknown = sorted(set(policy) - _TOOL_POLICY_KEYS)
        if unknown:
            raise ValueError(
                f"MCP tool policy '{name}' has unknown fields: {', '.join(unknown)}"
            )
        result[name.strip()] = dict(policy)
    return result


def _string_set(value: object, field_name: str) -> set[str]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of strings")
    result = {item.strip() for item in value if isinstance(item, str) and item.strip()}
    if len(result) != len(value):
        raise ValueError(f"{field_name} must contain unique non-empty strings")
    return result


def _bounded_int(value: object, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("MCP numeric settings must be integers")
    if not minimum <= value <= maximum:
        raise ValueError(f"MCP numeric setting must be between {minimum} and {maximum}")
    return value


def _required_text(value: object, field_name: str) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


__all__ = [
    "MCPManager",
    "TransportFactory",
    "create_mcp_manager",
    "parse_mcp_server_configs",
]
