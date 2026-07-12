from __future__ import annotations

"""Minimal MCP Streamable HTTP transport and remote definition models."""

import json
from dataclasses import dataclass, field
from typing import Mapping, Protocol
from urllib.parse import urlparse

import httpx


MCP_PROTOCOL_VERSION = "2025-06-18"


class MCPTransportError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        self.code = code
        self.message = message
        self.retryable = retryable
        super().__init__(message)


@dataclass(frozen=True)
class MCPAuthConfig:
    type: str
    env: str
    binding: str


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    url: str
    auth: MCPAuthConfig | None
    allow_tools: frozenset[str]
    timeout_ms: int = 30_000
    max_parallel: int = 4
    max_output_bytes: int = 1_000_000
    max_image_bytes: int = 4_000_000
    tool_policies: Mapping[str, Mapping[str, object]] = field(default_factory=dict)


@dataclass(frozen=True)
class MCPRemoteTool:
    name: str
    description: str
    input_schema: Mapping[str, object]
    output_schema: Mapping[str, object] | None = None
    annotations: Mapping[str, object] = field(default_factory=dict)


class MCPTransport(Protocol):
    async def initialize(self) -> None: ...

    async def list_tools(self) -> tuple[MCPRemoteTool, ...]: ...

    async def call_tool(self, tool: str, arguments: Mapping[str, object]) -> object: ...

    async def aclose(self) -> None: ...


class StreamableHttpTransport:
    """One MCP Streamable HTTP session backed by httpx."""

    def __init__(
        self,
        config: MCPServerConfig,
        *,
        token: str | None = None,
        http_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self._token = token
        self._session_id: str | None = None
        self._next_id = 0
        self._initialized = False
        self.diagnostics: list[str] = []
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout_ms / 1_000),
            follow_redirects=False,
            transport=http_transport,
        )

    async def initialize(self) -> None:
        if self._initialized:
            return
        result = await self._rpc(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "codepilot", "version": "0.3.0"},
            },
        )
        if not isinstance(result, Mapping):
            raise MCPTransportError(
                "mcp.protocol.invalid_initialize",
                f"MCP server '{self.config.name}' returned an invalid initialize result",
            )
        if result.get("protocolVersion") != MCP_PROTOCOL_VERSION:
            raise MCPTransportError(
                "mcp.protocol.version_mismatch",
                f"MCP server '{self.config.name}' did not negotiate {MCP_PROTOCOL_VERSION}",
            )
        await self._notification("notifications/initialized", {})
        self._initialized = True

    async def list_tools(self) -> tuple[MCPRemoteTool, ...]:
        self._require_initialized()
        cursor: str | None = None
        tools: list[MCPRemoteTool] = []
        while True:
            params = {"cursor": cursor} if cursor else {}
            result = await self._rpc("tools/list", params)
            if not isinstance(result, Mapping) or not isinstance(result.get("tools"), list):
                raise MCPTransportError(
                    "mcp.protocol.invalid_tool_list",
                    f"MCP server '{self.config.name}' returned an invalid tools/list result",
                )
            for item in result["tools"]:
                try:
                    tools.append(_remote_tool(item))
                except MCPTransportError as exc:
                    self.diagnostics.append(f"{exc.code}: {exc.message}")
            next_cursor = result.get("nextCursor")
            cursor = next_cursor.strip() if isinstance(next_cursor, str) else None
            if not cursor:
                return tuple(tools)

    async def call_tool(self, tool: str, arguments: Mapping[str, object]) -> object:
        self._require_initialized()
        result = await self._rpc(
            "tools/call",
            {"name": tool, "arguments": dict(arguments)},
        )
        if isinstance(result, Mapping) and result.get("isError") is True:
            raise MCPTransportError(
                "mcp.tool.remote_error",
                f"MCP tool '{self.config.name}.{tool}' returned an error",
            )
        return result

    async def aclose(self) -> None:
        try:
            if self._session_id is not None:
                try:
                    await self._client.delete(
                        self.config.url,
                        headers=self._headers(),
                    )
                except httpx.HTTPError:
                    pass
        finally:
            await self._client.aclose()
            self._session_id = None
            self._initialized = False
            self._token = None

    async def _rpc(self, method: str, params: Mapping[str, object]) -> object:
        self._next_id += 1
        request_id = self._next_id
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": dict(params),
        }
        response = await self._post(payload)
        message = _response_message(response)
        if message.get("id") != request_id:
            raise MCPTransportError(
                "mcp.protocol.response_mismatch",
                f"MCP server '{self.config.name}' returned a mismatched response id",
            )
        error = message.get("error")
        if isinstance(error, Mapping):
            code = error.get("code")
            raise MCPTransportError(
                "mcp.protocol.remote_error",
                f"MCP server '{self.config.name}' returned JSON-RPC error {code}",
            )
        if "result" not in message:
            raise MCPTransportError(
                "mcp.protocol.result_missing",
                f"MCP server '{self.config.name}' returned no result",
            )
        return message["result"]

    async def _notification(self, method: str, params: Mapping[str, object]) -> None:
        await self._post(
            {"jsonrpc": "2.0", "method": method, "params": dict(params)},
            allow_empty=True,
        )

    async def _post(
        self,
        payload: Mapping[str, object],
        *,
        allow_empty: bool = False,
    ) -> httpx.Response:
        try:
            async with self._client.stream(
                "POST",
                self.config.url,
                headers=self._headers(),
                json=dict(payload),
            ) as streamed:
                streamed.raise_for_status()
                chunks: list[bytes] = []
                size = 0
                async for chunk in streamed.aiter_bytes():
                    size += len(chunk)
                    if size > self.config.max_output_bytes:
                        raise MCPTransportError(
                            "mcp.transport.response_too_large",
                            f"MCP server '{self.config.name}' response exceeded its byte limit",
                        )
                    chunks.append(chunk)
                response = httpx.Response(
                    streamed.status_code,
                    headers=streamed.headers,
                    content=b"".join(chunks),
                    request=streamed.request,
                )
        except httpx.TimeoutException as exc:
            raise MCPTransportError(
                "mcp.transport.timeout",
                f"MCP server '{self.config.name}' timed out",
                retryable=True,
            ) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise MCPTransportError(
                "mcp.transport.http_status",
                f"MCP server '{self.config.name}' returned HTTP {status}",
                retryable=status in {408, 429} or status >= 500,
            ) from exc
        except httpx.HTTPError as exc:
            raise MCPTransportError(
                "mcp.transport.http_error",
                f"MCP server '{self.config.name}' request failed",
                retryable=True,
            ) from exc
        session_id = response.headers.get("MCP-Session-Id")
        if session_id:
            self._session_id = session_id
        if not response.content and not allow_empty:
            raise MCPTransportError(
                "mcp.protocol.empty_response",
                f"MCP server '{self.config.name}' returned an empty response",
            )
        return response

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
        }
        if self._session_id is not None:
            headers["MCP-Session-Id"] = self._session_id
        if self._token is not None:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise MCPTransportError(
                "mcp.client.not_initialized",
                f"MCP server '{self.config.name}' is not initialized",
            )


def validate_server_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("MCP server URL cannot contain credentials or fragments")
    if parsed.scheme == "https" and parsed.netloc:
        return url
    if parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
        return url
    raise ValueError("MCP server URL must use HTTPS, except for localhost testing")


def _response_message(response: httpx.Response) -> Mapping[str, object]:
    content_type = response.headers.get("content-type", "").lower()
    if "text/event-stream" in content_type:
        values = [
            json.loads(line[5:].strip())
            for line in response.text.splitlines()
            if line.startswith("data:") and line[5:].strip()
        ]
        if not values:
            raise MCPTransportError("mcp.protocol.invalid_sse", "MCP SSE response contained no data")
        value = values[-1]
    else:
        try:
            value = response.json()
        except ValueError as exc:
            raise MCPTransportError(
                "mcp.protocol.invalid_json",
                "MCP response was not valid JSON",
            ) from exc
    if not isinstance(value, Mapping) or value.get("jsonrpc") != "2.0":
        raise MCPTransportError("mcp.protocol.invalid_response", "Invalid MCP JSON-RPC response")
    return value


def _remote_tool(value: object) -> MCPRemoteTool:
    if not isinstance(value, Mapping):
        raise MCPTransportError("mcp.protocol.invalid_tool", "MCP tool definition must be an object")
    name = value.get("name")
    schema = value.get("inputSchema")
    if not isinstance(name, str) or not name.strip():
        raise MCPTransportError("mcp.protocol.invalid_tool", "MCP tool name is required")
    if not isinstance(schema, Mapping) or schema.get("type") != "object":
        raise MCPTransportError(
            "mcp.protocol.invalid_tool_schema",
            f"MCP tool '{name}' must declare an object inputSchema",
        )
    output_schema = value.get("outputSchema")
    return MCPRemoteTool(
        name=name.strip(),
        description=(
            value["description"].strip()
            if isinstance(value.get("description"), str) and value["description"].strip()
            else f"Call MCP tool {name.strip()}."
        ),
        input_schema=dict(schema),
        output_schema=dict(output_schema) if isinstance(output_schema, Mapping) else None,
        annotations=(
            dict(value["annotations"])
            if isinstance(value.get("annotations"), Mapping)
            else {}
        ),
    )


__all__ = [
    "MCPAuthConfig",
    "MCP_PROTOCOL_VERSION",
    "MCPRemoteTool",
    "MCPServerConfig",
    "MCPTransport",
    "MCPTransportError",
    "StreamableHttpTransport",
    "validate_server_url",
]
