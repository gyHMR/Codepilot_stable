from __future__ import annotations

"""Adapt configured MCP tools into canonical Codepilot registrations."""

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from codepilot.tools import (
    ArtifactContent,
    ArtifactRef,
    ConcurrencyPolicy,
    ImageContent,
    JsonObjectCodec,
    OutputLimits,
    OutputTrustPolicy,
    TextContent,
    TimeoutPolicy,
    ToolAccessRequest,
    ToolAccessResolution,
    ToolEffect,
    ToolHandlerError,
    ToolPolicy,
    ToolRegistration,
    ToolResource,
    ToolSpec,
    UnverifiedJsonCodec,
)


class MCPClient(Protocol):
    async def call_tool(self, server: str, tool: str, arguments: dict[str, Any]) -> Any:
        ...


@dataclass(frozen=True)
class MCPToolConfig:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None
    server: str
    tool: str
    configured_name: str
    read_only: bool
    risk_level: str
    requires_approval: bool
    network_access: bool
    credential_required: bool
    credential_binding: str | None
    output_trust: str
    timeout_ms: int
    max_parallel: int
    max_output_bytes: int
    max_image_bytes: int


def parse_mcp_tool_configs(raw_servers: list[dict[str, Any]] | None) -> list[MCPToolConfig]:
    if not raw_servers:
        return []
    configs: list[MCPToolConfig] = []
    for raw_server in raw_servers:
        if not isinstance(raw_server, dict):
            continue
        server = _required_name(raw_server.get("name"))
        tools = raw_server.get("tools")
        if server is None or not isinstance(tools, list):
            continue
        timeout_ms = _bounded_int(raw_server.get("timeout_ms"), default=30_000, minimum=100, maximum=120_000)
        max_parallel = _bounded_int(raw_server.get("max_parallel"), default=4, minimum=1, maximum=16)
        max_output_bytes = _bounded_int(
            raw_server.get("max_output_bytes"),
            default=1_000_000,
            minimum=1_024,
            maximum=8_000_000,
        )
        max_image_bytes = _bounded_int(
            raw_server.get("max_image_bytes"),
            default=4_000_000,
            minimum=1_024,
            maximum=16_000_000,
        )
        credential_binding = _optional_text(raw_server.get("credential_binding"))
        allow_tools = _string_set(raw_server.get("allow_tools"))
        for raw_tool in tools:
            if not isinstance(raw_tool, dict):
                continue
            if "parameters" in raw_tool:
                raise ValueError(
                    "MCP tool field 'parameters' is no longer supported; use 'inputSchema'"
                )
            if "tool" not in raw_tool and "name" in raw_tool:
                raise ValueError(
                    "MCP tool field 'name' cannot replace 'tool'; declare the remote tool explicitly"
                )
            tool = _required_name(raw_tool.get("tool"))
            if tool is None or (allow_tools and tool not in allow_tools):
                continue
            configured_name = _optional_text(raw_tool.get("name")) or tool
            read_only = raw_tool.get("read_only") is True
            risk_level = _risk_level(
                raw_tool.get("risk_level"),
                default="low" if read_only else "medium",
            )
            requires_approval = _bool(
                raw_tool.get("requires_approval"),
                default=not read_only,
            )
            input_schema = _object_schema(raw_tool.get("inputSchema"))
            output_schema = raw_tool.get("outputSchema")
            if not isinstance(output_schema, dict):
                output_schema = None
            description = _optional_text(raw_tool.get("description")) or (
                f"Call MCP tool {server}.{tool} and return its normalized result."
            )
            configs.append(
                MCPToolConfig(
                    name=_mcp_name(server, tool),
                    description=description,
                    input_schema=input_schema,
                    output_schema=output_schema,
                    server=server,
                    tool=tool,
                    configured_name=configured_name,
                    read_only=read_only,
                    risk_level=risk_level,
                    requires_approval=requires_approval,
                    network_access=_bool(raw_tool.get("network_access"), default=True),
                    credential_required=_bool(
                        raw_tool.get("credential_required"),
                        default=False,
                    ),
                    credential_binding=credential_binding,
                    output_trust=_output_trust(raw_tool.get("output_trust")),
                    timeout_ms=timeout_ms,
                    max_parallel=max_parallel,
                    max_output_bytes=max_output_bytes,
                    max_image_bytes=max_image_bytes,
                )
            )
    return configs


def create_mcp_registrations(
    configs: list[MCPToolConfig],
    *,
    client: MCPClient | None,
) -> list[ToolRegistration]:
    semaphores: dict[str, asyncio.Semaphore] = {}
    limits: dict[str, int] = {}
    registrations: list[ToolRegistration] = []
    for config in configs:
        previous = limits.setdefault(config.server, config.max_parallel)
        if previous != config.max_parallel:
            raise ValueError(f"MCP server {config.server} has inconsistent max_parallel values")
        semaphore = semaphores.setdefault(config.server, asyncio.Semaphore(config.max_parallel))
        registrations.append(_registration(config, client=client, semaphore=semaphore))
    return registrations


def _registration(
    config: MCPToolConfig,
    *,
    client: MCPClient | None,
    semaphore: asyncio.Semaphore,
) -> ToolRegistration:
    input_codec = JsonObjectCodec(config.input_schema)
    if config.output_schema is None:
        output_codec = UnverifiedJsonCodec(max_bytes=config.max_output_bytes)
        spec_output_schema = None
    else:
        wrapper_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"result": config.output_schema},
            "required": ["result"],
            "additionalProperties": False,
        }
        output_codec = JsonObjectCodec(wrapper_schema)
        spec_output_schema = wrapper_schema

    effects = {"external_state_read" if config.read_only else "external_state_write"}
    if config.network_access:
        effects.add("network_access")
    if config.credential_required:
        effects.add("credential_access")
    declared_effects = frozenset(effects)
    resource = ToolResource(
        f"mcp://{config.server}/{config.tool}",
        metadata={
            "server": config.server,
            "tool": config.tool,
            "configured_name": config.configured_name,
            **(
                {"credential_binding": config.credential_binding}
                if config.credential_binding is not None
                else {}
            ),
        },
    )

    class Resolver:
        def resolve(self, input, request):
            _ = request
            return ToolAccessResolution(
                input=input,
                access=ToolAccessRequest(
                    actions=("mcp.call", f"mcp.{config.server}.{config.tool}"),
                    resources=(resource,),
                    effects=declared_effects,
                    risk=config.risk_level,
                    reason=f"Call MCP server {config.server} tool {config.tool}",
                    safe_preview={
                        "server": config.server,
                        "tool": config.tool,
                        "read_only": config.read_only,
                        "credential_binding": config.credential_binding or "",
                    },
                ),
            )

    async def handler(input, context):
        if client is None:
            raise ToolHandlerError("mcp.unavailable", "MCP client is unavailable")
        context.cancellation.raise_if_cancelled()
        await context.progress.report(
            "mcp_call_started",
            data={"server": config.server, "tool": config.tool},
        )
        try:
            async with semaphore:
                raw_result = await client.call_tool(config.server, config.tool, dict(input))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ToolHandlerError(
                "mcp.call_failed",
                f"MCP call failed for {config.server}.{config.tool}",
                details={"error_type": type(exc).__name__},
            ) from exc
        context.cancellation.raise_if_cancelled()
        for kind in sorted(declared_effects):
            context.effects.report(
                ToolEffect(
                    kind=kind,
                    resource=resource,
                    operation=f"call {config.server}.{config.tool}",
                    status="completed",
                    certainty="observed",
                )
            )
        await context.progress.report(
            "mcp_call_completed",
            data={"server": config.server, "tool": config.tool},
        )
        normalized = _json_safe_result(raw_result)
        encoded_size = len(
            json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        if encoded_size > config.max_output_bytes:
            raise ToolHandlerError(
                "mcp.output_too_large",
                f"MCP result exceeds the configured {config.max_output_bytes} byte limit",
            )
        return {"result": normalized}

    return ToolRegistration(
        version="1.0.0",
        implementation_version="1",
        spec=ToolSpec(
            config.name,
            config.description,
            config.input_schema,
            spec_output_schema,
        ),
        category="external",
        source="mcp",
        owner=f"mcp:{config.server}",
        policy=ToolPolicy(
            allowed_modes=(
                frozenset({"plan", "execute"})
                if config.read_only
                else frozenset({"execute"})
            ),
            declared_effects=declared_effects,
            required_permissions=frozenset(
                {
                    "mcp.call",
                    *(
                        {f"credential:{config.credential_binding}"}
                        if config.credential_required and config.credential_binding
                        else set()
                    ),
                }
            ),
            base_risk=config.risk_level,
            approval="always" if config.requires_approval else "never",
            timeout=TimeoutPolicy(config.timeout_ms, config.timeout_ms),
            concurrency=ConcurrencyPolicy(mode="parallel"),
            output_limits=OutputLimits(
                max_data_bytes=config.max_output_bytes,
                max_content_bytes=min(config.max_output_bytes, 256_000),
                max_artifact_bytes=config.max_output_bytes,
            ),
            output_trust=OutputTrustPolicy(
                default_content_trust=config.output_trust,
                allow_structurally_validated=config.output_schema is None,
            ),
        ),
        input_codec=input_codec,
        output_codec=output_codec,
        handler=handler,
        renderer=_MCPRenderer(config),
        access_resolver=Resolver(),
    )


class _MCPRenderer:
    def __init__(self, config: MCPToolConfig) -> None:
        self.config = config

    def render(self, data):
        result = data.get("result")
        blocks = self._blocks(result)
        return tuple(blocks or [TextContent(text="MCP tool returned no supported content.")])

    def _blocks(self, result: object) -> list[object]:
        if isinstance(result, str):
            return [TextContent(text=result)]
        if isinstance(result, dict):
            omission = result.get("omission")
            if isinstance(omission, dict):
                return [TextContent(text=_omission_text(omission))]
            content = result.get("content")
            if isinstance(content, (list, tuple)):
                blocks = [self._block(item) for item in content[:16]]
                if len(content) > 16:
                    blocks.append(TextContent(text="MCP content omitted: artifact count exceeds 16."))
                return blocks
        return [TextContent(text=json.dumps(result, ensure_ascii=False, sort_keys=True))]

    def _block(self, item: object):
        if not isinstance(item, dict):
            return TextContent(text="MCP content omitted: unsupported block shape.")
        kind = _optional_text(item.get("type")) or "unknown"
        if kind == "text" and isinstance(item.get("text"), str):
            return TextContent(text=item["text"])
        if kind == "image":
            data = item.get("data")
            mime_type = _optional_text(item.get("mime_type") or item.get("mimeType"))
            if isinstance(data, str) and mime_type and _estimated_base64_bytes(data) <= self.config.max_image_bytes:
                return ImageContent(data=data, mime_type=mime_type, name=_optional_text(item.get("name")))
            return TextContent(text="MCP image omitted: missing type/data or image exceeds the server limit.")
        if kind in {"artifact", "resource"}:
            artifact_id = _optional_text(
                item.get("artifact_id") or item.get("id") or item.get("uri")
            )
            if artifact_id and _safe_artifact_id(artifact_id):
                size_bytes = item.get("size_bytes")
                if not isinstance(size_bytes, int) or isinstance(size_bytes, bool):
                    size_bytes = None
                if size_bytes is None or size_bytes <= self.config.max_output_bytes:
                    return ArtifactContent(
                        artifact=ArtifactRef(
                            artifact_id=artifact_id,
                            media_type=_optional_text(
                                item.get("mime_type") or item.get("media_type")
                            )
                            or "application/octet-stream",
                            name=_optional_text(item.get("name")),
                            size_bytes=size_bytes,
                        )
                    )
            return TextContent(text="MCP artifact omitted: invalid reference or artifact exceeds the server limit.")
        return TextContent(text=f"MCP content omitted: unsupported content type {kind}.")


def _json_safe_result(value: object) -> object:
    if isinstance(value, bytes):
        return {
            "omission": {
                "reason": "binary_content_not_supported",
                "size_bytes": len(value),
            }
        }
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError):
        return {
            "omission": {
                "reason": "unsupported_result_type",
                "type": type(value).__name__,
            }
        }


def _mcp_name(server: str, tool: str) -> str:
    return f"mcp__{_slug(server)}__{_slug(tool)}"


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    return text or "tool"


def _object_schema(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {"type": "object", "properties": {}, "additionalProperties": True}
    schema = dict(value)
    schema.setdefault("$schema", "https://json-schema.org/draft/2020-12/schema")
    schema.setdefault("type", "object")
    return schema


def _required_name(value: object) -> str | None:
    return _optional_text(value)


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _string_set(value: object) -> set[str]:
    if not isinstance(value, (list, tuple)):
        return set()
    return {text for item in value if (text := _optional_text(item)) is not None}


def _bool(value: object, *, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _risk_level(value: object, *, default: str) -> str:
    return str(value) if value in {"low", "medium", "high", "critical"} else default


def _output_trust(value: object) -> str:
    return str(value) if value in {"trusted", "untrusted"} else "untrusted"


def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return max(minimum, min(maximum, value))


def _estimated_base64_bytes(value: str) -> int:
    return max(0, len(value.rstrip("=")) * 3 // 4)


def _safe_artifact_id(value: str) -> bool:
    lowered = value.lower()
    if lowered.startswith("file:") or value.startswith(("/", "\\")):
        return False
    return not re.match(r"^[A-Za-z]:[\\/]", value)


def _omission_text(value: dict[str, object]) -> str:
    reason = _optional_text(value.get("reason")) or "unsupported_content"
    return f"MCP content omitted: {reason}."


__all__ = [
    "MCPClient",
    "MCPToolConfig",
    "create_mcp_registrations",
    "parse_mcp_tool_configs",
]
