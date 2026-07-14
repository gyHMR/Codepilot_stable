"""把 MCP 远程工具适配为统一 ToolRegistration 与执行结果。"""

from __future__ import annotations

"""Adapt configured MCP tools into canonical Codepilot registrations."""

import asyncio
import json
import re
from typing import Any, Mapping, Protocol

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

from .transport import MCPRemoteTool, MCPServerConfig, MCPTransportError


class MCPClient(Protocol):
    """MCP 远程工具列表与调用能力的最小客户端协议。"""
    async def call_tool(self, server: str, tool: str, arguments: dict[str, Any]) -> Any:
        ...


def create_mcp_registrations(
    config: MCPServerConfig,
    tools: tuple[MCPRemoteTool, ...],
    *,
    client: MCPClient | None,
) -> list[ToolRegistration]:
    return [_registration(config, tool, client=client) for tool in tools]


def _registration(
    config: MCPServerConfig,
    remote: MCPRemoteTool,
    *,
    client: MCPClient | None,
) -> ToolRegistration:
    policy_values = _resolved_policy(config, remote)
    input_schema = _object_schema(remote.input_schema)
    input_codec = JsonObjectCodec(input_schema)
    if remote.output_schema is None:
        output_codec = UnverifiedJsonCodec(max_bytes=config.max_output_bytes)
        spec_output_schema = None
    else:
        wrapper_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "result": {},
                "structured_output": dict(remote.output_schema),
            },
            "required": ["result", "structured_output"],
            "additionalProperties": False,
        }
        output_codec = JsonObjectCodec(wrapper_schema)
        spec_output_schema = wrapper_schema

    effects = {
        "external_state_read" if policy_values["read_only"] else "external_state_write",
        "network_access",
    }
    if config.auth is not None:
        effects.add("credential_access")
    declared_effects = frozenset(effects)
    resource = ToolResource(
        f"mcp://{config.name}/{remote.name}",
        metadata={
            "server": config.name,
            "tool": remote.name,
            **({"credential_binding": config.auth.binding} if config.auth else {}),
        },
    )

    class Resolver:
        def resolve(self, input, request):
            _ = request
            return ToolAccessResolution(
                input=input,
                access=ToolAccessRequest(
                    actions=("mcp.call", f"mcp.{config.name}.{remote.name}"),
                    resources=(resource,),
                    effects=declared_effects,
                    risk=policy_values["risk_level"],
                    reason=f"Call MCP server {config.name} tool {remote.name}",
                    safe_preview={
                        "server": config.name,
                        "tool": remote.name,
                        "read_only": policy_values["read_only"],
                        "credential_binding": config.auth.binding if config.auth else "",
                    },
                ),
            )

    async def handler(input, context):
        if client is None:
            raise ToolHandlerError("mcp.unavailable", "MCP client is unavailable")
        context.cancellation.raise_if_cancelled()
        await context.progress.report(
            "mcp_call_started",
            data={"server": config.name, "tool": remote.name},
        )
        try:
            raw_result = await client.call_tool(config.name, remote.name, dict(input))
        except asyncio.CancelledError:
            raise
        except MCPTransportError as exc:
            raise ToolHandlerError(
                exc.code,
                exc.message,
                retryable=exc.retryable,
            ) from exc
        except Exception as exc:
            raise ToolHandlerError(
                "mcp.call_failed",
                f"MCP call failed for {config.name}.{remote.name}",
                details={"error_type": type(exc).__name__},
            ) from exc
        context.cancellation.raise_if_cancelled()
        for kind in sorted(declared_effects):
            context.effects.report(
                ToolEffect(
                    kind=kind,
                    resource=resource,
                    operation=f"call {config.name}.{remote.name}",
                    status="completed",
                    certainty="observed" if kind == "network_access" else "reported",
                )
            )
        await context.progress.report(
            "mcp_call_completed",
            data={"server": config.name, "tool": remote.name},
        )
        normalized = _json_safe_result(raw_result)
        if remote.output_schema is None:
            return {"result": normalized}
        structured = (
            raw_result.get("structuredContent")
            if isinstance(raw_result, Mapping)
            else None
        )
        return {"result": normalized, "structured_output": structured}

    return ToolRegistration(
        version="1.0.0",
        implementation_version="1",
        spec=ToolSpec(
            _mcp_name(config.name, remote.name),
            remote.description,
            input_schema,
            spec_output_schema,
        ),
        category="external",
        source="mcp",
        owner=f"mcp:{config.name}",
        policy=ToolPolicy(
            allowed_modes=(
                frozenset({"plan", "execute"})
                if policy_values["read_only"]
                else frozenset({"execute"})
            ),
            declared_effects=declared_effects,
            required_permissions=frozenset(
                {
                    "mcp.call",
                    *(
                        {f"credential:{config.auth.binding}"}
                        if config.auth is not None
                        else set()
                    ),
                }
            ),
            base_risk=policy_values["risk_level"],
            approval="always" if policy_values["requires_approval"] else "never",
            timeout=TimeoutPolicy(config.timeout_ms, config.timeout_ms),
            concurrency=ConcurrencyPolicy(
                mode="parallel",
                group=f"mcp:{config.name}",
                max_parallel=config.max_parallel,
            ),
            output_limits=OutputLimits(
                max_data_bytes=config.max_output_bytes,
                max_content_bytes=min(config.max_output_bytes, 256_000),
                max_artifact_bytes=config.max_output_bytes,
            ),
            output_trust=OutputTrustPolicy(
                default_content_trust=policy_values["output_trust"],
                allow_structurally_validated=remote.output_schema is None,
            ),
        ),
        input_codec=input_codec,
        output_codec=output_codec,
        handler=handler,
        renderer=_MCPRenderer(config),
        access_resolver=Resolver(),
    )


class _MCPRenderer:
    def __init__(self, config: MCPServerConfig) -> None:
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
    if not isinstance(value, Mapping):
        raise ValueError("MCP input schema must be an object")
    schema = dict(value)
    schema.setdefault("$schema", "https://json-schema.org/draft/2020-12/schema")
    if schema.get("type") != "object":
        raise ValueError("MCP input schema must declare type=object")
    return schema


def _resolved_policy(
    config: MCPServerConfig,
    remote: MCPRemoteTool,
) -> dict[str, Any]:
    override = config.tool_policies.get(remote.name, {})
    read_only = override.get("read_only")
    if read_only is None:
        read_only = remote.annotations.get("readOnlyHint") is True
    elif not isinstance(read_only, bool):
        raise ValueError(f"MCP tool '{remote.name}' read_only must be bool")
    risk = override.get("risk_level")
    if risk is None:
        risk = "high" if remote.annotations.get("destructiveHint") is True else (
            "low" if read_only else "medium"
        )
    if risk not in {"low", "medium", "high", "critical"}:
        raise ValueError(f"MCP tool '{remote.name}' has invalid risk_level")
    approval = override.get("requires_approval")
    if approval is None:
        approval = not read_only
    elif not isinstance(approval, bool):
        raise ValueError(f"MCP tool '{remote.name}' requires_approval must be bool")
    trust = override.get("output_trust", "untrusted")
    if trust not in {"trusted", "untrusted"}:
        raise ValueError(f"MCP tool '{remote.name}' has invalid output_trust")
    return {
        "read_only": read_only,
        "risk_level": risk,
        "requires_approval": approval,
        "output_trust": trust,
    }


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


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
    "create_mcp_registrations",
]
