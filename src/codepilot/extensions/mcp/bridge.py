from __future__ import annotations

"""MCP 桥接：解析 MCP 服务器配置并创建代理工具。"""

from dataclasses import dataclass
from typing import Any, Protocol

from codepilot.protocols import TextContent, ToolMetadata, ToolRiskLevel
from codepilot.tools import AgentTool, AgentToolResult


class MCPClient(Protocol):
    """MCP 客户端协议：由外部 MCP 适配器实现。"""
    async def call_tool(self, server: str, tool: str, arguments: dict[str, Any]) -> Any:
        """调用 MCP 服务器工具并返回结果对象。"""


@dataclass(frozen=True)
class MCPToolPolicy:
    """MCP 工具策略：记录风险、审批、资源范围和输出可信度。"""

    risk_level: ToolRiskLevel = "medium"
    requires_approval: bool = True
    resource_scope: tuple[str, ...] = ("mcp",)
    network_access: bool = True
    credential_required: bool = False
    output_trust: str = "untrusted"

    @classmethod
    def from_raw(cls, raw: dict[str, Any], *, server_name: str) -> "MCPToolPolicy":
        return cls(
            risk_level=_risk_level(raw.get("risk_level")),
            requires_approval=_bool_config(raw.get("requires_approval"), default=True),
            resource_scope=_resource_scope(raw.get("resource_scope"), server_name=server_name),
            network_access=_bool_config(raw.get("network_access"), default=True),
            credential_required=_bool_config(raw.get("credential_required"), default=False),
            output_trust=_output_trust(raw.get("output_trust")),
        )

    @classmethod
    def from_config(cls, cfg: "MCPToolConfig") -> "MCPToolPolicy":
        return cls(
            risk_level=cfg.risk_level,
            requires_approval=cfg.requires_approval,
            resource_scope=cfg.resource_scope,
            network_access=cfg.network_access,
            credential_required=cfg.credential_required,
            output_trust=cfg.output_trust,
        )

    def to_metadata(self, *, name: str, server: str, tool: str) -> ToolMetadata:
        return ToolMetadata(
            name=name,
            category="mcp",
            read_only=False,
            concurrency_safe=False,
            exclusive=True,
            requires_approval=self.requires_approval,
            risk_level=self.risk_level,
            resource_scope=self.resource_scope,
            network_access=self.network_access,
            credential_required=self.credential_required,
            extra={
                "server": server,
                "tool": tool,
                "output_trust": self.output_trust,
                "capabilities": ["mcp.call"],
            },
        )


@dataclass
class MCPToolConfig:
    """MCP 工具配置：描述一个 MCP 服务器工具的映射关系。"""
    name: str                   # 代理工具名称
    description: str            # 工具描述
    parameters: dict[str, Any]  # JSON Schema 参数定义
    server: str                 # MCP 服务器名称
    tool: str                   # MCP 服务器上的工具名
    risk_level: ToolRiskLevel = "medium"
    requires_approval: bool = True
    resource_scope: tuple[str, ...] = ("mcp",)
    network_access: bool = True
    credential_required: bool = False
    output_trust: str = "untrusted"


def parse_mcp_tool_configs(raw_servers: list[dict[str, Any]] | None) -> list[MCPToolConfig]:
    """解析 MCP 服务器配置列表为 MCPToolConfig 列表。"""
    if not raw_servers:
        return []
    result: list[MCPToolConfig] = []
    for server in raw_servers:
        if not isinstance(server, dict):
            continue
        server_name = server.get("name")
        tools = server.get("tools")
        if not isinstance(server_name, str) or not isinstance(tools, list):
            continue
        for item in tools:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            tool = item.get("tool") or name
            description = item.get("description") or f"MCP tool proxy: {server_name}.{tool}"
            params = item.get("parameters")
            if not isinstance(name, str) or not isinstance(tool, str):
                continue
            if not isinstance(description, str):
                description = str(description)
            if not isinstance(params, dict):
                params = {"type": "object", "properties": {}, "required": [], "additionalProperties": True}
            policy = MCPToolPolicy.from_raw(item, server_name=server_name)
            result.append(
                MCPToolConfig(
                    name=name,
                    description=description,
                    parameters=params,
                    server=server_name,
                    tool=tool,
                    risk_level=policy.risk_level,
                    requires_approval=policy.requires_approval,
                    resource_scope=policy.resource_scope,
                    network_access=policy.network_access,
                    credential_required=policy.credential_required,
                    output_trust=policy.output_trust,
                )
            )
    return result


def create_mcp_proxy_tools(configs: list[MCPToolConfig], client: MCPClient | None) -> list[AgentTool]:
    """从 MCP 工具配置创建代理 AgentTool 列表（通过 MCPClient 转发调用）。"""
    tools: list[AgentTool] = []
    for cfg in configs:
        async def _execute(tool_call_id, params, signal=None, on_update=None, *, _cfg=cfg):  # type: ignore[no-untyped-def]
            _ = tool_call_id, signal, on_update
            args = params if isinstance(params, dict) else {}
            if client is None:
                return AgentToolResult(
                    content=[TextContent(text=f"MCP bridge unavailable for `{_cfg.name}`")],
                    is_error=True,
                    metadata={"output_trust": _cfg.output_trust},
                )
            try:
                result = await client.call_tool(_cfg.server, _cfg.tool, args)
            except Exception as exc:  # pragma: no cover - adapter-specific
                return AgentToolResult(
                    content=[TextContent(text=f"MCP call failed `{_cfg.server}.{_cfg.tool}`: {exc}")],
                    is_error=True,
                    metadata={"output_trust": _cfg.output_trust},
                )
            text, metadata = _normalize_mcp_result(result)
            metadata.setdefault("output_trust", _cfg.output_trust)
            return AgentToolResult(
                content=[TextContent(text=text)],
                details={"server": _cfg.server, "tool": _cfg.tool},
                metadata=metadata,
            )

        metadata = _mcp_tool_metadata(cfg)
        tools.append(
            AgentTool(
                name=cfg.name,
                label=f"MCP/{cfg.server}",
                description=cfg.description,
                parameters=cfg.parameters,
                execute=_execute,
                metadata=metadata,
            )
        )
    return tools


def _mcp_tool_metadata(cfg: MCPToolConfig) -> ToolMetadata:
    return MCPToolPolicy.from_config(cfg).to_metadata(
        name=cfg.name,
        server=cfg.server,
        tool=cfg.tool,
    )


def _risk_level(value: object) -> ToolRiskLevel:
    return value if value in {"low", "medium", "high"} else "medium"  # type: ignore[return-value]


def _resource_scope(value: object, *, server_name: str) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        cleaned = tuple(str(item).strip() for item in value if str(item).strip())
        if cleaned:
            return cleaned
    return ("mcp", server_name)


def _bool_config(value: object, *, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _output_trust(value: object) -> str:
    return value if value in {"trusted", "untrusted"} else "untrusted"  # type: ignore[return-value]


def _output_quality(
    *,
    decode_status: str,
    original_chars: int,
    returned_chars: int,
    may_be_binary: bool = False,
) -> dict[str, Any]:
    return {
        "output_quality": {
            "encoding": "utf-8",
            "decode_status": decode_status,
            "truncated": False,
            "original_chars": original_chars,
            "returned_chars": returned_chars,
            "may_be_binary": may_be_binary,
            "reliable_for_reasoning": decode_status == "ok" and not may_be_binary,
        }
    }


def _normalize_mcp_result(value: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(value, str):
        return value, _output_quality(
            decode_status="ok",
            original_chars=len(value),
            returned_chars=len(value),
        )
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
            status = "ok"
        except UnicodeDecodeError:
            text = value.decode("utf-8", errors="replace")
            status = "decoded_with_replacement"
        return text, _output_quality(
            decode_status=status,
            original_chars=len(value),
            returned_chars=len(text),
            may_be_binary=status != "ok",
        )
    if isinstance(value, dict):
        text = str(value)
        return text, _output_quality(
            decode_status="ok",
            original_chars=len(text),
            returned_chars=len(text),
        )
    if isinstance(value, list):
        text = "\n".join(str(x) for x in value)
        return text, _output_quality(
            decode_status="ok",
            original_chars=len(text),
            returned_chars=len(text),
        )
    text = str(value)
    return text, _output_quality(
        decode_status="ok",
        original_chars=len(text),
        returned_chars=len(text),
    )
