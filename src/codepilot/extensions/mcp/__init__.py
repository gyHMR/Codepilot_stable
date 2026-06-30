"""MCP（Model Context Protocol）桥接模块：将 MCP 服务器的工具代理为 AgentTool。"""

from .bridge import (
    MCPClient,
    MCPToolConfig,
    MCPToolPolicy,
    create_mcp_proxy_tools,
    parse_mcp_tool_configs,
)

__all__ = [
    "MCPClient",
    "MCPToolConfig",
    "MCPToolPolicy",
    "parse_mcp_tool_configs",
    "create_mcp_proxy_tools",
]
