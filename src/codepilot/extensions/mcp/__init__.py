# 新手导读：包门面文件：集中导出本层最常用的类型和入口，降低学习时的导入成本。
# 关注点：extensions 层负责把 Python 扩展、skill 和 MCP 外部能力加载成项目内统一能力。

"""MCP（Model Context Protocol）canonical tool adapter."""

from .adapter import MCPClient, create_mcp_registrations
from .client import MCPManager, create_mcp_manager, parse_mcp_server_configs
from .transport import (
    MCPAuthConfig,
    MCPRemoteTool,
    MCPServerConfig,
    MCPTransport,
    MCPTransportError,
    StreamableHttpTransport,
)

__all__ = [
    "MCPClient",
    "MCPAuthConfig",
    "MCPManager",
    "MCPRemoteTool",
    "MCPServerConfig",
    "MCPTransport",
    "MCPTransportError",
    "StreamableHttpTransport",
    "create_mcp_manager",
    "create_mcp_registrations",
    "parse_mcp_server_configs",
]
