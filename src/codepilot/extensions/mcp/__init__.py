# 新手导读：包门面文件：集中导出本层最常用的类型和入口，降低学习时的导入成本。
# 关注点：extensions 层负责把 Python 扩展、skill 和 MCP 外部能力加载成项目内统一能力。

"""MCP（Model Context Protocol）canonical tool adapter."""

from .adapter import (
    MCPClient,
    MCPToolConfig,
    create_mcp_registrations,
    parse_mcp_tool_configs,
)

__all__ = [
    "MCPClient",
    "MCPToolConfig",
    "parse_mcp_tool_configs",
    "create_mcp_registrations",
]
