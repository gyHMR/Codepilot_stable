"""
Codepilot 运行时组装层（Runtime Assembly Layer）。

本模块是 runtime 子包的统一入口，负责将配置、模型、提示词、工具、
会话、钩子和命令组装成可用的 Agent 会话。

核心导出：
- create_agent_session: Agent 会话工厂函数
- RuntimeService: 面向用户接口（CLI/Web/IM）的共享服务层
- 工作区资源和命令组装辅助函数
"""

from .command_registry import format_commands_for_help, list_runtime_commands
from .assembly import assemble_runtime, create_agent_session, explain_runtime_config
from .prompt import build_default_system_prompt
from .resources import WorkspaceModelConfig, WorkspaceResourceLoader, WorkspaceResources, WorkspaceSettings
from .service import RuntimeService
from .types import CreateAgentSessionOptions, SessionHandle, UserInput


__all__ = [
    # ── 会话创建 ──
    "CreateAgentSessionOptions",
    "assemble_runtime",
    "create_agent_session",
    "explain_runtime_config",
    # ── 运行时服务 ──
    "RuntimeService",
    "SessionHandle",
    "UserInput",
    # ── 工作区资源 ──
    "WorkspaceResourceLoader",
    "WorkspaceModelConfig",
    "WorkspaceResources",
    "WorkspaceSettings",
    # ── 提示词构建 ──
    "build_default_system_prompt",
    # ── 命令注册 ──
    "format_commands_for_help",
    "list_runtime_commands",
]
