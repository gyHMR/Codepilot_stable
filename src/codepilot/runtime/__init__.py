"""
Codepilot runtime assembly layer.

- create_agent_session factory
- RuntimeService shared by user-facing interfaces
- workspace resource, prompt, and command assembly helpers
"""

from .command_registry import format_commands_for_help, list_runtime_commands
from .convert_to_llm import convert_to_llm
from .factory import create_agent_session
from .resources import WorkspaceModelConfig, WorkspaceResourceLoader, WorkspaceResources, WorkspaceSettings
from .service import CreateSessionRequest, RuntimeService, SessionHandle, UserInput
from .system_prompt import SystemPromptBuildOptions, build_default_system_prompt, build_system_prompt
from .types import CreateAgentSessionOptions


__all__ = [
    "CreateAgentSessionOptions",
    "convert_to_llm",
    "create_agent_session",
    "RuntimeService",
    "CreateSessionRequest",
    "SessionHandle",
    "UserInput",
    "WorkspaceResourceLoader",
    "WorkspaceModelConfig",
    "WorkspaceResources",
    "WorkspaceSettings",
    "build_default_system_prompt",
    "build_system_prompt",
    "SystemPromptBuildOptions",
    "format_commands_for_help",
    "list_runtime_commands",
]
