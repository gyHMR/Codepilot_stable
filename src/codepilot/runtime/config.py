from __future__ import annotations

"""
运行时配置解析模块。

负责将上层传入的 CreateAgentSessionOptions 与工作区资源、
会话恢复元数据、默认值进行多源合并，生成最终的 ResolvedRuntimeConfig。

配置优先级（从高到低）：
1) 调用方显式传入的 options
2) 恢复的会话元数据（restored_meta）
3) 工作区配置文件（.codepilot/settings.json）
4) 硬编码默认值（RuntimeDefaults）
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from codepilot.core import ToolExecutionMode
from codepilot.sessions.persistence.store import SessionStore

from .resources import WorkspaceResourceLoader, WorkspaceResources
from .types import CreateAgentSessionOptions

# 配置来源标识
ConfigSource = str
T = TypeVar("T")


@dataclass(frozen=True)
class RuntimeDefaults:
    """运行时硬编码默认值。

    当 options、会话元数据、工作区配置均未指定时使用这些默认值。

    Attributes:
        system_prompt: 默认系统提示词（空字符串，由 prompt 模块填充）。
        thinking_level: 默认推理级别（"off" 关闭推理）。
        tool_execution: 默认工具执行模式（"parallel" 并行）。
        max_context_messages: 消息数量上限（None 表示不限制）。
        retain_recent_messages: 压缩时保留的最近消息数（24 条）。
        max_context_tokens: token 数量上限（None 表示不限制）。
        retry_enabled: 是否启用重试（默认 True）。
        max_retries: 最大重试次数（默认 2 次）。
        retry_base_delay_ms: 重试基础延迟（默认 1200ms，即指数退避起始值）。
        read_only_mode: 是否为只读模式（默认 False）。
        block_dangerous_bash: 是否阻止危险 bash 命令（默认 True）。
        bash_allow_patterns: bash 命令白名单。
        bash_block_patterns: bash 命令黑名单。
        edit_require_unique_match: edit 是否要求唯一匹配（默认 True）。
        extension_paths: 扩展加载路径。
        skill_paths: 技能加载路径。
        mcp_servers: MCP 服务器配置。
        prompt_guidelines: 额外的提示词准则。
        append_system_prompt: 追加到系统提示词末尾的文本。
        prompt_debug_sources: 是否包含调试来源信息。
        tool_snippets: 工具说明片段。
        enabled_builtin_tools: 启用的内置工具列表。
    """

    system_prompt: str = ""
    thinking_level: str = "off"
    tool_execution: ToolExecutionMode = "parallel"
    max_tool_calls_per_turn: int = 8
    max_context_messages: int | None = None
    retain_recent_messages: int = 24
    max_context_tokens: int | None = None
    retry_enabled: bool = True
    max_retries: int = 2
    retry_base_delay_ms: int = 1200
    read_only_mode: bool = False
    tool_permission_mode: str = "workspace-write"
    block_dangerous_bash: bool = True
    bash_allow_patterns: list[str] | None = None
    bash_block_patterns: list[str] | None = None
    edit_require_unique_match: bool = True
    extension_paths: list[str] | None = None
    skill_paths: list[str] | None = None
    mcp_servers: list[dict[str, Any]] | None = None
    prompt_guidelines: list[str] | None = None
    append_system_prompt: str | None = None
    prompt_debug_sources: bool = False
    tool_snippets: dict[str, str] | None = None
    enabled_builtin_tools: list[str] | None = None
    shell_timeout_seconds: int = 30
    shell_max_timeout_seconds: int = 120
    shell_stdout_limit: int = 20_000
    shell_stderr_limit: int = 10_000
    shell_allowed_env: list[str] | None = None


@dataclass(frozen=True)
class RuntimeInputs:
    """运行时输入数据（从 options 和工作区加载的原始数据）。

    Attributes:
        workspace: 工作区目录路径。
        resources: 工作区资源（settings.json、prompt.md 等），未加载时为 None。
        restored_meta: 恢复的会话元数据（provider/model_id/system_prompt 等）。
    """

    workspace: Path
    resources: WorkspaceResources | None
    restored_meta: dict[str, Any] | None


@dataclass(frozen=True)
class ResolvedRuntimeConfig:
    """最终解析完成的运行时配置。

    所有字段都经过多源合并（options -> 会话元数据 -> 工作区 -> 默认值），
    每个字段都确定了最终值和来源。

    Attributes:
        sources: 每个配置项的来源标识（如 "options"、"workspace"、"default"）。
        其余字段含义见 RuntimeDefaults 和 CreateAgentSessionOptions。
    """

    system_prompt: str
    thinking_level: str
    tool_execution: ToolExecutionMode
    max_context_messages: int | None
    retain_recent_messages: int
    max_context_tokens: int | None
    retry_enabled: bool
    max_retries: int
    retry_base_delay_ms: int
    read_only_mode: bool
    block_dangerous_bash: bool
    bash_allow_patterns: list[str] | None
    bash_block_patterns: list[str] | None
    edit_require_unique_match: bool
    extension_paths: list[str] | None
    skill_paths: list[str] | None
    mcp_servers: list[dict[str, Any]] | None
    prompt_guidelines: list[str] | None
    append_system_prompt: str | None
    prompt_debug_sources: bool
    tool_snippets: dict[str, str] | None
    enabled_builtin_tools: list[str] | None
    max_tool_calls_per_turn: int = 8
    tool_permission_mode: str = "workspace-write"
    shell_timeout_seconds: int = 30
    shell_max_timeout_seconds: int = 120
    shell_stdout_limit: int = 20_000
    shell_stderr_limit: int = 10_000
    shell_allowed_env: list[str] | None = None
    sources: dict[str, ConfigSource] = field(default_factory=dict)


# RuntimeConfig 是 ResolvedRuntimeConfig 的别名
RuntimeConfig = ResolvedRuntimeConfig


def load_runtime_inputs(options: CreateAgentSessionOptions) -> RuntimeInputs:
    """从 options 加载运行时输入数据。

    包括工作区路径、工作区资源文件、恢复的会话元数据。

    Args:
        options: 创建会话的配置选项。

    Returns:
        RuntimeInputs 对象，包含 workspace、resources 和 restored_meta。
    """
    workspace, resources = load_workspace_resources(options)
    return RuntimeInputs(
        workspace=workspace,
        resources=resources,
        restored_meta=read_restored_session_meta(workspace, options.session_id),
    )


def load_workspace_resources(options: CreateAgentSessionOptions) -> tuple[Path, WorkspaceResources | None]:
    """加载工作区资源（settings.json、prompt.md 等）。

    Args:
        options: 创建会话的配置选项。

    Returns:
        (工作区路径, 工作区资源) 元组；load_workspace_resources=False 时资源为 None。
    """
    workspace = Path(options.workspace_dir)
    resources = WorkspaceResourceLoader(workspace).load() if options.load_workspace_resources else None
    return workspace, resources


def read_restored_session_meta(workspace: Path, session_id: str | None) -> dict[str, Any] | None:
    """读取已恢复会话的元数据（provider、model_id、system_prompt 等）。

    Args:
        workspace: 工作区路径。
        session_id: 会话 ID（None 时直接返回 None）。

    Returns:
        会话元数据字典；session_id 为 None 或元数据不存在时返回 None。
    """
    if not session_id:
        return None
    return SessionStore(workspace_dir=workspace, session_id=session_id).read_meta()


def resolve_runtime_config(
    options: CreateAgentSessionOptions,
    inputs: RuntimeInputs,
    defaults: RuntimeDefaults = RuntimeDefaults(),
) -> ResolvedRuntimeConfig:
    """将多源配置合并为最终的 ResolvedRuntimeConfig。

    对每个配置项，按优先级依次检查：
    1) options 中的显式值
    2) 恢复的会话元数据
    3) 工作区 settings.json
    4) 硬编码默认值

    Args:
        options: 创建会话的配置选项。
        inputs: 运行时输入数据（含工作区资源和会话元数据）。
        defaults: 硬编码默认值。

    Returns:
        解析完成的 ResolvedRuntimeConfig，每个字段都确定了最终值。
    """
    resources = inputs.resources
    settings = resources.settings if resources is not None else None
    restored = inputs.restored_meta or {}
    sources: dict[str, ConfigSource] = {}

    def choose(name: str, *candidates: tuple[ConfigSource, T | None], default: T) -> T:
        """按优先级选择第一个非 None 的候选值，并记录来源。"""
        for source, value in candidates:
            if value is not None:
                sources[name] = source
                return value
        sources[name] = "default"
        return default

    # 逐项解析配置，每个 choose 调用对应一个配置项的多源合并
    system_prompt = choose(
        "system_prompt",
        ("options", options.system_prompt),
        ("restored_session", restored.get("system_prompt") if isinstance(restored.get("system_prompt"), str) else None),
        ("workspace", resources.prompt if resources is not None else None),
        ("workspace", settings.system_prompt if settings is not None else None),
        default=defaults.system_prompt,
    )
    thinking_level = choose(
        "thinking_level",
        ("options", options.thinking_level),
        ("workspace", settings.thinking_level if settings is not None else None),
        default=defaults.thinking_level,
    )
    tool_execution = choose(
        "tool_execution",
        ("options", options.tool_execution),
        ("workspace", settings.tool_execution if settings is not None else None),
        default=defaults.tool_execution,
    )
    max_tool_calls_per_turn = choose(
        "max_tool_calls_per_turn",
        ("options", options.max_tool_calls_per_turn),
        ("workspace", settings.max_tool_calls_per_turn if settings is not None else None),
        default=defaults.max_tool_calls_per_turn,
    )
    max_context_messages = choose(
        "max_context_messages",
        ("options", options.max_context_messages),
        ("workspace", settings.max_context_messages if settings is not None else None),
        default=defaults.max_context_messages,
    )
    retain_recent_messages = choose(
        "retain_recent_messages",
        ("options", options.retain_recent_messages),
        ("workspace", settings.retain_recent_messages if settings is not None else None),
        default=defaults.retain_recent_messages,
    )
    max_context_tokens = choose(
        "max_context_tokens",
        ("options", options.max_context_tokens),
        ("workspace", settings.max_context_tokens if settings is not None else None),
        default=defaults.max_context_tokens,
    )
    retry_enabled = choose(
        "retry_enabled",
        ("options", options.retry_enabled),
        ("workspace", settings.retry_enabled if settings is not None else None),
        default=defaults.retry_enabled,
    )
    max_retries = choose(
        "max_retries",
        ("options", options.max_retries),
        ("workspace", settings.max_retries if settings is not None else None),
        default=defaults.max_retries,
    )
    retry_base_delay_ms = choose(
        "retry_base_delay_ms",
        ("options", options.retry_base_delay_ms),
        ("workspace", settings.retry_base_delay_ms if settings is not None else None),
        default=defaults.retry_base_delay_ms,
    )
    read_only_mode = choose(
        "read_only_mode",
        ("options", options.read_only_mode),
        ("workspace", settings.read_only_mode if settings is not None else None),
        default=defaults.read_only_mode,
    )
    tool_permission_mode = choose(
        "tool_permission_mode",
        ("options", options.tool_permission_mode),
        ("workspace", settings.tool_permission_mode if settings is not None else None),
        default=("read-only" if read_only_mode else defaults.tool_permission_mode),
    )
    if read_only_mode:
        tool_permission_mode = "read-only"
    block_dangerous_bash = choose(
        "block_dangerous_bash",
        ("options", options.block_dangerous_bash),
        ("workspace", settings.block_dangerous_bash if settings is not None else None),
        default=defaults.block_dangerous_bash,
    )
    bash_allow_patterns = choose(
        "bash_allow_patterns",
        ("options", options.bash_allow_patterns),
        ("workspace", settings.bash_allow_patterns if settings is not None else None),
        default=defaults.bash_allow_patterns,
    )
    bash_block_patterns = choose(
        "bash_block_patterns",
        ("options", options.bash_block_patterns),
        ("workspace", settings.bash_block_patterns if settings is not None else None),
        default=defaults.bash_block_patterns,
    )
    edit_require_unique_match = choose(
        "edit_require_unique_match",
        ("options", options.edit_require_unique_match),
        ("workspace", settings.edit_require_unique_match if settings is not None else None),
        default=defaults.edit_require_unique_match,
    )
    extension_paths = choose(
        "extension_paths",
        ("options", options.extension_paths),
        ("workspace", settings.extension_paths if settings is not None else None),
        default=defaults.extension_paths,
    )
    skill_paths = choose(
        "skill_paths",
        ("options", options.skill_paths),
        ("workspace", settings.skill_paths if settings is not None else None),
        default=defaults.skill_paths,
    )
    mcp_servers = choose(
        "mcp_servers",
        ("options", options.mcp_servers),
        ("workspace", settings.mcp_servers if settings is not None else None),
        default=defaults.mcp_servers,
    )
    prompt_guidelines = choose(
        "prompt_guidelines",
        ("options", options.prompt_guidelines),
        ("workspace", settings.prompt_guidelines if settings is not None else None),
        default=defaults.prompt_guidelines,
    )
    append_system_prompt = choose(
        "append_system_prompt",
        ("options", options.append_system_prompt),
        ("workspace", settings.append_system_prompt if settings is not None else None),
        default=defaults.append_system_prompt,
    )
    prompt_debug_sources = choose(
        "prompt_debug_sources",
        ("options", options.prompt_debug_sources),
        ("workspace", settings.prompt_debug_sources if settings is not None else None),
        default=defaults.prompt_debug_sources,
    )
    tool_snippets = choose(
        "tool_snippets",
        ("options", options.tool_snippets),
        ("workspace", settings.tool_snippets if settings is not None else None),
        default=defaults.tool_snippets,
    )
    enabled_builtin_tools = choose(
        "enabled_builtin_tools",
        ("options", options.enabled_builtin_tools),
        ("workspace", resources.enabled_tools if resources is not None else None),
        default=defaults.enabled_builtin_tools,
    )
    shell_timeout_seconds = choose(
        "shell_timeout_seconds",
        ("options", options.shell_timeout_seconds),
        ("workspace", settings.shell_timeout_seconds if settings is not None else None),
        default=defaults.shell_timeout_seconds,
    )
    shell_max_timeout_seconds = choose(
        "shell_max_timeout_seconds",
        ("options", options.shell_max_timeout_seconds),
        ("workspace", settings.shell_max_timeout_seconds if settings is not None else None),
        default=defaults.shell_max_timeout_seconds,
    )
    shell_stdout_limit = choose(
        "shell_stdout_limit",
        ("options", options.shell_stdout_limit),
        ("workspace", settings.shell_stdout_limit if settings is not None else None),
        default=defaults.shell_stdout_limit,
    )
    shell_stderr_limit = choose(
        "shell_stderr_limit",
        ("options", options.shell_stderr_limit),
        ("workspace", settings.shell_stderr_limit if settings is not None else None),
        default=defaults.shell_stderr_limit,
    )
    shell_allowed_env = choose(
        "shell_allowed_env",
        ("options", options.shell_allowed_env),
        ("workspace", settings.shell_allowed_env if settings is not None else None),
        default=defaults.shell_allowed_env,
    )

    return ResolvedRuntimeConfig(
        system_prompt=system_prompt,
        thinking_level=thinking_level,
        tool_execution=tool_execution,
        max_tool_calls_per_turn=max_tool_calls_per_turn,
        max_context_messages=max_context_messages,
        retain_recent_messages=retain_recent_messages,
        max_context_tokens=max_context_tokens,
        retry_enabled=retry_enabled,
        max_retries=max_retries,
        retry_base_delay_ms=retry_base_delay_ms,
        read_only_mode=read_only_mode,
        tool_permission_mode=tool_permission_mode,
        block_dangerous_bash=block_dangerous_bash,
        bash_allow_patterns=bash_allow_patterns,
        bash_block_patterns=bash_block_patterns,
        edit_require_unique_match=edit_require_unique_match,
        extension_paths=extension_paths,
        skill_paths=skill_paths,
        mcp_servers=mcp_servers,
        prompt_guidelines=prompt_guidelines,
        append_system_prompt=append_system_prompt,
        prompt_debug_sources=prompt_debug_sources,
        tool_snippets=tool_snippets,
        enabled_builtin_tools=enabled_builtin_tools,
        shell_timeout_seconds=min(
            max(1, shell_timeout_seconds),
            min(120, max(1, shell_max_timeout_seconds)),
        ),
        shell_max_timeout_seconds=min(120, max(1, shell_max_timeout_seconds)),
        shell_stdout_limit=shell_stdout_limit,
        shell_stderr_limit=shell_stderr_limit,
        shell_allowed_env=shell_allowed_env,
        sources=sources,
    )
