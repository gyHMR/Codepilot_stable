from __future__ import annotations

"""
Runtime 对外类型定义模块。

定义了创建 Agent 会话所需的配置选项 CreateAgentSessionOptions，
以及运行模式、输出/输入函数等辅助类型。

装配产物（Runtime 装配优化引入）：
- RuntimeDiagnostic: 装配诊断信息
- ResolvedRuntimeProfile: 统一运行时配置
- CapabilityCatalog: 能力目录
- RuntimeAssembly: 完整装配产物
"""

import asyncio
from dataclasses import dataclass, field
from types import MappingProxyType
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal, Mapping, Optional, cast

from codepilot.protocols import Message, Model
from codepilot.core import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentMessage,
    BeforeToolCallContext,
    BeforeToolCallResult,
    StreamFn,
    ToolExecutionMode,
)
from codepilot.extensions.types import LifecycleHook, RegisteredCommand
from codepilot.sessions.context.repository_context import RepositoryBootstrap
from codepilot.sessions.types import AgentSessionOptions, ConvertToLlmFn
from codepilot.tools import AgentTool, ToolMetadata
from codepilot.tools.approval import ApprovalProvider

if TYPE_CHECKING:
    from codepilot.sessions.session import AgentSession


# ── 配置分组（阶段二引入） ────────────────────────────────────────

@dataclass(frozen=True)
class ModelSelection:
    """模型选择配置。

    描述使用哪个模型以及如何连接。

    Attributes:
        provider: provider 名称（如 deepseek、openai）。
        model_id: 模型 ID（如 deepseek-chat）。
    """
    provider: str | None = None
    model_id: str | None = None

    @property
    def display_id(self) -> str:
        """返回显示用的模型标识（provider/model_id）。"""
        if self.provider and self.model_id:
            return f"{self.provider}/{self.model_id}"
        return self.model_id or "(not configured)"


@dataclass
class CreateAgentSessionOptions:
    """创建 Agent 会话的友好配置选项。

    支持三种模型指定方式（按优先级）：
    1) 直接传 model 对象（最高优先级）。
    2) 传 provider + model_id（由工厂自动从内置目录解析）。
    3) 传入已有 session_id（工厂从会话元数据恢复 provider/model_id/system_prompt）。

    Attributes:
        workspace_dir: 工作区目录路径。
        model: 直接指定的模型对象（可选）。
        provider: provider 名称（可选，与 model_id 配合使用）。
        model_id: 模型 ID（可选，与 provider 配合使用）。
        get_api_key: API Key 获取函数（可选）。
        system_prompt: 系统提示词（可选，覆盖默认值）。
        tools: 额外的工具列表。
        session_id: 已有会话 ID（用于恢复会话）。
        messages: 初始消息列表。
        thinking_level: 推理思考级别。
        tool_execution: 工具执行模式（parallel/sequential）。
        load_workspace_resources: 是否加载工作区资源文件。
        enabled_builtin_tools: 启用的内置工具名称列表。
        max_context_messages: 上下文消息数量上限。
        max_context_tokens: 上下文 token 数量上限。
        retain_recent_messages: 压缩时保留的最近消息数。
        summary_builder: 自定义摘要构建器。
        retry_enabled: 是否启用重试。
        max_retries: 最大重试次数。
        retry_base_delay_ms: 重试基础延迟（毫秒）。
        read_only_mode: 是否为只读模式（禁止文件修改）。
        block_dangerous_bash: 是否阻止危险的 bash 命令。
        bash_allow_patterns: bash 命令白名单模式。
        bash_block_patterns: bash 命令黑名单模式。
        edit_require_unique_match: edit 工具是否要求唯一匹配。
        prompt_guidelines: 额外的提示词准则。
        append_system_prompt: 追加到系统提示词末尾的文本。
        tool_snippets: 工具说明片段（用于系统提示词）。
        extension_paths: 扩展加载路径列表。
        skill_paths: 技能加载路径列表。
        prompt_debug_sources: 是否在提示词中包含调试来源信息。
        mcp_servers: MCP 服务器配置列表。
        mcp_client: MCP 客户端实例。
        extension_commands: 已注册的扩展命令。
        before_prompt_hooks: 提示词执行前的生命周期钩子。
        after_prompt_hooks: 提示词执行后的生命周期钩子。
        before_tool_call: 工具调用前的拦截钩子。
        after_tool_call: 工具调用后的拦截钩子。
        stream_fn: Session 级 LLM 流函数，主要用于确定性 Harness Eval。
    """

    workspace_dir: str | Path
    model: Optional[Model] = None
    provider: Optional[str] = None
    model_id: Optional[str] = None
    get_api_key: Optional[Callable[[str], str | None | Awaitable[str | None]]] = None
    system_prompt: Optional[str] = None
    tools: list[AgentTool] = field(default_factory=list)
    session_id: Optional[str] = None
    messages: list[AgentMessage] = field(default_factory=list)
    thinking_level: Optional[str] = None
    tool_execution: Optional[ToolExecutionMode] = None
    max_tool_calls_per_turn: Optional[int] = None
    context_governance_enabled: bool = True
    memory_enabled: bool = True
    task_control_enabled: bool = True
    load_workspace_resources: bool = True
    enabled_builtin_tools: Optional[list[str]] = None
    max_context_messages: Optional[int] = None
    max_context_tokens: Optional[int] = None
    retain_recent_messages: Optional[int] = None
    summary_builder: Optional[Callable[[list[Message]], str]] = None
    retry_enabled: Optional[bool] = None
    max_retries: Optional[int] = None
    retry_base_delay_ms: Optional[int] = None
    read_only_mode: Optional[bool] = None
    tool_permission_mode: Optional[Literal["read-only", "workspace-write", "ask"]] = None
    block_dangerous_bash: Optional[bool] = None
    bash_allow_patterns: Optional[list[str]] = None
    bash_block_patterns: Optional[list[str]] = None
    edit_require_unique_match: Optional[bool] = None
    prompt_guidelines: Optional[list[str]] = None
    append_system_prompt: Optional[str] = None
    tool_snippets: Optional[dict[str, str]] = None
    extension_paths: Optional[list[str]] = None
    skill_paths: Optional[list[str]] = None
    prompt_debug_sources: Optional[bool] = None
    mcp_servers: Optional[list[dict[str, Any]]] = None
    mcp_client: Any | None = None
    approval_provider: ApprovalProvider | None = None
    shell_timeout_seconds: Optional[int] = None
    shell_max_timeout_seconds: Optional[int] = None
    shell_stdout_limit: Optional[int] = None
    shell_stderr_limit: Optional[int] = None
    shell_allowed_env: Optional[list[str]] = None
    extension_commands: dict[str, RegisteredCommand] = field(default_factory=dict)
    before_prompt_hooks: list[LifecycleHook] = field(default_factory=list)
    after_prompt_hooks: list[LifecycleHook] = field(default_factory=list)
    before_tool_call: Optional[
        Callable[[BeforeToolCallContext, Any | None], BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None]]
    ] = None
    after_tool_call: Optional[
        Callable[[AfterToolCallContext, Any | None], AfterToolCallResult | None | Awaitable[AfterToolCallResult | None]]
    ] = None
    stream_fn: StreamFn | None = None


RuntimePermissionMode = Literal["read-only", "workspace-write", "ask"]
RuntimeDiagnosticSeverity = Literal["info", "warning", "error"]
ConfigSourceKind = Literal["cli", "session", "project", "user", "default"]
ActiveRunStatus = Literal["running", "completed", "failed", "aborted"]
RegisteredToolSource = Literal["builtin", "caller", "extension", "mcp"]
_RUNTIME_PERMISSION_MODES = frozenset({"read-only", "workspace-write", "ask"})
_RUNTIME_DIAGNOSTIC_SEVERITIES = frozenset({"info", "warning", "error"})
_CONFIG_SOURCE_KINDS = frozenset({"cli", "session", "project", "user", "default"})
_ACTIVE_RUN_STATUSES = frozenset({"running", "completed", "failed", "aborted"})
_REGISTERED_TOOL_SOURCES = frozenset({"builtin", "caller", "extension", "mcp"})


# ── 装配产物类型（Runtime 装配优化引入） ──────────────────────────

@dataclass(frozen=True)
class RuntimeDiagnostic:
    """装配诊断信息。

    记录装配过程中的警告、错误和信息，用于 CLI 启动展示和调试。

    Attributes:
        severity: 严重级别（info/warning/error）。
        code: 诊断代码（如 tool.name_conflict）。
        message: 诊断消息。
        source: 来源（可选）。
    """
    severity: RuntimeDiagnosticSeverity
    code: str
    message: str
    source: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "severity",
            _ensure_runtime_diagnostic_severity(self.severity),
        )
        object.__setattr__(
            self,
            "code",
            _require_runtime_text(self.code, field_name="code"),
        )
        object.__setattr__(
            self,
            "message",
            _require_runtime_text(self.message, field_name="message"),
        )
        object.__setattr__(
            self,
            "source",
            _optional_runtime_text(self.source, field_name="source"),
        )


@dataclass(frozen=True)
class ConfigValueSource:
    """配置值来源。

    Attributes:
        kind: 来源类型（cli/session/project/user/default）。
        location: 具体位置（可选，如文件路径）。
    """
    kind: ConfigSourceKind
    location: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            _ensure_config_source_kind(self.kind),
        )
        object.__setattr__(
            self,
            "location",
            _optional_runtime_text(self.location, field_name="location"),
        )


@dataclass(frozen=True)
class ResolvedRuntimeProfile:
    """统一运行时配置。

    汇总所有配置的最终值和来源，供 CLI status/config explain 查询。

    Attributes:
        model: 解析后的模型对象。
        credential_source: 凭证来源（env/local-file/missing）。
        credential_location: 凭证位置（可选，如环境变量名）。
        permission_mode: 工具权限模式。
        sources: 每个配置项的来源。
    """
    model: Model
    credential_source: str
    credential_location: str | None = None
    permission_mode: RuntimePermissionMode = "workspace-write"
    sources: Mapping[str, ConfigValueSource] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model, Model):
            raise TypeError("ResolvedRuntimeProfile.model must be Model")
        object.__setattr__(
            self,
            "credential_source",
            _require_runtime_text(
                self.credential_source,
                field_name="credential_source",
            ),
        )
        object.__setattr__(
            self,
            "credential_location",
            _optional_runtime_text(
                self.credential_location,
                field_name="credential_location",
            ),
        )
        object.__setattr__(
            self,
            "permission_mode",
            _ensure_runtime_permission_mode(self.permission_mode),
        )
        object.__setattr__(
            self,
            "sources",
            _copy_config_sources(self.sources),
        )


@dataclass(frozen=True)
class ResolvedConfigValue:
    """供接口层展示的单个已解析配置值。"""

    key: str
    value: Any
    source: ConfigValueSource

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "key",
            _require_runtime_text(self.key, field_name="key"),
        )
        if not isinstance(self.source, ConfigValueSource):
            raise TypeError("ResolvedConfigValue.source must be ConfigValueSource")


@dataclass(frozen=True)
class RegisteredTool:
    """已注册的工具信息。

    Attributes:
        name: 工具名称。
        tool: 工具对象。
        metadata: 工具执行元数据。
        source: 来源（builtin/caller/extension/mcp）。
        origin: 具体来源（如扩展路径）。
    """
    name: str
    tool: AgentTool
    metadata: ToolMetadata | None
    source: RegisteredToolSource
    origin: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _require_runtime_text(self.name, field_name="tool name"),
        )
        if not isinstance(self.tool, AgentTool):
            raise TypeError("RegisteredTool.tool must be AgentTool")
        if self.metadata is not None and not isinstance(self.metadata, ToolMetadata):
            raise TypeError("RegisteredTool.metadata must be ToolMetadata or None")
        object.__setattr__(
            self,
            "source",
            _ensure_registered_tool_source(self.source),
        )
        object.__setattr__(
            self,
            "origin",
            _optional_runtime_text(self.origin, field_name="tool origin"),
        )


@dataclass(frozen=True)
class CapabilityCatalog:
    """能力目录。

    统一表示工具和命令。

    Attributes:
        tools: 已注册的工具列表。
        commands: 已注册的命令字典。
    """
    tools: tuple[RegisteredTool, ...] = field(default_factory=tuple)
    commands: Mapping[str, RegisteredCommand] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tools",
            _copy_registered_tools(self.tools),
        )
        object.__setattr__(
            self,
            "commands",
            _copy_registered_commands(self.commands),
        )


@dataclass(frozen=True)
class RuntimeAssembly:
    """完整装配产物。

    保存装配过程的所有结果，供 RuntimeService 和 CLI 查询。

    Attributes:
        session_options: 最终的会话选项。
        profile: 统一运行时配置。
        repository: 仓库引导信息。
        capabilities: 能力目录。
        diagnostics: 装配诊断列表。
    """
    session_options: AgentSessionOptions
    profile: ResolvedRuntimeProfile
    repository: RepositoryBootstrap
    capabilities: CapabilityCatalog
    diagnostics: tuple[RuntimeDiagnostic, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.session_options, AgentSessionOptions):
            raise TypeError("RuntimeAssembly.session_options must be AgentSessionOptions")
        if not isinstance(self.profile, ResolvedRuntimeProfile):
            raise TypeError("RuntimeAssembly.profile must be ResolvedRuntimeProfile")
        if not isinstance(self.repository, RepositoryBootstrap):
            raise TypeError("RuntimeAssembly.repository must be RepositoryBootstrap")
        if not isinstance(self.capabilities, CapabilityCatalog):
            raise TypeError("RuntimeAssembly.capabilities must be CapabilityCatalog")
        object.__setattr__(
            self,
            "diagnostics",
            _copy_runtime_diagnostics(self.diagnostics),
        )


@dataclass(frozen=True)
class SessionHandle:
    """RuntimeService 创建并注册的会话句柄。"""

    session_id: str
    session: AgentSession
    assembly: RuntimeAssembly

    def __post_init__(self) -> None:
        session_id = _require_runtime_text(
            self.session_id,
            field_name="session_id",
        )
        if not isinstance(self.assembly, RuntimeAssembly):
            raise TypeError("SessionHandle.assembly must be RuntimeAssembly")
        session_session_id = _session_identity(self.session)
        if session_session_id != session_id:
            raise ValueError(
                "SessionHandle.session_id must match session.session_id"
            )
        assembly_session_id = _require_runtime_text(
            self.assembly.session_options.session_id,
            field_name="assembly.session_options.session_id",
        )
        if assembly_session_id != session_id:
            raise ValueError(
                "SessionHandle.session_id must match assembly.session_options.session_id"
            )
        object.__setattr__(self, "session_id", session_id)


@dataclass(frozen=True)
class UserInput:
    """发送给会话的用户输入。"""

    text: str
    images: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("UserInput.text must be a string")
        text = self.text.strip()
        if not text:
            raise ValueError("UserInput.text is required")
        object.__setattr__(self, "text", text)
        if self.images is None:
            return
        images: list[str] = []
        for image in self.images:
            if not isinstance(image, str):
                raise TypeError("UserInput.images must contain strings")
            normalized = image.strip()
            if not normalized:
                raise ValueError("UserInput.images cannot contain empty image paths")
            images.append(normalized)
        object.__setattr__(self, "images", tuple(images))


@dataclass(frozen=True)
class SessionStatus:
    """供接口层展示的会话状态快照。"""

    session_id: str
    model_id: str
    workspace: str
    permission_mode: RuntimePermissionMode
    message_count: int
    leaf_id: str
    is_running: bool = False
    credential_source: str = "unknown"
    warnings: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "session_id",
            _require_runtime_text(self.session_id, field_name="session_id"),
        )
        object.__setattr__(
            self,
            "model_id",
            _require_runtime_text(self.model_id, field_name="model_id"),
        )
        object.__setattr__(
            self,
            "workspace",
            _require_runtime_text(self.workspace, field_name="workspace"),
        )
        object.__setattr__(
            self,
            "permission_mode",
            _ensure_runtime_permission_mode(self.permission_mode),
        )
        object.__setattr__(
            self,
            "message_count",
            _ensure_non_negative_runtime_int(
                self.message_count,
                field_name="message_count",
            ),
        )
        object.__setattr__(
            self,
            "leaf_id",
            _require_runtime_text(self.leaf_id, field_name="leaf_id"),
        )
        if not isinstance(self.is_running, bool):
            raise TypeError("SessionStatus.is_running must be bool")
        object.__setattr__(
            self,
            "credential_source",
            _require_runtime_text(
                self.credential_source,
                field_name="credential_source",
            ),
        )
        object.__setattr__(
            self,
            "warnings",
            _clean_runtime_warnings(self.warnings),
        )


@dataclass
class ActiveRun:
    """RuntimeService 内部的活动运行记录。"""

    run_id: str
    session_id: str
    task: asyncio.Task[Any] | None = None
    started_at: float = 0
    status: ActiveRunStatus = "running"

    def __post_init__(self) -> None:
        _ensure_active_run_status(self.status)


def _ensure_active_run_status(value: object) -> ActiveRunStatus:
    if value not in _ACTIVE_RUN_STATUSES:
        raise ValueError(f"Unknown active run status: {value}")
    return cast(ActiveRunStatus, value)


def _ensure_registered_tool_source(value: object) -> RegisteredToolSource:
    if isinstance(value, str):
        value = value.strip()
    if value not in _REGISTERED_TOOL_SOURCES:
        raise ValueError(f"Unknown tool source: {value}")
    return cast(RegisteredToolSource, value)


def _ensure_runtime_permission_mode(value: object) -> RuntimePermissionMode:
    if value not in _RUNTIME_PERMISSION_MODES:
        raise ValueError(f"Unknown permission_mode: {value}")
    return cast(RuntimePermissionMode, value)


def _ensure_runtime_diagnostic_severity(
    value: object,
) -> RuntimeDiagnosticSeverity:
    if isinstance(value, str):
        value = value.strip()
    if value not in _RUNTIME_DIAGNOSTIC_SEVERITIES:
        raise ValueError(f"Unknown diagnostic severity: {value}")
    return cast(RuntimeDiagnosticSeverity, value)


def _ensure_config_source_kind(value: object) -> ConfigSourceKind:
    if isinstance(value, str):
        value = value.strip()
    if value not in _CONFIG_SOURCE_KINDS:
        raise ValueError(f"Unknown config source kind: {value}")
    return cast(ConfigSourceKind, value)


def _require_runtime_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_runtime_text(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = value.strip()
    return text or None


def _ensure_non_negative_runtime_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} cannot be negative")
    return value


def _clean_runtime_warnings(
    warnings: object,
) -> tuple[str, ...] | None:
    if warnings is None:
        return None
    if isinstance(warnings, (str, bytes)):
        raise TypeError("SessionStatus.warnings must be a sequence of strings")
    cleaned: list[str] = []
    for warning in warnings:
        if not isinstance(warning, str):
            raise TypeError("SessionStatus.warnings must contain strings")
        text = warning.strip()
        if text:
            cleaned.append(text)
    return tuple(cleaned)


def _copy_config_sources(
    sources: object,
) -> Mapping[str, ConfigValueSource]:
    if not isinstance(sources, Mapping):
        raise TypeError("ResolvedRuntimeProfile.sources must be a mapping")
    copied: dict[str, ConfigValueSource] = {}
    for key, source in sources.items():
        clean_key = _require_runtime_text(key, field_name="source key")
        if not isinstance(source, ConfigValueSource):
            raise TypeError(
                "ResolvedRuntimeProfile.sources values must be ConfigValueSource"
            )
        copied[clean_key] = source
    return MappingProxyType(copied)


def _copy_registered_tools(
    tools: object,
) -> tuple[RegisteredTool, ...]:
    if isinstance(tools, (str, bytes)):
        raise TypeError("CapabilityCatalog.tools must be a sequence of RegisteredTool")
    copied: list[RegisteredTool] = []
    for tool in tools:
        if not isinstance(tool, RegisteredTool):
            raise TypeError(
                "CapabilityCatalog.tools values must be RegisteredTool"
            )
        copied.append(tool)
    return tuple(copied)


def _copy_registered_commands(
    commands: object,
) -> Mapping[str, RegisteredCommand]:
    if not isinstance(commands, Mapping):
        raise TypeError("CapabilityCatalog.commands must be a mapping")
    copied: dict[str, RegisteredCommand] = {}
    for key, command in commands.items():
        clean_key = _require_runtime_text(key, field_name="command name")
        if not isinstance(command, RegisteredCommand):
            raise TypeError(
                "CapabilityCatalog.commands values must be RegisteredCommand"
            )
        copied[clean_key] = command
    return MappingProxyType(copied)


def _copy_runtime_diagnostics(
    diagnostics: object,
) -> tuple[RuntimeDiagnostic, ...]:
    if isinstance(diagnostics, (str, bytes)):
        raise TypeError("RuntimeAssembly.diagnostics must be a sequence of RuntimeDiagnostic")
    copied: list[RuntimeDiagnostic] = []
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, RuntimeDiagnostic):
            raise TypeError(
                "RuntimeAssembly.diagnostics values must be RuntimeDiagnostic"
            )
        copied.append(diagnostic)
    return tuple(copied)


def _session_identity(session: object) -> str:
    try:
        value = getattr(session, "session_id")
    except AttributeError as exc:
        raise TypeError("SessionHandle.session must expose session_id") from exc
    return _require_runtime_text(value, field_name="session.session_id")
