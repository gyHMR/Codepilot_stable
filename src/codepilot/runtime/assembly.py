from __future__ import annotations

# 新手导读：assembly.py 是 runtime 启动装配的唯一主线。
# 关注点：打开会话时，输入 DTO、资源加载、配置解析、模型解析、工具装配、prompt 和 hook 都在这里串起来。

# ---- assembly input DTO ----

# 新手导读：assembly_input.py 定义 runtime 装配阶段的内部输入。
# 关注点：interfaces 只能提交 SessionOpenIntent；这里的类型只供 runtime assembly/bootstrap 使用。

"""Runtime assembly input types."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Optional

from codepilot.core.task import PlanningBudgetProfile, TaskMode
from codepilot.core.contracts import (
    AgentMessage,
    ToolExecutionMode,
)
from codepilot.llm.provider_types import ProviderSimpleStreamFn
from codepilot.protocols import Model
from codepilot.protocols.commands import (
    AfterToolCallContext,
    AfterToolCallResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    LifecycleHook,
    RegisteredCommand,
)
from codepilot.tools import AgentTool
from codepilot.tools.policy import ApprovalProvider


@dataclass
class RuntimeAssemblyIntent:
    """Runtime-internal intent for assembling a runnable session.

    `SessionOpenIntent` is the public application intent. This richer type is
    produced inside runtime and includes executable tools, hooks, model
    overrides, and resource-loading controls needed by the assembly pipeline.
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
    memory_enabled: bool = True
    task_control_enabled: bool = True
    task_mode: TaskMode | None = None
    planning_budget_profile: PlanningBudgetProfile | None = None
    max_task_replans_per_run: Optional[int] = None
    load_workspace_resources: bool = True
    enabled_builtin_tools: Optional[list[str]] = None
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
        Callable[
            [BeforeToolCallContext, Any | None],
            BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None],
        ]
    ] = None
    after_tool_call: Optional[
        Callable[
            [AfterToolCallContext, Any | None],
            AfterToolCallResult | None | Awaitable[AfterToolCallResult | None],
        ]
    ] = None
    stream_fn: ProviderSimpleStreamFn | None = None


RuntimePermissionMode = Literal["read-only", "workspace-write", "ask"]



# ---- assembly result DTO ----

# 新手导读：assembly_types.py 只描述 runtime 内部装配产物。
# 关注点：这些类型可以持有 ToolRuntime 等 live object，但不会暴露给 interfaces。

"""Internal runtime assembly records.

RuntimeGateway exposes UserAction/RuntimeFrame and read-only views. The richer
objects in this module are the private result of wiring model, tools, commands,
hooks, repository context, and session options together.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, cast

from codepilot.core.task import (
    PlanningBudgetProfile,
    TaskMode,
    ensure_planning_budget_profile,
    ensure_task_mode,
)
from codepilot.protocols import Model
from codepilot.protocols.commands import RegisteredCommand
from codepilot.sessions.contracts import SessionOptions
from codepilot.sessions.storage import RepositoryBootstrap
from codepilot.tools import AgentTool, ToolMetadata
from codepilot.tools.engine import ToolRuntime



RuntimeDiagnosticSeverity = Literal["info", "warning", "error"]
ConfigSourceKind = Literal["cli", "session", "project", "user", "default"]
RegisteredToolSource = Literal["builtin", "caller", "extension", "mcp"]

_RUNTIME_PERMISSION_MODES = frozenset({"read-only", "workspace-write", "ask"})
_RUNTIME_DIAGNOSTIC_SEVERITIES = frozenset({"info", "warning", "error"})
_CONFIG_SOURCE_KINDS = frozenset({"cli", "session", "project", "user", "default"})
_REGISTERED_TOOL_SOURCES = frozenset({"builtin", "caller", "extension", "mcp"})


@dataclass(frozen=True)
class RuntimeDiagnostic:
    """Assembly diagnostic emitted during runtime wiring."""

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
    """Where a resolved runtime config value came from."""

    kind: ConfigSourceKind
    location: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _ensure_config_source_kind(self.kind))
        object.__setattr__(
            self,
            "location",
            _optional_runtime_text(self.location, field_name="location"),
        )


@dataclass(frozen=True)
class ResolvedRuntimeProfile:
    """Resolved runtime config used for status/config explanation views."""

    model: Model
    credential_source: str
    credential_location: str | None = None
    permission_mode: RuntimePermissionMode = "workspace-write"
    task_mode: TaskMode = "edit"
    planning_budget_profile: PlanningBudgetProfile = "balanced"
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
        object.__setattr__(self, "task_mode", ensure_task_mode(self.task_mode))
        object.__setattr__(
            self,
            "planning_budget_profile",
            ensure_planning_budget_profile(self.planning_budget_profile),
        )
        object.__setattr__(self, "sources", _copy_config_sources(self.sources))


@dataclass(frozen=True)
class ResolvedConfigValue:
    """Single resolved config value for interface rendering."""

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
    """Tool registered during runtime assembly."""

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
        object.__setattr__(self, "source", _ensure_registered_tool_source(self.source))
        object.__setattr__(
            self,
            "origin",
            _optional_runtime_text(self.origin, field_name="tool origin"),
        )


@dataclass(frozen=True)
class CapabilityCatalog:
    """Capabilities discovered while opening a runtime session."""

    tools: tuple[RegisteredTool, ...] = field(default_factory=tuple)
    commands: Mapping[str, RegisteredCommand] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tools", _copy_registered_tools(self.tools))
        object.__setattr__(self, "commands", _copy_registered_commands(self.commands))


@dataclass(frozen=True)
class RuntimeAssembly:
    """Private runtime wiring record for one open application session."""

    session_options: SessionOptions
    profile: ResolvedRuntimeProfile
    repository: RepositoryBootstrap
    capabilities: CapabilityCatalog
    tool_runtime: ToolRuntime
    model_port: Any
    tool_port: Any
    diagnostics: tuple[RuntimeDiagnostic, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.session_options, SessionOptions):
            raise TypeError("RuntimeAssembly.session_options must be SessionOptions")
        if not isinstance(self.profile, ResolvedRuntimeProfile):
            raise TypeError("RuntimeAssembly.profile must be ResolvedRuntimeProfile")
        if not isinstance(self.repository, RepositoryBootstrap):
            raise TypeError("RuntimeAssembly.repository must be RepositoryBootstrap")
        if not isinstance(self.capabilities, CapabilityCatalog):
            raise TypeError("RuntimeAssembly.capabilities must be CapabilityCatalog")
        if not isinstance(self.tool_runtime, ToolRuntime):
            raise TypeError("RuntimeAssembly.tool_runtime must be ToolRuntime")
        object.__setattr__(
            self,
            "diagnostics",
            _copy_runtime_diagnostics(self.diagnostics),
        )


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


def _copy_config_sources(sources: object) -> Mapping[str, ConfigValueSource]:
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


def _copy_registered_tools(tools: object) -> tuple[RegisteredTool, ...]:
    if isinstance(tools, (str, bytes)):
        raise TypeError("CapabilityCatalog.tools must be a sequence of RegisteredTool")
    copied: list[RegisteredTool] = []
    for tool in tools:
        if not isinstance(tool, RegisteredTool):
            raise TypeError("CapabilityCatalog.tools values must be RegisteredTool")
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
        raise TypeError(
            "RuntimeAssembly.diagnostics must be a sequence of RuntimeDiagnostic"
        )
    copied: list[RuntimeDiagnostic] = []
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, RuntimeDiagnostic):
            raise TypeError(
                "RuntimeAssembly.diagnostics values must be RuntimeDiagnostic"
            )
        copied.append(diagnostic)
    return tuple(copied)



# ---- workspace resources ----

# 新手导读：resources 负责加载工作区资源、会话元数据和项目配置文件。
# 关注点：它让 runtime 装配阶段不直接散落文件路径读取逻辑。

"""
工作区资源加载模块。

负责从工作区的 `.codepilot/` 目录加载配置文件：
1) settings.json: 模型和运行时参数配置
2) model.local.json: 自定义模型配置（本地模型或非内置 provider）
3) prompt.md: 自定义系统提示词
4) tools.json: 启用的内置工具列表
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from codepilot.core.task import PlanningBudgetProfile, TaskMode
from codepilot.core.contracts import ToolExecutionMode
from codepilot.protocols import Model, ModelCapabilities


# 支持的模型 API 协议集合
SUPPORTED_MODEL_APIS = {"openai-compatible", "anthropic-messages"}


@dataclass(frozen=True)
class WorkspaceModelConfig:
    """工作区自定义模型配置。

    从 `.codepilot/model.local.json` 加载，用于配置非内置的自定义模型。

    Attributes:
        api: API 协议标识（"openai-compatible" 或 "anthropic-messages"）。
        provider: provider 名称。
        model_id: 模型 ID。
        base_url: API 端点基础 URL。
        api_key: 明文 API Key（可选，优先使用环境变量）。
        api_key_env: API Key 对应的环境变量名（可选）。
        context_window: 上下文窗口大小（默认 128K）。
        max_tokens: 最大输出 token 数（默认 8192）。
        reasoning: 是否支持推理模式。
        vision: 是否支持图片输入。
    """

    api: str
    provider: str
    model_id: str
    base_url: str
    api_key: str | None = None
    api_key_env: str | None = None
    context_window: int = 128_000
    max_tokens: int = 8192
    reasoning: bool = False
    vision: bool = False

    def __post_init__(self) -> None:
        """初始化后校验：确保 API 协议、必填字段和数值范围有效。"""
        if self.api not in SUPPORTED_MODEL_APIS:
            raise ValueError(f"Unsupported API protocol: {self.api}")
        if not self.provider or not self.model_id or not self.base_url:
            raise ValueError("provider, model_id and base_url are required")
        if self.context_window <= 0 or self.max_tokens <= 0:
            raise ValueError("context_window and max_tokens must be positive")

    def to_model(self) -> Model:
        """将工作区模型配置转换为通用的 Model 对象。"""
        return Model(
            id=self.model_id,
            name=self.model_id,
            api=self.api,
            provider=self.provider,
            base_url=self.base_url,
            reasoning=self.reasoning,
            input=["text", "image"] if self.vision else ["text"],
            context_window=self.context_window,
            max_tokens=self.max_tokens,
            capabilities=ModelCapabilities(
                tools=True,
                vision=self.vision,
                streaming=True,
                reasoning=self.reasoning,
                system_prompt=True,
                tool_choice=self.api == "openai-compatible",
                parallel_tool_calls=self.api == "openai-compatible",
            ),
        )

    def resolve_api_key(self) -> str | None:
        """解析 API Key：优先从环境变量读取，其次使用明文配置。"""
        if self.api_key_env:
            value = os.getenv(self.api_key_env)
            if value:
                return value
        return self.api_key

    def build_api_key_resolver(self) -> Callable[[str], str | None]:
        """构建 API Key 解析器函数（忽略 provider 参数，始终返回本配置的 Key）。"""
        return lambda _provider: self.resolve_api_key()


@dataclass
class WorkspaceSettings:
    """工作区 settings.json 配置。

    所有字段都是可选的，未设置时使用 RuntimeDefaults 中的默认值。
    通过 WorkspaceResourceLoader._load_settings() 从 JSON 文件加载。

    Attributes:
        provider: provider 名称。
        model_id: 模型 ID。
        system_prompt: 自定义系统提示词。
        thinking_level: 推理级别。
        tool_execution: 工具执行模式。
        retry_enabled: 是否启用重试。
        max_retries: 最大重试次数。
        retry_base_delay_ms: 重试基础延迟（毫秒）。
        read_only_mode: 是否为只读模式。
        block_dangerous_bash: 是否阻止危险 bash 命令。
        bash_allow_patterns: bash 命令白名单。
        bash_block_patterns: bash 命令黑名单。
        edit_require_unique_match: edit 是否要求唯一匹配。
        prompt_guidelines: 额外的提示词准则。
        append_system_prompt: 追加到系统提示词末尾的文本。
        tool_snippets: 工具说明片段。
        extension_paths: 扩展加载路径。
        skill_paths: 技能加载路径。
        prompt_debug_sources: 是否包含调试来源信息。
        mcp_servers: MCP 服务器配置。
    """

    provider: Optional[str] = None
    model_id: Optional[str] = None
    system_prompt: Optional[str] = None
    thinking_level: Optional[str] = None
    tool_execution: Optional[ToolExecutionMode] = None
    task_mode: Optional[TaskMode] = None
    planning_budget_profile: Optional[PlanningBudgetProfile] = None
    max_tool_calls_per_turn: Optional[int] = None
    retry_enabled: Optional[bool] = None
    max_retries: Optional[int] = None
    retry_base_delay_ms: Optional[int] = None
    read_only_mode: Optional[bool] = None
    tool_permission_mode: Optional[str] = None
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
    shell_timeout_seconds: Optional[int] = None
    shell_max_timeout_seconds: Optional[int] = None
    shell_stdout_limit: Optional[int] = None
    shell_stderr_limit: Optional[int] = None
    shell_allowed_env: Optional[list[str]] = None


@dataclass
class WorkspaceResources:
    """工作区资源汇总。

    Attributes:
        settings: settings.json 中的配置。
        model: model.local.json 中的自定义模型配置（不存在时为 None）。
        prompt: prompt.md 中的自定义系统提示词（不存在时为 None）。
        enabled_tools: tools.json 中启用的工具列表（不存在时为 None）。
    """

    settings: WorkspaceSettings
    model: WorkspaceModelConfig | None
    prompt: Optional[str]
    enabled_tools: Optional[list[str]]


class WorkspaceResourceLoader:
    """工作区资源加载器。

    从工作区的 `.codepilot/` 目录加载所有配置文件，
    并进行类型安全的解析和校验。

    使用示例::

        loader = WorkspaceResourceLoader("/path/to/workspace")
        resources = loader.load()
        if resources.model:
            model = resources.model.to_model()
    """

    def __init__(self, workspace_dir: str | Path) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.resource_root = self.workspace_dir / ".codepilot"
        self.settings_file = self.resource_root / "settings.json"
        self.model_file = self.resource_root / "model.local.json"
        self.prompt_file = self.resource_root / "prompt.md"
        self.tools_file = self.resource_root / "tools.json"

    def load(self) -> WorkspaceResources:
        """加载所有工作区资源文件。"""
        return WorkspaceResources(
            settings=self._load_settings(),
            model=self._load_model(),
            prompt=self._load_prompt(),
            enabled_tools=self._load_tools(),
        )

    def _load_settings(self) -> WorkspaceSettings:
        """加载并解析 settings.json。"""
        if not self.settings_file.exists():
            return WorkspaceSettings()
        raw = self._safe_load_json(self.settings_file)
        if not isinstance(raw, dict):
            return WorkspaceSettings()

        # 校验 tool_execution 枚举值
        tool_execution = raw.get("tool_execution")
        if tool_execution not in {"parallel", "sequential"}:
            tool_execution = None
        raw_task_mode = raw.get("task_mode")
        task_mode = (
            raw_task_mode
            if isinstance(raw_task_mode, str)
            and raw_task_mode in {"read", "edit", "plan"}
            else None
        )
        raw_planning_budget_profile = raw.get("planning_budget_profile")
        planning_budget_profile = (
            raw_planning_budget_profile
            if isinstance(raw_planning_budget_profile, str)
            and raw_planning_budget_profile in {"conservative", "balanced", "wide"}
            else None
        )
        permission_mode = raw.get("tool_permission_mode")
        if permission_mode not in {"read-only", "workspace-write", "ask"}:
            permission_mode = None

        return WorkspaceSettings(
            provider=raw.get("provider") if isinstance(raw.get("provider"), str) else None,
            model_id=raw.get("model_id") if isinstance(raw.get("model_id"), str) else None,
            system_prompt=raw.get("system_prompt") if isinstance(raw.get("system_prompt"), str) else None,
            thinking_level=raw.get("thinking_level") if isinstance(raw.get("thinking_level"), str) else None,
            tool_execution=tool_execution,
            task_mode=task_mode,
            planning_budget_profile=planning_budget_profile,
            max_tool_calls_per_turn=self._to_positive_int(raw.get("max_tool_calls_per_turn")),
            retry_enabled=raw.get("retry_enabled") if isinstance(raw.get("retry_enabled"), bool) else None,
            max_retries=self._to_positive_int(raw.get("max_retries")),
            retry_base_delay_ms=self._to_positive_int(raw.get("retry_base_delay_ms")),
            read_only_mode=raw.get("read_only_mode") if isinstance(raw.get("read_only_mode"), bool) else None,
            tool_permission_mode=permission_mode,
            block_dangerous_bash=raw.get("block_dangerous_bash")
            if isinstance(raw.get("block_dangerous_bash"), bool)
            else None,
            bash_allow_patterns=self._to_string_list(raw.get("bash_allow_patterns")),
            bash_block_patterns=self._to_string_list(raw.get("bash_block_patterns")),
            edit_require_unique_match=raw.get("edit_require_unique_match")
            if isinstance(raw.get("edit_require_unique_match"), bool)
            else None,
            prompt_guidelines=self._to_string_list(raw.get("prompt_guidelines")),
            append_system_prompt=raw.get("append_system_prompt")
            if isinstance(raw.get("append_system_prompt"), str)
            else None,
            tool_snippets=self._to_string_map(raw.get("tool_snippets")),
            extension_paths=self._to_string_list(raw.get("extension_paths")),
            skill_paths=self._to_string_list(raw.get("skill_paths")),
            prompt_debug_sources=raw.get("prompt_debug_sources")
            if isinstance(raw.get("prompt_debug_sources"), bool)
            else None,
            mcp_servers=self._to_object_list(raw.get("mcp_servers")),
            shell_timeout_seconds=self._to_positive_int(raw.get("shell_timeout_seconds")),
            shell_max_timeout_seconds=self._to_positive_int(raw.get("shell_max_timeout_seconds")),
            shell_stdout_limit=self._to_positive_int(raw.get("shell_stdout_limit")),
            shell_stderr_limit=self._to_positive_int(raw.get("shell_stderr_limit")),
            shell_allowed_env=self._to_string_list(raw.get("shell_allowed_env")),
        )

    def _load_model(self) -> WorkspaceModelConfig | None:
        """加载并解析 model.local.json。"""
        if not self.model_file.exists():
            return None
        raw = self._safe_load_json(self.model_file)
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid model config JSON: {self.model_file}")
        try:
            return WorkspaceModelConfig(
                api=str(raw.get("api", "")),
                provider=str(raw.get("provider", "")),
                model_id=str(raw.get("model_id", "")),
                base_url=str(raw.get("base_url", "")),
                api_key=raw.get("api_key") if isinstance(raw.get("api_key"), str) else None,
                api_key_env=raw.get("api_key_env") if isinstance(raw.get("api_key_env"), str) else None,
                context_window=int(raw.get("context_window", 128_000)),
                max_tokens=int(raw.get("max_tokens", 8192)),
                reasoning=bool(raw.get("reasoning", False)),
                vision=bool(raw.get("vision", False)),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid model config {self.model_file}: {exc}") from exc

    def _load_prompt(self) -> Optional[str]:
        """加载 prompt.md 文件内容。"""
        if not self.prompt_file.exists():
            return None
        text = self.prompt_file.read_text(encoding="utf-8").strip()
        return text or None

    def _load_tools(self) -> Optional[list[str]]:
        """加载并解析 tools.json 中的 enabled 列表。"""
        if not self.tools_file.exists():
            return None
        raw = self._safe_load_json(self.tools_file)
        if not isinstance(raw, dict):
            return None
        enabled = raw.get("enabled")
        if not isinstance(enabled, list):
            return None
        return [item for item in enabled if isinstance(item, str)]

    @staticmethod
    def _safe_load_json(path: Path) -> Any:
        """安全加载 JSON 文件，解析失败时返回 None。"""
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _to_positive_int(value: Any) -> Optional[int]:
        """将值转换为正整数（排除 bool 类型），无效时返回 None。"""
        if isinstance(value, bool):
            return None
        if isinstance(value, int) and value > 0:
            return value
        return None

    @staticmethod
    def _to_string_list(value: Any) -> Optional[list[str]]:
        """将值转换为字符串列表，过滤非字符串元素。"""
        if not isinstance(value, list):
            return None
        return [item for item in value if isinstance(item, str)]

    @staticmethod
    def _to_string_map(value: Any) -> Optional[dict[str, str]]:
        """将值转换为字符串字典，过滤非字符串的键值。"""
        if not isinstance(value, dict):
            return None
        result: dict[str, str] = {}
        for k, v in value.items():
            if isinstance(k, str) and isinstance(v, str):
                result[k] = v
        return result

    @staticmethod
    def _to_object_list(value: Any) -> Optional[list[dict[str, Any]]]:
        """将值转换为字典列表，过滤非字典元素。"""
        if not isinstance(value, list):
            return None
        result: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                result.append(dict(item))
        return result

# ---- runtime config resolution ----

# 新手导读：配置解析把 CLI、会话恢复、工作区配置和默认值合并成 RuntimeConfig。
# 关注点：排查某个配置为什么生效时，要看 sources 如何记录来源。

"""
运行时配置解析模块。

负责将上层传入的 RuntimeAssemblyIntent 与工作区资源、
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

from codepilot.core.task import (
    PlanningBudgetProfile,
    TaskMode,
    ensure_planning_budget_profile,
    ensure_task_mode,
)
from codepilot.core.contracts import ToolExecutionMode
from codepilot.sessions.storage import SessionOpenMetadata, load_session_open_metadata


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
    task_mode: TaskMode = "edit"
    planning_budget_profile: PlanningBudgetProfile = "balanced"
    max_tool_calls_per_turn: int = 8
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
    restored_meta: SessionOpenMetadata | None


@dataclass(frozen=True)
class ResolvedRuntimeConfig:
    """最终解析完成的运行时配置。

    所有字段都经过多源合并（options -> 会话元数据 -> 工作区 -> 默认值），
    每个字段都确定了最终值和来源。

    Attributes:
        sources: 每个配置项的来源标识（如 "options"、"workspace"、"default"）。
        其余字段含义见 RuntimeDefaults 和 RuntimeAssemblyIntent。
    """

    system_prompt: str
    thinking_level: str
    tool_execution: ToolExecutionMode
    task_mode: TaskMode
    planning_budget_profile: PlanningBudgetProfile
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


def load_runtime_inputs(options: RuntimeAssemblyIntent) -> RuntimeInputs:
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
        restored_meta=load_session_open_metadata(workspace, options.session_id),
    )


def load_workspace_resources(options: RuntimeAssemblyIntent) -> tuple[Path, WorkspaceResources | None]:
    """加载工作区资源（settings.json、prompt.md 等）。

    Args:
        options: 创建会话的配置选项。

    Returns:
        (工作区路径, 工作区资源) 元组；load_workspace_resources=False 时资源为 None。
    """
    workspace = Path(options.workspace_dir)
    resources = WorkspaceResourceLoader(workspace).load() if options.load_workspace_resources else None
    return workspace, resources


def resolve_runtime_config(
    options: RuntimeAssemblyIntent,
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
    restored = inputs.restored_meta
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
        ("restored_session", restored.system_prompt if restored is not None else None),
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
    raw_task_mode = choose(
        "task_mode",
        ("options", options.task_mode),
        ("workspace", settings.task_mode if settings is not None else None),
        default=defaults.task_mode,
    )
    task_mode = ensure_task_mode(raw_task_mode)
    raw_planning_budget_profile = choose(
        "planning_budget_profile",
        ("options", options.planning_budget_profile),
        (
            "workspace",
            settings.planning_budget_profile if settings is not None else None,
        ),
        default=defaults.planning_budget_profile,
    )
    planning_budget_profile = ensure_planning_budget_profile(raw_planning_budget_profile)
    max_tool_calls_per_turn = choose(
        "max_tool_calls_per_turn",
        ("options", options.max_tool_calls_per_turn),
        ("workspace", settings.max_tool_calls_per_turn if settings is not None else None),
        default=defaults.max_tool_calls_per_turn,
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
    if task_mode == "read":
        read_only_mode = True
    tool_permission_mode = choose(
        "tool_permission_mode",
        ("options", options.tool_permission_mode),
        ("workspace", settings.tool_permission_mode if settings is not None else None),
        default=("read-only" if read_only_mode else defaults.tool_permission_mode),
    )
    if task_mode == "read" and options.tool_permission_mode not in {None, "read-only"}:
        raise ValueError("task_mode=read requires tool_permission_mode=read-only")
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
        task_mode=task_mode,
        planning_budget_profile=planning_budget_profile,
        max_tool_calls_per_turn=max_tool_calls_per_turn,
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

# ---- model resolution ----

# 新手导读：模型解析负责从选项、配置和内置目录中找出最终 Model。
# 关注点：它只决定“用哪个模型”，不负责真正调用模型。

"""
模型解析模块。

负责从多种来源中确定最终使用的模型，按优先级依次尝试：
1) options.model: 调用方直接指定的模型对象（最高优先级）
2) options.provider + options.model_id: 从内置模型目录解析
3) 恢复的会话元数据中的 provider + model_id
4) 工作区 model.local.json 中的自定义模型配置
5) 工作区 settings.json 中的 provider + model_id

以上均未匹配时抛出 ValueError。
"""

from dataclasses import dataclass
from typing import Awaitable, Callable

from codepilot.llm.catalog import get_model
from codepilot.protocols import Model



@dataclass(frozen=True)
class ResolvedModel:
    """解析完成的模型配置。

    Attributes:
        model: 解析得到的 Model 对象。
        get_api_key: API Key 获取函数（可选，为 None 时使用环境变量默认逻辑）。
    """

    model: Model
    get_api_key: Callable[[str], str | None | Awaitable[str | None]] | None = None
    source: ConfigValueSource = ConfigValueSource(kind="default")


def resolve_model(
    options: RuntimeAssemblyIntent,
    inputs: RuntimeInputs,
) -> ResolvedModel:
    """按优先级从多种来源解析模型配置。

    优先级顺序：
    1) options.model（直接指定）
    2) options.provider + options.model_id（从内置目录查找）
    3) 恢复的会话元数据中的 provider + model_id
    4) 工作区 model.local.json（自定义模型）
    5) 工作区 settings.json 中的 provider + model_id

    Args:
        options: 创建会话的配置选项。
        inputs: 运行时输入数据（含工作区资源和会话元数据）。

    Returns:
        ResolvedModel 对象，包含 Model 和 API Key 获取函数。

    Raises:
        ValueError: 所有来源都无法解析模型时抛出。
    """
    resources = inputs.resources

    # 优先级 1：直接指定的模型对象
    model = options.model
    if model is not None:
        return ResolvedModel(
            model=model,
            get_api_key=options.get_api_key,
            source=ConfigValueSource(kind="cli"),
        )

    # 优先级 2：provider + model_id（从内置目录查找）
    if options.provider and options.model_id:
        return ResolvedModel(
            model=get_model(options.provider, options.model_id),
            get_api_key=options.get_api_key,
            source=ConfigValueSource(kind="cli"),
        )

    # 优先级 3：恢复的会话元数据
    restored_meta = inputs.restored_meta
    provider = restored_meta.provider if restored_meta is not None else None
    model_id = restored_meta.model_id if restored_meta is not None else None
    if isinstance(provider, str) and isinstance(model_id, str):
        return ResolvedModel(
            model=get_model(provider, model_id),
            get_api_key=options.get_api_key,
            source=ConfigValueSource(
                kind="session",
                location=options.session_id,
            ),
        )

    # 优先级 4：工作区 model.local.json
    if resources and resources.model is not None:
        return ResolvedModel(
            model=resources.model.to_model(),
            get_api_key=options.get_api_key or resources.model.build_api_key_resolver(),
            source=ConfigValueSource(
                kind="project",
                location=".codepilot/model.local.json",
            ),
        )

    # 优先级 5：工作区 settings.json
    if resources and resources.settings.provider and resources.settings.model_id:
        return ResolvedModel(
            model=get_model(resources.settings.provider, resources.settings.model_id),
            get_api_key=options.get_api_key,
            source=ConfigValueSource(
                kind="project",
                location=".codepilot/settings.json",
            ),
        )

    # 所有来源均未匹配
    raise ValueError(
        "Unable to resolve model: create .codepilot/model.local.json "
        "or provide --model provider/model-id"
    )

# ---- tool assembly ----

# 新手导读：工具装配器负责合并内置、调用者、扩展和 MCP 工具，并创建 ToolRuntime。
# 关注点：read-only 模式下模型可见工具和注册表里的有效工具也在这里保持一致。

"""
工具组装模块。

负责将来自不同来源的工具合并为统一的工具列表：
1) 内置工具（ls、find、read、grep、edit、write、bash 等）
2) 调用方通过 options.tools 传入的自定义工具
3) 扩展加载的工具（通过 extension_paths）
4) MCP 代理工具（通过 mcp_servers 配置）

阶段C改进：
- 为工具记录来源和 metadata
- 增加单工具校验
- 增加冲突诊断
- Skill、MCP、Extension 错误统一进入 diagnostics
"""

from dataclasses import dataclass, field
from pathlib import Path

from codepilot.extensions import load_extensions, load_skills
from codepilot.extensions.mcp import create_mcp_proxy_tools, parse_mcp_tool_configs
from codepilot.extensions.types import LoadedExtensions
from codepilot.protocols import Tool
from codepilot.tools.builtins import (
    create_builtin_tools,
    get_builtin_tool_metadata,
)
from codepilot.tools.policy import PermissionPolicy
from codepilot.tools.policy import DeferredApprovalProvider
from codepilot.tools.authoring import infer_tool_metadata
from codepilot.tools.authoring import ToolRegistry
from codepilot.tools.engine import ToolRuntime
from codepilot.tools.workspace import ShellExecutionPolicy
from codepilot.tools.authoring import AgentTool



@dataclass(frozen=True)
class AssembledTools:
    """工具组装结果。

    Attributes:
        tools: 模型可见的工具描述列表（不含执行器）。
        registered_tools: 已注册的工具详细信息列表。
        tool_runtime: 统一工具运行时。
        loaded_extensions: 已加载的扩展信息（包含钩子、命令等）。
        loaded_skills: 已加载的技能信息（包含钩子、命令等）。
        diagnostics: 装配诊断列表。
    """
    tools: list[Tool]
    registered_tools: list[RegisteredTool]
    tool_runtime: ToolRuntime
    loaded_extensions: LoadedExtensions
    loaded_skills: LoadedExtensions
    diagnostics: list[RuntimeDiagnostic] = field(default_factory=list)


def validate_tool_definition(tool: AgentTool) -> list[RuntimeDiagnostic]:
    """校验单个工具定义。

    Args:
        tool: 要校验的工具。

    Returns:
        诊断列表（空表示校验通过）。
    """
    diagnostics: list[RuntimeDiagnostic] = []

    # 检查 name 非空
    if not tool.name or not tool.name.strip():
        diagnostics.append(RuntimeDiagnostic(
            severity="error",
            code="tool.invalid_name",
            message="Tool name is empty",
        ))

    # 检查 description 非空
    if not tool.description or not tool.description.strip():
        diagnostics.append(RuntimeDiagnostic(
            severity="warning",
            code="tool.missing_description",
            message=f"Tool '{tool.name}' has no description",
        ))

    # 检查 parameters 是对象形式的 JSON Schema
    if not isinstance(tool.parameters, dict):
        diagnostics.append(RuntimeDiagnostic(
            severity="error",
            code="tool.invalid_parameters",
            message=f"Tool '{tool.name}' parameters must be a dict",
        ))
    elif "type" in tool.parameters and tool.parameters["type"] != "object":
        diagnostics.append(RuntimeDiagnostic(
            severity="warning",
            code="tool.parameters_not_object",
            message=f"Tool '{tool.name}' parameters type should be 'object'",
        ))

    # 检查 execute 可调用
    if not callable(tool.execute):
        diagnostics.append(RuntimeDiagnostic(
            severity="error",
            code="tool.execute_not_callable",
            message=f"Tool '{tool.name}' execute is not callable",
        ))

    return diagnostics


def assemble_tools(
    workspace: Path,
    options: RuntimeAssemblyIntent,
    config: RuntimeConfig,
) -> AssembledTools:
    """组装所有来源的工具并应用权限策略。

    流程：
    1. 加载扩展和技能（获取工具、钩子、命令等）。
    2. 创建 MCP 代理工具。
    3. 创建内置工具。
    4. 按优先级合并所有工具（后者覆盖同名前者）。
    5. 校验工具定义。
    6. 记录工具来源和 metadata。
    7. 检测名称冲突并生成诊断。
    8. 如果是只读模式，过滤掉非只读工具。
    9. 创建 ToolRuntime 并应用权限策略。

    Args:
        workspace: 工作区目录路径。
        options: 创建会话的配置选项。
        config: 已解析的运行时配置。

    Returns:
        AssembledTools 对象，包含最终工具列表和加载的扩展/技能信息。
    """
    diagnostics: list[RuntimeDiagnostic] = []

    # 加载扩展和技能
    loaded_extensions = load_extensions(workspace, configured_paths=config.extension_paths)
    loaded_skills = load_skills(workspace, configured_paths=config.skill_paths)

    # 收集扩展和技能的错误
    for error in loaded_extensions.errors:
        diagnostics.append(RuntimeDiagnostic(
            severity="warning",
            code="extension.load_error",
            message=error,
            source="extension",
        ))
    for error in loaded_skills.errors:
        diagnostics.append(RuntimeDiagnostic(
            severity="warning",
            code="skill.load_error",
            message=error,
            source="skill",
        ))

    # 创建 MCP 代理工具
    mcp_tools = create_mcp_proxy_tools(
        parse_mcp_tool_configs(config.mcp_servers),
        client=options.mcp_client,
    )

    # 检查 MCP client 缺失
    if config.mcp_servers and not options.mcp_client:
        diagnostics.append(RuntimeDiagnostic(
            severity="warning",
            code="mcp.client_missing",
            message="MCP servers configured but no MCP client provided",
        ))

    # 创建内置工具
    builtin_tools = create_builtin_tools(
        workspace,
        enabled_names=config.enabled_builtin_tools,
        edit_require_unique_match=config.edit_require_unique_match,
        shell_policy=ShellExecutionPolicy(
            timeout_seconds=config.shell_timeout_seconds,
            max_timeout_seconds=config.shell_max_timeout_seconds,
            stdout_limit=config.shell_stdout_limit,
            stderr_limit=config.shell_stderr_limit,
            allowed_env=tuple(config.shell_allowed_env or ()),
        ),
    )

    # 按名称去重合并：内置 -> 自定义 -> 技能 -> 扩展 -> MCP（后者覆盖前者）
    tool_map: dict[str, tuple[AgentTool, str, str | None]] = {}

    # 内置工具
    for tool in builtin_tools:
        if tool.name in tool_map:
            diagnostics.append(RuntimeDiagnostic(
                severity="warning",
                code="tool.name_conflict",
                message=f"Tool '{tool.name}' from builtin overrides previous registration",
            ))
        tool_map[tool.name] = (tool, "builtin", None)

    # 调用方工具
    for tool in options.tools:
        if get_builtin_tool_metadata(tool.name) is not None:
            diagnostics.append(RuntimeDiagnostic(
                severity="warning",
                code="tool.reserved_name",
                message=f"Tool '{tool.name}' from caller uses a reserved builtin name",
                source="caller",
            ))
            continue
        if tool.name in tool_map:
            prev_source = tool_map[tool.name][1]
            diagnostics.append(RuntimeDiagnostic(
                severity="warning",
                code="tool.name_conflict",
                message=f"Tool '{tool.name}' from caller overrides {prev_source}",
            ))
        tool_map[tool.name] = (tool, "caller", None)

    # Skill 按需加载工具
    for tool in loaded_skills.tools:
        if get_builtin_tool_metadata(tool.name) is not None:
            diagnostics.append(RuntimeDiagnostic(
                severity="warning",
                code="tool.reserved_name",
                message=f"Tool '{tool.name}' from skill uses a reserved builtin name",
                source="skill",
            ))
            continue
        if tool.name in tool_map:
            prev_source = tool_map[tool.name][1]
            diagnostics.append(RuntimeDiagnostic(
                severity="warning",
                code="tool.name_conflict",
                message=f"Tool '{tool.name}' from skill overrides {prev_source}",
            ))
        tool_map[tool.name] = (tool, "extension", "skill")

    # 扩展工具
    for tool in loaded_extensions.tools:
        if get_builtin_tool_metadata(tool.name) is not None:
            diagnostics.append(RuntimeDiagnostic(
                severity="warning",
                code="tool.reserved_name",
                message=f"Tool '{tool.name}' from extension uses a reserved builtin name",
                source="extension",
            ))
            continue
        if tool.name in tool_map:
            prev_source = tool_map[tool.name][1]
            diagnostics.append(RuntimeDiagnostic(
                severity="warning",
                code="tool.name_conflict",
                message=f"Tool '{tool.name}' from extension overrides {prev_source}",
            ))
        tool_map[tool.name] = (tool, "extension", None)

    # MCP 工具
    for tool in mcp_tools:
        if get_builtin_tool_metadata(tool.name) is not None:
            diagnostics.append(RuntimeDiagnostic(
                severity="warning",
                code="tool.reserved_name",
                message=f"Tool '{tool.name}' from MCP uses a reserved builtin name",
                source="mcp",
            ))
            continue
        if tool.name in tool_map:
            prev_source = tool_map[tool.name][1]
            diagnostics.append(RuntimeDiagnostic(
                severity="warning",
                code="tool.name_conflict",
                message=f"Tool '{tool.name}' from MCP overrides {prev_source}",
            ))
        tool_map[tool.name] = (tool, "mcp", None)

    # 校验工具定义并构建 RegisteredTool 列表
    registered_tools: list[RegisteredTool] = []
    for name, (tool, source, origin) in tool_map.items():
        tool_diagnostics = validate_tool_definition(tool)
        diagnostics.extend(tool_diagnostics)
        if any(diag.severity == "error" for diag in tool_diagnostics):
            continue

        metadata = (
            get_builtin_tool_metadata(tool.name)
            if source == "builtin"
            else tool.metadata or infer_tool_metadata(tool)
        )

        registered_tools.append(RegisteredTool(
            name=name,
            tool=tool,
            metadata=metadata,
            source=source,
            origin=origin,
        ))

    # 注册到 ToolRegistry
    registry = ToolRegistry()
    for reg_tool in registered_tools:
        registry.register(reg_tool.tool, metadata=reg_tool.metadata)

    # 只读模式：过滤掉非只读工具
    if config.read_only_mode:
        registry = _read_only_registry(registry)

    effective_registered_tools = [
        reg_tool
        for reg_tool in registered_tools
        if registry.get(reg_tool.name) is not None
    ]

    # 创建 ToolRuntime 并应用权限策略
    runtime = ToolRuntime(
        registry=registry,
        permission_policy=PermissionPolicy(
            mode=config.tool_permission_mode,  # type: ignore[arg-type]
            block_dangerous_bash=config.block_dangerous_bash,
            bash_allow_patterns=config.bash_allow_patterns,
            bash_block_patterns=config.bash_block_patterns,
        ),
        approval_provider=options.approval_provider or DeferredApprovalProvider(),
    )

    return AssembledTools(
        tools=[tool.to_spec() for tool in registry.list()],
        registered_tools=effective_registered_tools,
        tool_runtime=runtime,
        loaded_extensions=loaded_extensions,
        loaded_skills=loaded_skills,
        diagnostics=diagnostics,
    )


def _read_only_registry(registry: ToolRegistry) -> ToolRegistry:
    """过滤工具注册表，只保留标记为 read_only 的工具。"""
    filtered = ToolRegistry()
    for tool in registry.list():
        metadata = registry.metadata_for(tool.name)
        if metadata is not None and metadata.read_only:
            filtered.register(tool, metadata=metadata)
    return filtered

# ---- runtime context ----

# 新手导读：runtime context 汇总工作区、扩展加载结果和组装时诊断，供后续 session 创建使用。
# 关注点：它是装配过程里的共享信息包，不是 sessions/context 的上下文投影。

"""
Runtime 系统提示词启动上下文。

本模块只负责构建创建 SessionController 时可注入系统提示词的静态启动信息：
- 仓库 bootstrap 概览；
- 配置/扩展/skills 提供的 prompt guidelines；
- 配置/扩展/skills 提供的追加系统提示词段落；
- 工具说明片段。

注意：顶层目录概览虽然会出现在初始系统提示词里，但它不是唯一事实来源。
每次模型调用前，sessions.context.ContextGovernor 会刷新 RepositorySnapshot，
并把动态仓库状态、证据和记忆召回投影到本轮 ContextView 中。
"""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from codepilot.extensions.types import LoadedExtensions
from codepilot.sessions.storage import (
    build_repository_bootstrap,
    render_repository_context,
)



@dataclass(frozen=True)
class RuntimeContext:
    """供系统提示词渲染使用的启动上下文。

    RuntimeContext is built once during session assembly and then handed to the
    prompt renderer.  It copies and freezes collection fields so later mutation
    of config or extension objects cannot silently change the prompt contract.
    """

    repository_context: str
    prompt_guidelines: tuple[str, ...]
    append_sections: tuple[str, ...]
    tool_snippets: Mapping[str, str]
    memory_text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "repository_context", _strip_text(self.repository_context))
        object.__setattr__(
            self,
            "prompt_guidelines",
            tuple(_clean_text_items(self.prompt_guidelines)),
        )
        object.__setattr__(
            self,
            "append_sections",
            tuple(_clean_text_items(self.append_sections)),
        )
        object.__setattr__(
            self,
            "tool_snippets",
            MappingProxyType(_clean_tool_snippets(self.tool_snippets)),
        )
        object.__setattr__(self, "memory_text", _strip_text(self.memory_text))


def build_runtime_context(
    workspace: Path,
    config: ResolvedRuntimeConfig,
    loaded_extensions: LoadedExtensions,
    loaded_skills: LoadedExtensions,
) -> RuntimeContext:
    """构建系统提示词启动上下文。

    这里不做 active files、recent evidence、memory retrieval 或当前任务投影；
    这些动态内容由 sessions.context.ContextGovernor 在每次模型调用前处理。
    """

    prompt_guidelines = [
        *(config.prompt_guidelines or []),
        *loaded_extensions.prompt_guidelines,
        *loaded_skills.prompt_guidelines,
    ]
    prompt_guidelines.extend([f"[skill-diagnostic] {d}" for d in loaded_skills.diagnostics])

    append_sections = []
    if config.append_system_prompt:
        append_sections.append(config.append_system_prompt)
    append_sections.extend(loaded_extensions.append_prompts)
    append_sections.extend(loaded_skills.append_prompts)
    if config.prompt_debug_sources:
        append_sections.append("\n".join(_build_prompt_debug_lines(loaded_extensions, loaded_skills)))

    return RuntimeContext(
        repository_context=render_repository_context(build_repository_bootstrap(workspace)),
        prompt_guidelines=prompt_guidelines,
        append_sections=append_sections,
        tool_snippets=config.tool_snippets or {},
        memory_text="",
    )


def _build_prompt_debug_lines(
    loaded_extensions: LoadedExtensions,
    loaded_skills: LoadedExtensions,
) -> list[str]:
    """构建提示词来源调试信息（用于 prompt_debug_sources 模式）。"""

    debug_lines: list[str] = ["## Prompt Sources", "### extensions"]
    debug_lines.extend([f"- {p}" for p in loaded_extensions.loaded_paths] or ["- (none)"])
    debug_lines.append("### skills")
    debug_lines.extend([f"- {p}" for p in loaded_skills.loaded_paths] or ["- (none)"])
    if loaded_extensions.errors or loaded_skills.errors:
        debug_lines.append("### errors")
        debug_lines.extend([f"- {e}" for e in [*loaded_extensions.errors, *loaded_skills.errors]])
    if loaded_skills.diagnostics:
        debug_lines.append("### diagnostics")
        debug_lines.extend([f"- {d}" for d in loaded_skills.diagnostics])
    return debug_lines


def _strip_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _clean_text_items(values: object) -> list[str]:
    if values is None:
        return []
    return [
        text
        for text in (_strip_text(value) for value in values)
        if text
    ]


def _clean_tool_snippets(values: Mapping[str, str] | None) -> dict[str, str]:
    if not values:
        return {}
    snippets: dict[str, str] = {}
    for key, value in values.items():
        name = _strip_text(key)
        snippet = _strip_text(value)
        if name and snippet:
            snippets[name] = snippet
    return snippets



# ---- system prompt assembly ----

# 新手导读：prompt 组装把系统提示、工具/扩展指南和项目规则合成运行时 system prompt。
# 关注点：这里适合看 Agent 的初始行为约束从哪里来。

"""
系统提示词渲染模块。

负责构建最终发送给 LLM 的系统提示词。

阶段B改进：
- 使用 PromptPlan 进行结构化段落组装
- 每个段落有明确的名称、来源和优先级
- 保持最终传入 Agent 的仍是字符串
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from typing import Any

@dataclass(frozen=True)
class PromptSection:
    """系统提示词中的一个有序段落。"""

    name: str
    content: str
    source: str
    priority: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _require_prompt_text(self.name, field_name="section name"),
        )
        object.__setattr__(
            self,
            "content",
            _require_prompt_text(self.content, field_name="section content"),
        )
        object.__setattr__(
            self,
            "source",
            _require_prompt_text(self.source, field_name="section source"),
        )
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise TypeError("Prompt section priority must be int")


@dataclass
class PromptPlan:
    """结构化系统提示词计划。

    PromptPlan owns section assembly rules: names are unique, empty optional
    content is skipped at the boundary, and rendering always follows priority.
    """

    sections: list[PromptSection] = field(default_factory=list)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for section in self.sections:
            if not isinstance(section, PromptSection):
                raise TypeError("PromptPlan sections must be PromptSection instances")
            if section.name in seen:
                raise ValueError(f"Duplicate prompt section: {section.name}")
            seen.add(section.name)

    def add_section(
        self,
        *,
        name: str,
        content: str,
        source: str,
        priority: int,
    ) -> bool:
        """Add one section, returning False when optional content is empty."""

        if not _normalize_prompt_text(content):
            return False
        section = PromptSection(
            name=name,
            content=content,
            source=source,
            priority=priority,
        )
        if any(existing.name == section.name for existing in self.sections):
            raise ValueError(f"Duplicate prompt section: {section.name}")
        self.sections.append(section)
        return True

    def render(self) -> str:
        sections = self._ordered_sections()
        return "\n\n".join(
            section.content
            for section in sections
        )

    def get_sources(self) -> dict[str, str]:
        return {
            section.name: section.source
            for section in self._ordered_sections()
        }

    def _ordered_sections(self) -> list[PromptSection]:
        return sorted(self.sections, key=lambda section: section.priority)


# ── 段落优先级常量 ────────────────────────────────────────────────

PRIORITY_IDENTITY = 10       # 身份定义
PRIORITY_RULES = 20          # 核心规则
PRIORITY_SAFETY = 30         # 安全策略
PRIORITY_WORKSPACE = 40      # 工作区指令
PRIORITY_REPOSITORY = 50     # 仓库上下文
PRIORITY_MEMORY = 60         # 长期记忆
PRIORITY_CAPABILITY = 70     # 能力说明
PRIORITY_EXTENSIONS = 80     # 扩展内容
PRIORITY_RUNTIME = 90        # 运行时事实


def build_runtime_system_prompt(
    *,
    base_system_prompt: str,
    tools: list[Any],
    runtime_context: RuntimeContext,
    workspace: Path,
) -> str:
    """构建运行时系统提示词（工厂调用入口）。

    使用 PromptPlan 进行结构化段落组装。

    Args:
        base_system_prompt: 基础系统提示词（来自配置或会话恢复）。
        tools: 当前会话可用的工具列表。
        runtime_context: 运行时上下文（仓库信息、准则、记忆等）。
        workspace: 工作区目录路径。

    Returns:
        完整的系统提示词字符串。
    """
    plan = PromptPlan()

    # 1. 身份和核心规则
    if base_system_prompt:
        plan.add_section(
            name="identity",
            content=base_system_prompt,
            source="config",
            priority=PRIORITY_IDENTITY,
        )
    else:
        plan.add_section(
            name="identity",
            content=_build_default_identity(tools, runtime_context),
            source="default",
            priority=PRIORITY_IDENTITY,
        )

    # 2. 安全策略
    plan.add_section(
        name="safety",
        content=_build_safety_section(),
        source="default",
        priority=PRIORITY_SAFETY,
    )

    # 3. 仓库上下文
    if runtime_context.repository_context:
        plan.add_section(
            name="repository",
            content=runtime_context.repository_context,
            source="workspace",
            priority=PRIORITY_REPOSITORY,
        )

    # 4. 长期记忆
    if runtime_context.memory_text:
        plan.add_section(
            name="memory",
            content=f"长期记忆（MEMORY）：\n{runtime_context.memory_text}",
            source="memory",
            priority=PRIORITY_MEMORY,
        )

    # 5. 扩展内容
    for i, section in enumerate(runtime_context.append_sections or []):
        plan.add_section(
            name=f"extension_{i}",
            content=section,
            source="extension",
            priority=PRIORITY_EXTENSIONS,
        )

    # 6. 运行时事实
    date = datetime.now().strftime("%Y-%m-%d")
    cwd_text = str(workspace.resolve()).replace("\\", "/")
    plan.add_section(
        name="runtime_facts",
        content=f"当前日期：{date}\n当前工作目录：{cwd_text}",
        source="runtime",
        priority=PRIORITY_RUNTIME,
    )

    return plan.render()


def _build_default_identity(
    tools: list[Any],
    runtime_context: RuntimeContext,
    *,
    tool_names: list[str] | None = None,
) -> str:
    """构建默认的身份和规则段落。"""
    tool_names = tool_names if tool_names is not None else _canonical_tool_names(tools)
    snippets = {**_default_tool_snippets(), **(runtime_context.tool_snippets or {})}
    visible_tools = [name for name in tool_names if name in snippets]
    tools_list = "\n".join([f"- {name}: {snippets[name]}" for name in visible_tools]) if visible_tools else "- （由运行时提供）"
    tools_text = "、".join(tool_names) if tool_names else "（由运行时提供）"

    # 合并准则
    guidelines = [
        "先理解目标与约束，再开始操作；需求不清时只提最小必要问题。",
        "对代码与文件系统的判断，优先基于工具结果，不凭空猜测。",
        "变更应小步、可验证、可回滚，优先修复根因而不是症状。",
        "涉及风险操作时先提示影响范围，再执行更安全替代方案。",
        "输出要简洁直接：先结论，再关键证据，再下一步。",
    ]
    if runtime_context.prompt_guidelines:
        guidelines.extend([g.strip() for g in runtime_context.prompt_guidelines if g.strip()])
    guidelines_text = "\n".join([f"{i + 1}. {g}" for i, g in enumerate(guidelines)])

    return f"""你是一个专业、可靠的编程助手。

工作原则（必须遵守）：
{guidelines_text}

可用工具（当前会话）：
- 工具名：{tools_text}
- 工具说明：
{tools_list}

工具使用规范：
1. 查目录优先 ls/find，查内容优先 read/grep；不要用 bash 代替常规读写工具。
2. 修改前先读文件并定位上下文，确认修改点后再 edit/write。
3. edit 只做精确替换；需要大段重构或新文件时再用 write。
4. 执行 bash 前先检查副作用，禁止与目标无关的破坏性命令。
5. 若可先做只读验证，就先只读验证，再执行写操作。
6. bash 命令默认已经在当前工作目录运行；不要 cd /workspace，不要把 /tmp 当作工作区。
7. 本地环境可能是 Windows；验证优先直接运行 python -m pytest ...，避免 cat/ls/pwd/which/python3 等 Linux 专属写法。

短任务探索规则：
1. 任务缺少具体文件、符号或错误信息时，先获取最小事实，不立即修改。
2. "修一下测试"：先识别测试入口并运行相关测试，取得真实失败。
3. "加个接口"：先搜索现有路由或相邻接口，读取项目已有写法。
4. "为什么挂了"：优先使用用户提供的错误、最近失败结果和会话历史；若"这里"没有明确指代，再询问最小必要信息。
5. 已有明确文件、符号或堆栈时，直接进行针对性搜索和读取。
6. 每次探索应逐步收窄范围，避免一次读取大量无关文件。

代码质量要求：
1. 保持现有风格与命名习惯；
2. 优先修复根因，不只绕过症状；
3. 对关键行为变更，补充最小测试或验证步骤；
4. 若执行失败，明确错误原因、影响范围与修复建议；
5. 变更完成后给出"做了什么 / 为什么这样做 / 如何验证"。"""


def _build_safety_section() -> str:
    """构建安全策略段落。"""
    return """安全边界：
1. 不输出或泄露敏感密钥；
2. 不执行明显危险、不可逆且与目标无关的命令；
3. 涉及潜在破坏操作时，先说明影响范围并给出替代方案。"""


def build_default_system_prompt(tool_names: list[str] | None = None) -> str:
    """构建默认系统提示词（便捷入口）。

    Args:
        tool_names: 可选的工具名称列表。

    Returns:
        默认系统提示词字符串。
    """
    plan = PromptPlan()
    plan.add_section(
        name="identity",
        content=_build_default_identity(
            [],
            RuntimeContext(
                repository_context="",
                prompt_guidelines=[],
                append_sections=[],
                tool_snippets={},
                memory_text="",
            ),
            tool_names=tool_names or [],
        ),
        source="default",
        priority=PRIORITY_IDENTITY,
    )
    plan.add_section(
        name="safety",
        content=_build_safety_section(),
        source="default",
        priority=PRIORITY_SAFETY,
    )
    date = datetime.now().strftime("%Y-%m-%d")
    cwd_text = str(Path.cwd().resolve()).replace("\\", "/")
    plan.add_section(
        name="runtime_facts",
        content=f"当前日期：{date}\n当前工作目录：{cwd_text}",
        source="runtime",
        priority=PRIORITY_RUNTIME,
    )
    return plan.render()


def _default_tool_snippets() -> dict[str, str]:
    """默认的工具说明片段（用于系统提示词中的工具说明部分）。"""
    return {
        "ls": "列出目录内容（文件名、目录、大小）。",
        "find": "按 glob 查找文件路径。",
        "read": "读取文本文件内容。",
        "grep": "按正则在文件中搜索内容。",
        "edit": "对文件做精确文本替换。",
        "write": "写入新文件或重写文件。",
        "bash": "执行命令行命令（需注意风险）。",
    }


def _canonical_tool_names(tools: list[Any]) -> list[str]:
    """提取工具名称列表（去重并保持顺序）。"""
    names: list[str] = []
    seen: set[str] = set()
    for tool in tools:
        if tool.name not in seen:
            seen.add(tool.name)
            names.append(tool.name)
    return names


def _normalize_prompt_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _require_prompt_text(value: object, *, field_name: str) -> str:
    text = _normalize_prompt_text(value)
    if not text:
        raise ValueError(f"Prompt {field_name} cannot be empty")
    return text

# ---- hook pipeline ----

# 新手导读：hook_pipeline 把调用者 hook、扩展 hook 和 session hook 按顺序组合。
# 关注点：如果 before/after 行为看起来重复，先看这里的组合顺序。

"""
钩子组合（Hook Composition）模块。

负责将调用方提供的钩子与扩展加载的钩子组合成可执行的管道。

三种钩子类型：
1) before_tool_call: 工具调用前的策略拦截钩子
   - 可用于项目规则、扩展策略、临时禁用等外部拦截
   - 不替代 ToolRuntime 的权限/审批，也不替代具体工具的参数与路径校验
   - 按顺序执行，任一返回 block=True 即拦截
2) after_tool_call: 工具调用后的处理钩子（可用于结果后处理、日志记录）
   - 按顺序执行，每个钩子可修改结果内容
3) lifecycle_hooks: 生命周期钩子（before_prompt / after_prompt）
   - 简单合并为列表，按顺序执行
"""

import inspect
from typing import Any, Awaitable, Callable

from codepilot.protocols.commands import (
    AfterToolCallContext,
    AfterToolCallResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
)


# 钩子函数类型别名
BeforeToolHook = Callable[
    [BeforeToolCallContext, Any | None],
    BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None],
]
AfterToolHook = Callable[
    [AfterToolCallContext, Any | None],
    AfterToolCallResult | None | Awaitable[AfterToolCallResult | None],
]
LifecycleHookFn = Callable[[Any], None | Awaitable[None]]


def compose_before_tool_call(
    base: BeforeToolHook | None,
    hooks: list[BeforeToolHook],
):
    """组合 before_tool_call 钩子管道。

    将调用方的 base 钩子和扩展加载的 hooks 合并为一个执行器。
    执行时按顺序调用，任一钩子返回 block=True 即立即拦截（短路）。

    Args:
        base: 调用方提供的基础钩子（可选）。
        hooks: 扩展加载的钩子列表。

    Returns:
        组合后的异步执行器函数；无钩子时返回 None。
    """
    chain: list[BeforeToolHook] = []
    if base:
        chain.append(base)
    chain.extend(hooks)
    if not chain:
        return None

    async def _runner(ctx: BeforeToolCallContext, signal: Any | None):
        for hook in chain:
            result = hook(ctx, signal)
            if inspect.isawaitable(result):
                result = await result  # type: ignore[assignment]
            # 任一钩子返回 block=True 即短路拦截
            if result and result.block:
                return result
        return None

    return _runner


def compose_after_tool_call(
    base: AfterToolHook | None,
    hooks: list[AfterToolHook],
):
    """组合 after_tool_call 钩子管道。

    将调用方的 base 钩子和扩展加载的 hooks 合并为一个执行器。
    执行时按顺序调用，每个钩子可修改上下文中的结果内容、详情和错误标志。

    Args:
        base: 调用方提供的基础钩子（可选）。
        hooks: 扩展加载的钩子列表。

    Returns:
        组合后的异步执行器函数；无钩子时返回 None。
    """
    chain: list[AfterToolHook] = []
    if base:
        chain.append(base)
    chain.extend(hooks)
    if not chain:
        return None

    async def _runner(ctx: AfterToolCallContext, signal: Any | None):
        final = AfterToolCallResult()
        for hook in chain:
            result = hook(ctx, signal)
            if inspect.isawaitable(result):
                result = await result  # type: ignore[assignment]
            if not result:
                continue
            # 钩子可以覆盖结果的 content、details 和 is_error
            if result.content is not None:
                ctx.result.content = result.content
                final.content = result.content
            if result.details is not None:
                ctx.result.details = result.details
                final.details = result.details
            if result.is_error is not None:
                ctx.is_error = result.is_error
                final.is_error = result.is_error
        # 如果没有任何钩子修改了结果，返回 None
        if final.content is None and final.details is None and final.is_error is None:
            return None
        return final

    return _runner


def compose_lifecycle_hooks(
    base_hooks: list[LifecycleHookFn] | None,
    loaded_hooks: list[LifecycleHookFn],
) -> list[LifecycleHookFn]:
    """组合生命周期钩子列表（before_prompt / after_prompt）。

    简单地将调用方钩子和扩展钩子合并为一个列表，按顺序执行。

    Args:
        base_hooks: 调用方提供的钩子列表（可选）。
        loaded_hooks: 扩展加载的钩子列表。

    Returns:
        合并后的钩子函数列表。
    """
    chain: list[LifecycleHookFn] = []
    chain.extend(base_hooks or [])
    chain.extend(loaded_hooks)
    return chain

# ---- assembly flow ----

# 新手导读：runtime 组装主入口：把配置、模型、工具、扩展、prompt 和 session 选项串成 RuntimeAssembly。
# 关注点：想理解项目启动流程，先从 assemble_runtime() 的步骤读起。

"""
Runtime 装配模块。

负责把上层友好的 RuntimeAssemblyIntent 装配成一次会话创建所需的
RuntimeAssembly 和 SessionController。

装配流程：
1) 加载运行时输入（工作区资源、会话恢复元数据）
2) 解析模型配置（从 options / 会话元数据 / 工作区配置中确定模型）
3) 解析运行时配置（多源合并）
4) 组装工具列表（内置工具 + 扩展工具 + MCP 工具）
5) 构建运行时上下文（仓库信息、提示词准则等）
6) 构建系统提示词
7) 组装钩子（before/after tool call、生命周期钩子）
8) 构造最终的 SessionOptions 并创建 SessionController
9) 保存装配产物（RuntimeAssembly）
"""

from dataclasses import replace
import os

from codepilot.core.model_step import convert_to_llm
from codepilot.llm.catalog import get_env_api_key_name
from codepilot.llm.registry import register_builtin_api_providers
from codepilot.llm.adapter import ProviderModelPort
from codepilot.sessions.controller import SessionController
from codepilot.sessions.controller import create_session_controller as _create_session_controller
from codepilot.sessions.contracts import SessionOptions
from codepilot.sessions.storage import build_repository_bootstrap
from codepilot.tools.adapter import ToolRuntimePort



class UnknownRuntimeConfigKeyError(KeyError):
    """请求解释的配置项不属于 Runtime 配置。"""


def assemble_runtime(options: RuntimeAssemblyIntent) -> tuple[SessionController, RuntimeAssembly]:
    """完整装配流程，返回 SessionController 和 RuntimeAssembly。

    装配流程：
    1. load_runtime_inputs: 加载工作区资源和会话恢复元数据。
    2. resolve_model: 解析模型配置。
    3. resolve_runtime_config: 多源合并运行时配置。
    4. assemble_tools: 组装工具列表（内置 + 扩展 + MCP）。
    5. build_runtime_context: 构建运行时上下文。
    6. build_runtime_system_prompt: 构建系统提示词。
    7. compose_*: 组装钩子函数。
    8. 构建 RuntimeAssembly。

    Args:
        options: 友好的创建会话选项。

    Returns:
        (SessionController, RuntimeAssembly) 元组。
    """
    diagnostics: list[RuntimeDiagnostic] = []
    register_builtin_api_providers()

    # 步骤 1：加载运行时输入数据
    inputs = load_runtime_inputs(options)

    # 步骤 2：解析模型
    resolved_model = resolve_model(options, inputs)

    # 确定凭证来源
    credential_source, credential_location = _resolve_credential_source(
        options,
        inputs,
        resolved_model,
    )

    # 步骤 3：解析运行时配置
    config = resolve_runtime_config(options, inputs)

    # 构建配置来源
    sources = _build_config_sources(config.sources)
    sources.update({
        "model": resolved_model.source,
        "provider": resolved_model.source,
        "model_id": resolved_model.source,
    })

    # 构建 ResolvedRuntimeProfile
    profile = ResolvedRuntimeProfile(
        model=resolved_model.model,
        credential_source=credential_source,
        credential_location=credential_location,
        permission_mode=config.tool_permission_mode,  # type: ignore[arg-type]
        task_mode=config.task_mode,
        planning_budget_profile=config.planning_budget_profile,
        sources=sources,
    )

    # 步骤 4：组装工具
    assembled_tools = assemble_tools(inputs.workspace, options, config)

    # 收集工具装配诊断
    for diag in assembled_tools.diagnostics:
        diagnostics.append(diag)

    # 步骤 5：构建运行时上下文
    runtime_context = build_runtime_context(
        inputs.workspace,
        config,
        assembled_tools.loaded_extensions,
        assembled_tools.loaded_skills,
    )

    # 步骤 6：构建系统提示词
    system_prompt = build_runtime_system_prompt(
        base_system_prompt=config.system_prompt,
        tools=assembled_tools.tools,
        runtime_context=runtime_context,
        workspace=inputs.workspace,
    )

    # 步骤 7：组装钩子函数（合并调用方钩子和扩展钩子）
    before_tool_call = compose_before_tool_call(
        options.before_tool_call,
        assembled_tools.loaded_extensions.before_tool_hooks,
    )
    after_tool_call = compose_after_tool_call(
        options.after_tool_call,
        assembled_tools.loaded_extensions.after_tool_hooks,
    )
    before_prompt_hooks = compose_lifecycle_hooks(
        options.before_prompt_hooks,
        assembled_tools.loaded_extensions.before_prompt_hooks,
    )
    after_prompt_hooks = compose_lifecycle_hooks(
        options.after_prompt_hooks,
        assembled_tools.loaded_extensions.after_prompt_hooks,
    )

    # 步骤 8：构造最终的 SessionOptions
    session_options = SessionOptions(
        model=resolved_model.model,
        workspace_dir=inputs.workspace,
        system_prompt=system_prompt,
        session_id=options.session_id,
        messages=options.messages,
        thinking_level=config.thinking_level,
        tool_execution=config.tool_execution,
        max_tool_calls_per_turn=config.max_tool_calls_per_turn,
        memory_enabled=options.memory_enabled,
        task_control_enabled=options.task_control_enabled,
        task_mode=config.task_mode,
        planning_budget_profile=config.planning_budget_profile,
        max_task_replans_per_run=options.max_task_replans_per_run or 2,
        convert_to_llm=convert_to_llm,
        get_api_key=resolved_model.get_api_key,
        retry_enabled=config.retry_enabled,
        max_retries=config.max_retries,
        retry_base_delay_ms=config.retry_base_delay_ms,
        extension_commands={
            **options.extension_commands,
            **assembled_tools.loaded_skills.commands,
            **assembled_tools.loaded_extensions.commands,
        },
        before_prompt_hooks=before_prompt_hooks,
        after_prompt_hooks=after_prompt_hooks,
        before_tool_call=before_tool_call,
        after_tool_call=after_tool_call,
        stream_fn=options.stream_fn,
        prepare_context=None,
    )

    # 步骤 9：构建能力目录和装配产物
    capability_catalog = CapabilityCatalog(
        tools=assembled_tools.registered_tools,
        commands={
            **assembled_tools.loaded_skills.commands,
            **assembled_tools.loaded_extensions.commands,
        },
    )
    controller = _create_session_controller(session_options)
    effective_session_options = replace(
        session_options,
        session_id=controller.session_id,
    )
    assembly = RuntimeAssembly(
        session_options=effective_session_options,
        profile=profile,
        repository=build_repository_bootstrap(inputs.workspace),
        capabilities=capability_catalog,
        tool_runtime=assembled_tools.tool_runtime,
        model_port=ProviderModelPort(
            model=effective_session_options.model,
            stream_fn=effective_session_options.stream_fn,
            convert_messages=effective_session_options.convert_to_llm,
            get_api_key=effective_session_options.get_api_key,
        ),
        tool_port=ToolRuntimePort(
            assembled_tools.tool_runtime,
            before_tool_call=effective_session_options.before_tool_call,
            after_tool_call=effective_session_options.after_tool_call,
        ),
        diagnostics=diagnostics,
    )

    return controller, assembly


def _resolve_credential_source(
    options: RuntimeAssemblyIntent,
    inputs: RuntimeInputs,
    resolved_model: ResolvedModel,
) -> tuple[str, str | None]:
    """解析凭证来源。

    Returns:
        (凭证来源, 凭证位置) 元组。
    """
    resources = inputs.resources

    if options.get_api_key:
        return "caller", "get_api_key function"

    if (
        resources
        and resources.model
        and resolved_model.source.location == ".codepilot/model.local.json"
    ):
        if resources.model.api_key_env:
            if os.getenv(resources.model.api_key_env):
                return "env", resources.model.api_key_env
        if resources.model.api_key:
            return "local-file", ".codepilot/model.local.json"

    env_name = get_env_api_key_name(resolved_model.model.provider)
    if env_name and os.getenv(env_name):
        return "env", env_name
    if resolved_model.model.api == "openai-compatible":
        fallback_env_name = get_env_api_key_name("openai")
        if fallback_env_name and os.getenv(fallback_env_name):
            return "env", fallback_env_name

    return "missing", None


def explain_runtime_config(
    options: RuntimeAssemblyIntent,
    key: str,
) -> ResolvedConfigValue:
    """解析单个配置项的最终值和来源，供 CLI 等接口展示。"""

    inputs = load_runtime_inputs(options)
    if key in {"model", "provider", "model_id"}:
        resolved_model = resolve_model(options, inputs)
        model = resolved_model.model
        values = {
            "model": f"{model.provider}/{model.id}" if model.provider else model.id,
            "provider": model.provider,
            "model_id": model.id,
        }
        return ResolvedConfigValue(
            key=key,
            value=values[key],
            source=resolved_model.source,
        )

    config = resolve_runtime_config(options, inputs)
    if key not in config.sources:
        raise UnknownRuntimeConfigKeyError(key)
    return ResolvedConfigValue(
        key=key,
        value=getattr(config, key),
        source=_build_config_sources(config.sources)[key],
    )


def _build_config_sources(raw_sources: dict[str, str]) -> dict[str, ConfigValueSource]:
    """构建配置来源映射。"""
    source_map = {
        "options": ConfigValueSource(kind="cli"),
        "restored_session": ConfigValueSource(kind="session"),
        "workspace": ConfigValueSource(kind="project", location=".codepilot/settings.json"),
        "default": ConfigValueSource(kind="default"),
    }
    sources: dict[str, ConfigValueSource] = {}
    for key, value in raw_sources.items():
        try:
            sources[key] = source_map[value]
        except KeyError as exc:
            raise ValueError(f"Unknown runtime config source: {value}") from exc
    return sources

__all__ = [
    "AppSessionView",
    "AssembledTools",
    "CapabilityCatalog",
    "ConfigSourceKind",
    "ConfigValueSource",
    "RegisteredTool",
    "RegisteredToolSource",
    "ResolvedConfigValue",
    "ResolvedModel",
    "ResolvedRuntimeConfig",
    "ResolvedRuntimeProfile",
    "RuntimeAssembly",
    "RuntimeAssemblyIntent",
    "RuntimeDefaults",
    "RuntimeDiagnostic",
    "RuntimeDiagnosticSeverity",
    "RuntimeInputs",
    "RuntimePermissionMode",
    "SessionOpenIntent",
    "SessionRef",
    "UnknownRuntimeConfigKeyError",
    "WorkspaceResources",
    "assemble_runtime",
    "build_runtime_context",
    "build_runtime_system_prompt",
    "explain_runtime_config",
    "load_runtime_inputs",
    "resolve_model",
    "resolve_runtime_config",
]
