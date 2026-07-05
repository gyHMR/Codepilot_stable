from __future__ import annotations

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

from codepilot.core.task_control import (
    PlanningBudgetProfile,
    TaskMode,
    ensure_planning_budget_profile,
    ensure_task_mode,
)
from codepilot.protocols import Model
from codepilot.protocols.commands import RegisteredCommand
from codepilot.sessions.repository import RepositoryBootstrap
from codepilot.sessions.types import SessionOptions
from codepilot.tools import AgentTool, ToolMetadata
from codepilot.tools.execution import ToolRuntime

from .assembly_input import RuntimePermissionMode


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


__all__ = [
    "CapabilityCatalog",
    "ConfigSourceKind",
    "ConfigValueSource",
    "RegisteredTool",
    "RegisteredToolSource",
    "ResolvedConfigValue",
    "ResolvedRuntimeProfile",
    "RuntimeAssembly",
    "RuntimeDiagnostic",
    "RuntimeDiagnosticSeverity",
]
