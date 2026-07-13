from __future__ import annotations

"""Runtime configuration loading and interface-facing config views."""

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING, cast

from codepilot.core.plan import (
    PlanningBudgetProfile,
    RunMode,
    ensure_planning_budget_profile,
    ensure_run_mode,
)
from codepilot.protocols import Model, ModelCapabilities
from codepilot.sessions.service import SessionStateService


@dataclass(frozen=True)
class SessionOpenMetadata:
    provider: str | None = None
    model_id: str | None = None
    system_prompt: str | None = None


def load_session_open_metadata(
    workspace_dir: str | Path,
    session_id: str | None,
) -> SessionOpenMetadata | None:
    if not session_id:
        return None
    state = SessionStateService(workspace_dir).get_session(session_id)
    if state is None:
        return None
    return SessionOpenMetadata(provider=state.model.provider, model_id=state.model.model)

if TYPE_CHECKING:
    from .actions import SessionOpenIntent


RuntimePermissionMode = Literal["read-only", "workspace-write", "ask"]
ConfigSourceKind = Literal["cli", "session", "project", "default"]
SUPPORTED_MODEL_APIS = {"openai-compatible", "anthropic-messages"}
_PERMISSION_MODES = {"read-only", "workspace-write", "ask"}
_REMOVED_TOOL_SECURITY_KEYS = {
    "block_dangerous_bash",
    "bash_allow_patterns",
    "bash_block_patterns",
    "tool_execution",
    "tool_snippets",
}


class UnknownRuntimeConfigKeyError(KeyError):
    """Requested runtime configuration key is not known."""


@dataclass(frozen=True)
class ConfigValueSource:
    kind: ConfigSourceKind
    location: str | None = None


@dataclass(frozen=True)
class ResolvedConfigValue:
    key: str
    value: Any
    source: ConfigValueSource


@dataclass(frozen=True)
class WorkspaceConfigCheck:
    rows: tuple[tuple[str, object], ...]
    border_style: str


@dataclass(frozen=True)
class WorkspaceConfigView:
    model_rows: tuple[tuple[str, object], ...]
    settings_rows: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class WorkspaceModelConfig:
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
        if self.api not in SUPPORTED_MODEL_APIS:
            raise ValueError(f"Unsupported API protocol: {self.api}")
        if not self.provider or not self.model_id or not self.base_url:
            raise ValueError("provider, model_id and base_url are required")
        if self.context_window <= 0 or self.max_tokens <= 0:
            raise ValueError("context_window and max_tokens must be positive")

    def to_model(self) -> Model:
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
        if self.api_key_env:
            value = os.getenv(self.api_key_env)
            if value:
                return value
        return self.api_key

    def build_api_key_resolver(self):
        return lambda _provider: self.resolve_api_key()


@dataclass
class WorkspaceSettings:
    provider: str | None = None
    model_id: str | None = None
    system_prompt: str | None = None
    thinking_level: str | None = None
    current_mode: RunMode | None = None
    planning_budget_profile: PlanningBudgetProfile | None = None
    max_tool_calls_per_turn: int | None = None
    retry_enabled: bool | None = None
    max_retries: int | None = None
    retry_base_delay_ms: int | None = None
    tool_permission_mode: RuntimePermissionMode | None = None
    edit_require_unique_match: bool | None = None
    prompt_guidelines: list[str] | None = None
    append_system_prompt: str | None = None
    extension_paths: list[str] | None = None
    skill_paths: list[str] | None = None
    prompt_debug_sources: bool | None = None
    mcp_servers: list[dict[str, Any]] | None = None
    shell_timeout_seconds: int | None = None
    shell_max_timeout_seconds: int | None = None
    shell_stdout_limit: int | None = None
    shell_stderr_limit: int | None = None
    shell_allowed_env: list[str] | None = None


@dataclass
class WorkspaceResources:
    settings: WorkspaceSettings = field(default_factory=WorkspaceSettings)
    model: WorkspaceModelConfig | None = None
    prompt: str | None = None
    enabled_tools: list[str] | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    workspace: Path
    settings: WorkspaceSettings
    local_model: WorkspaceModelConfig | None
    restored: SessionOpenMetadata | None
    system_prompt: str
    thinking_level: str
    current_mode: RunMode
    planning_budget_profile: PlanningBudgetProfile
    max_tool_calls_per_turn: int
    retry_enabled: bool
    max_retries: int
    retry_base_delay_ms: int
    tool_permission_mode: RuntimePermissionMode
    edit_require_unique_match: bool
    prompt_guidelines: list[str] | None
    append_system_prompt: str | None
    extension_paths: list[str] | None
    skill_paths: list[str] | None
    prompt_debug_sources: bool
    mcp_servers: list[dict[str, Any]] | None
    enabled_builtin_tools: list[str] | None
    shell_timeout_seconds: int
    shell_max_timeout_seconds: int
    shell_stdout_limit: int
    shell_stderr_limit: int
    shell_allowed_env: list[str] | None
    sources: dict[str, ConfigValueSource]


class WorkspaceResourceLoader:
    """Load the few workspace files runtime needs from `.codepilot/`."""

    def __init__(self, workspace_dir: str | Path) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.resource_root = self.workspace_dir / ".codepilot"
        self.settings_file = self.resource_root / "settings.json"
        self.model_file = self.resource_root / "model.local.json"
        self.prompt_file = self.resource_root / "prompt.md"
        self.tools_file = self.resource_root / "tools.json"

    def load(self) -> WorkspaceResources:
        return WorkspaceResources(
            settings=self._load_settings(),
            model=self._load_model(),
            prompt=self._load_prompt(),
            enabled_tools=self._load_tools(),
        )

    def _load_settings(self) -> WorkspaceSettings:
        raw = self._load_json_object(self.settings_file)
        if raw is None:
            return WorkspaceSettings()
        removed_keys = sorted(_REMOVED_TOOL_SECURITY_KEYS & raw.keys())
        if removed_keys:
            raise ValueError(
                "Removed runtime settings are not supported: "
                + ", ".join(removed_keys)
            )

        current_mode = raw.get("current_mode")
        planning_budget_profile = raw.get("planning_budget_profile")
        permission_mode = raw.get("tool_permission_mode")

        return WorkspaceSettings(
            provider=_string(raw.get("provider")),
            model_id=_string(raw.get("model_id")),
            system_prompt=_string(raw.get("system_prompt")),
            thinking_level=_string(raw.get("thinking_level")),
            current_mode=cast(RunMode, current_mode)
            if current_mode in {"read", "plan", "build"}
            else None,
            planning_budget_profile=cast(PlanningBudgetProfile, planning_budget_profile)
            if planning_budget_profile in {"conservative", "balanced", "wide"}
            else None,
            max_tool_calls_per_turn=_positive_int(raw.get("max_tool_calls_per_turn")),
            retry_enabled=_bool(raw.get("retry_enabled")),
            max_retries=_positive_int(raw.get("max_retries")),
            retry_base_delay_ms=_positive_int(raw.get("retry_base_delay_ms")),
            tool_permission_mode=cast(RuntimePermissionMode, permission_mode)
            if permission_mode in _PERMISSION_MODES
            else None,
            edit_require_unique_match=_bool(raw.get("edit_require_unique_match")),
            prompt_guidelines=_string_list(raw.get("prompt_guidelines")),
            append_system_prompt=_string(raw.get("append_system_prompt")),
            extension_paths=_string_list(raw.get("extension_paths")),
            skill_paths=_string_list(raw.get("skill_paths")),
            prompt_debug_sources=_bool(raw.get("prompt_debug_sources")),
            mcp_servers=_object_list(raw.get("mcp_servers")),
            shell_timeout_seconds=_positive_int(raw.get("shell_timeout_seconds")),
            shell_max_timeout_seconds=_positive_int(raw.get("shell_max_timeout_seconds")),
            shell_stdout_limit=_positive_int(raw.get("shell_stdout_limit")),
            shell_stderr_limit=_positive_int(raw.get("shell_stderr_limit")),
            shell_allowed_env=_string_list(raw.get("shell_allowed_env")),
        )

    def _load_model(self) -> WorkspaceModelConfig | None:
        raw = self._load_json_object(self.model_file)
        if raw is None:
            return None
        try:
            return WorkspaceModelConfig(
                api=str(raw.get("api", "")),
                provider=str(raw.get("provider", "")),
                model_id=str(raw.get("model_id", "")),
                base_url=str(raw.get("base_url", "")),
                api_key=_string(raw.get("api_key")),
                api_key_env=_string(raw.get("api_key_env")),
                context_window=int(raw.get("context_window", 128_000)),
                max_tokens=int(raw.get("max_tokens", 8192)),
                reasoning=bool(raw.get("reasoning", False)),
                vision=bool(raw.get("vision", False)),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid model config {self.model_file}: {exc}") from exc

    def _load_prompt(self) -> str | None:
        if not self.prompt_file.exists():
            return None
        text = self.prompt_file.read_text(encoding="utf-8").strip()
        return text or None

    def _load_tools(self) -> list[str] | None:
        raw = self._load_json_object(self.tools_file)
        if raw is None:
            return None
        enabled = raw.get("enabled")
        return _string_list(enabled)

    @staticmethod
    def _load_json_object(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return raw if isinstance(raw, dict) else None


def load_runtime_config(intent: "SessionOpenIntent") -> RuntimeConfig:
    """Load workspace resources and merge them with the open-session intent."""

    workspace = Path(intent.workspace_dir)
    resources = (
        WorkspaceResourceLoader(workspace).load()
        if intent.load_workspace_resources
        else WorkspaceResources()
    )
    settings = resources.settings
    restored = load_session_open_metadata(workspace, intent.session_id)
    sources: dict[str, ConfigValueSource] = {}

    def choose(
        name: str,
        *candidates: tuple[ConfigValueSource, Any | None],
        default: Any,
    ) -> Any:
        for source, value in candidates:
            if value is not None:
                sources[name] = source
                return value
        sources[name] = ConfigValueSource("default")
        return default

    system_prompt = choose(
        "system_prompt",
        (_cli_source(), intent.system_prompt),
        (ConfigValueSource("session", intent.session_id), restored.system_prompt if restored else None),
        (_project_source("prompt.md"), resources.prompt),
        (_project_source("settings.json"), settings.system_prompt),
        default="",
    )
    current_mode = ensure_run_mode(
        choose(
            "current_mode",
            (_cli_source(), intent.current_mode),
            (_project_source("settings.json"), settings.current_mode),
            default="build",
        )
    )
    planning_budget_profile = ensure_planning_budget_profile(
        choose(
            "planning_budget_profile",
            (_cli_source(), intent.planning_budget_profile),
            (_project_source("settings.json"), settings.planning_budget_profile),
            default="balanced",
        )
    )
    permission_mode = choose(
        "tool_permission_mode",
        (_cli_source(), intent.tool_permission_mode),
        (_project_source("settings.json"), settings.tool_permission_mode),
        default=("read-only" if current_mode in {"read", "plan"} else "workspace-write"),
    )
    permission_mode = _permission_mode(permission_mode)
    if current_mode in {"read", "plan"}:
        permission_mode = "read-only"

    shell_max_timeout_seconds = _positive_int(
        choose(
            "shell_max_timeout_seconds",
            (_cli_source(), intent.shell_max_timeout_seconds),
            (_project_source("settings.json"), settings.shell_max_timeout_seconds),
            default=120,
        ),
        default=120,
    )

    return RuntimeConfig(
        workspace=workspace,
        settings=settings,
        local_model=resources.model,
        restored=restored,
        system_prompt=system_prompt,
        thinking_level=str(
            choose(
                "thinking_level",
                (_cli_source(), intent.thinking_level),
                (_project_source("settings.json"), settings.thinking_level),
                default="off",
            )
        ),
        current_mode=current_mode,
        planning_budget_profile=planning_budget_profile,
        max_tool_calls_per_turn=_positive_int(
            choose(
                "max_tool_calls_per_turn",
                (_cli_source(), intent.max_tool_calls_per_turn),
                (_project_source("settings.json"), settings.max_tool_calls_per_turn),
                default=16,
            ),
            default=16,
        ),
        retry_enabled=bool(
            choose(
                "retry_enabled",
                (_cli_source(), intent.retry_enabled),
                (_project_source("settings.json"), settings.retry_enabled),
                default=True,
            )
        ),
        max_retries=_positive_int(
            choose(
                "max_retries",
                (_cli_source(), intent.max_retries),
                (_project_source("settings.json"), settings.max_retries),
                default=2,
            ),
            default=2,
        ),
        retry_base_delay_ms=_positive_int(
            choose(
                "retry_base_delay_ms",
                (_cli_source(), intent.retry_base_delay_ms),
                (_project_source("settings.json"), settings.retry_base_delay_ms),
                default=1200,
            ),
            default=1200,
        ),
        tool_permission_mode=permission_mode,
        edit_require_unique_match=bool(
            choose(
                "edit_require_unique_match",
                (_cli_source(), intent.edit_require_unique_match),
                (_project_source("settings.json"), settings.edit_require_unique_match),
                default=True,
            )
        ),
        prompt_guidelines=choose(
            "prompt_guidelines",
            (_cli_source(), intent.prompt_guidelines),
            (_project_source("settings.json"), settings.prompt_guidelines),
            default=None,
        ),
        append_system_prompt=choose(
            "append_system_prompt",
            (_cli_source(), intent.append_system_prompt),
            (_project_source("settings.json"), settings.append_system_prompt),
            default=None,
        ),
        extension_paths=choose(
            "extension_paths",
            (_cli_source(), intent.extension_paths),
            (_project_source("settings.json"), settings.extension_paths),
            default=None,
        ),
        skill_paths=choose(
            "skill_paths",
            (_cli_source(), intent.skill_paths),
            (_project_source("settings.json"), settings.skill_paths),
            default=None,
        ),
        prompt_debug_sources=bool(
            choose(
                "prompt_debug_sources",
                (_cli_source(), intent.prompt_debug_sources),
                (_project_source("settings.json"), settings.prompt_debug_sources),
                default=False,
            )
        ),
        mcp_servers=choose(
            "mcp_servers",
            (_cli_source(), intent.mcp_servers),
            (_project_source("settings.json"), settings.mcp_servers),
            default=None,
        ),
        enabled_builtin_tools=choose(
            "enabled_builtin_tools",
            (_cli_source(), intent.enabled_builtin_tools),
            (_project_source("tools.json"), resources.enabled_tools),
            default=None,
        ),
        shell_timeout_seconds=min(
            _positive_int(
                choose(
                    "shell_timeout_seconds",
                    (_cli_source(), intent.shell_timeout_seconds),
                    (_project_source("settings.json"), settings.shell_timeout_seconds),
                    default=30,
                ),
                default=30,
            ),
            min(120, shell_max_timeout_seconds),
        ),
        shell_max_timeout_seconds=min(120, shell_max_timeout_seconds),
        shell_stdout_limit=_positive_int(
            choose(
                "shell_stdout_limit",
                (_cli_source(), intent.shell_stdout_limit),
                (_project_source("settings.json"), settings.shell_stdout_limit),
                default=20_000,
            ),
            default=20_000,
        ),
        shell_stderr_limit=_positive_int(
            choose(
                "shell_stderr_limit",
                (_cli_source(), intent.shell_stderr_limit),
                (_project_source("settings.json"), settings.shell_stderr_limit),
                default=10_000,
            ),
            default=10_000,
        ),
        shell_allowed_env=choose(
            "shell_allowed_env",
            (_cli_source(), intent.shell_allowed_env),
            (_project_source("settings.json"), settings.shell_allowed_env),
            default=None,
        ),
        sources=sources,
    )


def explain_session_open_config(
    intent: "SessionOpenIntent",
    key: str,
) -> ResolvedConfigValue:
    config = load_runtime_config(intent)
    if key in {"model", "provider", "model_id"}:
        from .model import resolve_runtime_model

        resolved = resolve_runtime_model(intent, config)
        model = resolved.model
        values = {
            "model": f"{model.provider}/{model.id}" if model.provider else model.id,
            "provider": model.provider,
            "model_id": model.id,
        }
        return ResolvedConfigValue(key=key, value=values[key], source=resolved.source)
    if key not in config.sources or not hasattr(config, key):
        raise UnknownRuntimeConfigKeyError(key)
    return ResolvedConfigValue(
        key=key,
        value=getattr(config, key),
        source=config.sources[key],
    )


def resolve_workspace_session_intent(
    workspace: str | Path,
    *,
    provider: str | None = None,
    model_id: str | None = None,
):
    if bool(provider) != bool(model_id):
        raise ValueError("--provider and --model must be provided together")

    from .actions import SessionOpenIntent

    workspace_path = Path(workspace)
    if provider and model_id:
        return SessionOpenIntent(
            workspace_dir=workspace_path,
            provider=provider,
            model_id=model_id,
        )
    resources = WorkspaceResourceLoader(workspace_path).load()
    if resources.model is not None:
        return SessionOpenIntent(
            workspace_dir=workspace_path,
            model=resources.model.to_model(),
            get_api_key=resources.model.build_api_key_resolver(),
        )
    if resources.settings.provider and resources.settings.model_id:
        return SessionOpenIntent(
            workspace_dir=workspace_path,
            provider=resources.settings.provider,
            model_id=resources.settings.model_id,
        )
    raise ValueError("No project model config found; provide --provider and --model")


def check_workspace_model_config(workspace: str | Path) -> WorkspaceConfigCheck:
    loader = WorkspaceResourceLoader(workspace)
    model = loader.load().model
    if model is None:
        raise ValueError(f"Model config not found: {loader.model_file}")
    if model.api_key_env and os.getenv(model.api_key_env):
        credential_source = f"environment:{model.api_key_env}"
    elif model.api_key:
        credential_source = "local-file (do not commit)"
    else:
        credential_source = "missing"
    rows: list[tuple[str, object]] = [
        ("config", loader.model_file),
        ("api", model.api),
        ("provider", model.provider),
        ("model_id", model.model_id),
        ("base_url", model.base_url),
        ("credential", credential_source),
    ]
    if credential_source == "missing":
        rows.append(("status", "MISSING_CREDENTIAL"))
        rows.append(("next", f"Set {model.api_key_env} or add api_key to {loader.model_file}"))
        border_style = "warning"
    else:
        rows.append(("status", "valid"))
        border_style = "success"
    return WorkspaceConfigCheck(rows=tuple(rows), border_style=border_style)


def describe_workspace_config(workspace: str | Path) -> WorkspaceConfigView:
    resources = WorkspaceResourceLoader(workspace).load()
    model_rows: list[tuple[str, object]] = []
    settings_rows: list[tuple[str, object]] = []

    if resources.model:
        model = resources.model
        model_rows.extend(
            [
                ("provider", model.provider),
                ("model_id", model.model_id),
                ("base_url", model.base_url),
                ("api", model.api),
            ]
        )
        if model.api_key_env and os.getenv(model.api_key_env):
            model_rows.append(("credential", f"env:{model.api_key_env}"))
        elif model.api_key:
            model_rows.append(("credential", "local-file (do not commit)"))
        else:
            model_rows.append(("credential", "[warning]MISSING[/warning]"))
    else:
        model_rows.append(("status", "[dim](not configured)[/dim]"))

    for field_info in dataclasses.fields(resources.settings):
        value = getattr(resources.settings, field_info.name)
        if value is None or _is_sensitive_config_field(field_info.name):
            continue
        settings_rows.append((field_info.name, str(value)))
    if not settings_rows:
        settings_rows.append(("status", "[dim](using defaults)[/dim]"))

    return WorkspaceConfigView(
        model_rows=tuple(model_rows),
        settings_rows=tuple(settings_rows),
    )


def _cli_source() -> ConfigValueSource:
    return ConfigValueSource("cli")


def _project_source(file_name: str) -> ConfigValueSource:
    return ConfigValueSource("project", f".codepilot/{file_name}")


def _permission_mode(value: object) -> RuntimePermissionMode:
    if value not in _PERMISSION_MODES:
        raise ValueError(f"Unknown permission mode: {value}")
    return cast(RuntimePermissionMode, value)


def _string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _positive_int(value: object, *, default: int | None = None) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return default
    return value


def _string_list(value: object) -> list[str] | None:
    if not isinstance(value, list):
        return None
    return [item for item in value if isinstance(item, str)]


def _string_map(value: object) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, str] = {}
    for key, item in value.items():
        if isinstance(key, str) and isinstance(item, str):
            result[key] = item
    return result or None


def _object_list(value: object) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    result = [dict(item) for item in value if isinstance(item, dict)]
    return result or None


def _is_sensitive_config_field(name: str) -> bool:
    lower_name = name.lower()
    return "key" in lower_name or "secret" in lower_name


__all__ = [
    "ConfigSourceKind",
    "ConfigValueSource",
    "ResolvedConfigValue",
    "RuntimeConfig",
    "RuntimePermissionMode",
    "UnknownRuntimeConfigKeyError",
    "WorkspaceConfigCheck",
    "WorkspaceConfigView",
    "WorkspaceModelConfig",
    "WorkspaceResourceLoader",
    "WorkspaceResources",
    "WorkspaceSettings",
    "check_workspace_model_config",
    "describe_workspace_config",
    "explain_session_open_config",
    "load_runtime_config",
    "resolve_workspace_session_intent",
]
