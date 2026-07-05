from __future__ import annotations

# 新手导读：configuration.py 是 runtime 暴露给接口层的配置解释入口。
# 关注点：接口层提交 SessionOpenIntent；内部装配细节仍留在 runtime 层。

"""Runtime configuration views for interfaces."""

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .opening import SessionOpenIntent, _to_runtime_assembly_intent

if TYPE_CHECKING:
    from .assemble import ResolvedConfigValue


class UnknownRuntimeConfigKeyError(KeyError):
    """Requested runtime configuration key is not known."""


@dataclass(frozen=True)
class WorkspaceConfigCheck:
    """Sanitized model configuration check for interface rendering."""

    rows: tuple[tuple[str, object], ...]
    border_style: str


@dataclass(frozen=True)
class WorkspaceConfigView:
    """Sanitized workspace configuration summary for interface rendering."""

    model_rows: tuple[tuple[str, object], ...]
    settings_rows: tuple[tuple[str, object], ...]


def explain_session_open_config(
    intent: SessionOpenIntent,
    key: str,
) -> "ResolvedConfigValue":
    """Explain one resolved config value for a public open-session intent."""

    from .assemble import (
        UnknownRuntimeConfigKeyError as AssemblyConfigKeyError,
        explain_runtime_config,
    )

    try:
        return explain_runtime_config(_to_runtime_assembly_intent(intent), key)
    except AssemblyConfigKeyError as exc:
        raise UnknownRuntimeConfigKeyError(*exc.args) from exc


def resolve_workspace_session_intent(
    workspace: str | Path,
    *,
    provider: str | None = None,
    model_id: str | None = None,
) -> SessionOpenIntent:
    """Resolve a public session-open intent from CLI overrides or workspace config."""

    if bool(provider) != bool(model_id):
        raise ValueError("--provider and --model must be provided together")
    workspace_path = Path(workspace)
    if provider and model_id:
        return SessionOpenIntent(
            workspace_dir=workspace_path,
            provider=provider,
            model_id=model_id,
        )
    from .assemble import WorkspaceResourceLoader

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
    """Return a sanitized model-config health view for a workspace."""

    from .assemble import WorkspaceResourceLoader

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
    """Return sanitized model and settings rows for interface rendering."""

    from .assemble import WorkspaceResourceLoader

    loaded = WorkspaceResourceLoader(workspace).load()
    model = loaded.model
    settings = loaded.settings
    model_rows: list[tuple[str, object]] = []
    settings_rows: list[tuple[str, object]] = []

    if model:
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

    if settings and dataclasses.is_dataclass(settings):
        for field in dataclasses.fields(settings):
            value = getattr(settings, field.name)
            if value is None or _is_sensitive_config_field(field.name):
                continue
            settings_rows.append((field.name, str(value)))
    if not settings_rows:
        settings_rows.append(("status", "[dim](using defaults)[/dim]"))

    return WorkspaceConfigView(
        model_rows=tuple(model_rows),
        settings_rows=tuple(settings_rows),
    )


def _is_sensitive_config_field(name: str) -> bool:
    lower_name = name.lower()
    return "key" in lower_name or "secret" in lower_name


__all__ = [
    "UnknownRuntimeConfigKeyError",
    "WorkspaceConfigCheck",
    "WorkspaceConfigView",
    "check_workspace_model_config",
    "describe_workspace_config",
    "explain_session_open_config",
    "resolve_workspace_session_intent",
]
