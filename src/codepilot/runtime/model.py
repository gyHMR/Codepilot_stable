from __future__ import annotations

"""Resolve the model and credentials for an opened runtime session."""

import os
from dataclasses import dataclass
from typing import Any

from codepilot.llm.catalog import get_env_api_key_name, get_model
from codepilot.protocols import Model

from .config import ConfigValueSource, RuntimeConfig
from .opening import SessionOpenIntent


@dataclass(frozen=True)
class RuntimeModel:
    model: Model
    get_api_key: Any | None
    source: ConfigValueSource
    credential_source: str
    credential_location: str | None = None

    @property
    def display_name(self) -> str:
        return f"{self.model.provider}/{self.model.id}" if self.model.provider else self.model.id


def resolve_runtime_model(
    intent: SessionOpenIntent,
    config: RuntimeConfig,
) -> RuntimeModel:
    """Choose the effective model from intent, restored session, or workspace files."""

    if intent.model is not None:
        return _with_credentials(
            intent,
            config,
            model=intent.model,
            get_api_key=intent.get_api_key,
            source=ConfigValueSource("cli"),
        )

    if intent.provider and intent.model_id:
        return _with_credentials(
            intent,
            config,
            model=get_model(intent.provider, intent.model_id),
            get_api_key=intent.get_api_key,
            source=ConfigValueSource("cli"),
        )

    restored = config.restored
    if restored and restored.provider and restored.model_id:
        return _with_credentials(
            intent,
            config,
            model=get_model(restored.provider, restored.model_id),
            get_api_key=intent.get_api_key,
            source=ConfigValueSource("session", intent.session_id),
        )

    if config.local_model is not None:
        return _with_credentials(
            intent,
            config,
            model=config.local_model.to_model(),
            get_api_key=intent.get_api_key or config.local_model.build_api_key_resolver(),
            source=ConfigValueSource("project", ".codepilot/model.local.json"),
        )

    settings = config.settings
    if settings.provider and settings.model_id:
        return _with_credentials(
            intent,
            config,
            model=get_model(settings.provider, settings.model_id),
            get_api_key=intent.get_api_key,
            source=ConfigValueSource("project", ".codepilot/settings.json"),
        )

    raise ValueError(
        "Unable to resolve model: create .codepilot/model.local.json "
        "or provide --model provider/model-id"
    )


def _with_credentials(
    intent: SessionOpenIntent,
    config: RuntimeConfig,
    *,
    model: Model,
    get_api_key: Any | None,
    source: ConfigValueSource,
) -> RuntimeModel:
    credential_source, credential_location = _credential_source(
        intent,
        config,
        model,
        source,
        get_api_key=get_api_key,
    )
    return RuntimeModel(
        model=model,
        get_api_key=get_api_key,
        source=source,
        credential_source=credential_source,
        credential_location=credential_location,
    )


def _credential_source(
    intent: SessionOpenIntent,
    config: RuntimeConfig,
    model: Model,
    source: ConfigValueSource,
    *,
    get_api_key: Any | None,
) -> tuple[str, str | None]:
    if get_api_key is not None or intent.get_api_key is not None:
        return "caller", "get_api_key function"

    if source.location == ".codepilot/model.local.json" and config.local_model:
        local = config.local_model
        if local.api_key_env and os.getenv(local.api_key_env):
            return "env", local.api_key_env
        if local.api_key:
            return "local-file", ".codepilot/model.local.json"

    env_name = get_env_api_key_name(model.provider)
    if env_name and os.getenv(env_name):
        return "env", env_name
    if model.api == "openai-compatible":
        fallback = get_env_api_key_name("openai")
        if fallback and os.getenv(fallback):
            return "env", fallback
    return "missing", None


__all__ = [
    "RuntimeModel",
    "resolve_runtime_model",
]
