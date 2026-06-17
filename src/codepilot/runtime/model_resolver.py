from __future__ import annotations

"""Model resolution for runtime session creation."""

from typing import Any

from codepilot.llm.models import get_model
from codepilot.llm.types import Model

from .resources import WorkspaceResources
from .types import CreateAgentSessionOptions


def resolve_model(
    options: CreateAgentSessionOptions,
    resources: WorkspaceResources | None,
    restored_meta: dict[str, Any] | None,
) -> Model:
    model = options.model
    if model is None and options.provider and options.model_id:
        model = get_model(options.provider, options.model_id)
    if model is None and resources and resources.settings.provider and resources.settings.model_id:
        model = get_model(resources.settings.provider, resources.settings.model_id)
    if model is None and restored_meta:
        provider = restored_meta.get("provider")
        model_id = restored_meta.get("model_id")
        if isinstance(provider, str) and isinstance(model_id, str):
            model = get_model(provider, model_id)
    if model is None:
        raise ValueError("Unable to resolve model: provide model/provider+model_id or a valid existing session_id")
    return model
