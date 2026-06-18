from __future__ import annotations

"""Model resolution for runtime session creation."""

from dataclasses import dataclass
from typing import Awaitable, Callable

from codepilot.llm.models import get_model
from codepilot.protocols import Model

from .config import RuntimeInputs
from .types import CreateAgentSessionOptions


@dataclass(frozen=True)
class ResolvedModel:
    model: Model
    get_api_key: Callable[[str], str | None | Awaitable[str | None]] | None = None


def resolve_model(
    options: CreateAgentSessionOptions,
    inputs: RuntimeInputs,
) -> ResolvedModel:
    resources = inputs.resources
    model = options.model
    if model is not None:
        return ResolvedModel(model=model, get_api_key=options.get_api_key)

    if options.provider and options.model_id:
        return ResolvedModel(
            model=get_model(options.provider, options.model_id),
            get_api_key=options.get_api_key,
        )

    restored_meta = inputs.restored_meta or {}
    provider = restored_meta.get("provider")
    model_id = restored_meta.get("model_id")
    if isinstance(provider, str) and isinstance(model_id, str):
        return ResolvedModel(
            model=get_model(provider, model_id),
            get_api_key=options.get_api_key,
        )

    if resources and resources.model is not None:
        return ResolvedModel(
            model=resources.model.to_model(),
            get_api_key=options.get_api_key or resources.model.build_api_key_resolver(),
        )

    if resources and resources.settings.provider and resources.settings.model_id:
        return ResolvedModel(
            model=get_model(resources.settings.provider, resources.settings.model_id),
            get_api_key=options.get_api_key,
        )

    raise ValueError(
        "Unable to resolve model: create .codepilot/model.local.json "
        "or provide --provider and --model-id"
    )
