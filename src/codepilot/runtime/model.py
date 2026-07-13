from __future__ import annotations

"""Resolve the model and credentials for an opened runtime session."""

import os
import asyncio
from collections import Counter
from dataclasses import dataclass
from typing import Any

from codepilot.llm.catalog import get_env_api_key_name, get_model
from codepilot.llm.ports import LLMFailed
from codepilot.protocols import (
    AssistantMessage,
    Message,
    Model,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
)

from .config import ConfigValueSource, RuntimeConfig
from .actions import SessionOpenIntent


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


@dataclass(frozen=True)
class RetryingModelPort:
    """Runtime-owned provider retry wrapper; Core still observes one model action."""

    base: Any
    enabled: bool = True
    max_retries: int = 0
    base_delay_ms: int = 0

    async def stream(self, request):
        retries = max(0, self.max_retries) if self.enabled else 0
        for attempt in range(retries + 1):
            retry = False
            async for event in self.base.stream(request):
                if (
                    isinstance(event, LLMFailed)
                    and attempt < retries
                    and _retryable_error(event.error)
                ):
                    retry = True
                    break
                yield event
            if not retry:
                return
            if self.base_delay_ms > 0:
                await asyncio.sleep(self.base_delay_ms / 1000)


def convert_to_llm(
    messages: list[Message],
    *,
    strip_thinking: bool = False,
    thinking_to_text: bool = False,
) -> list[Message]:
    """Apply provider-facing message transforms without Core context governance."""

    converted: list[Message] = []
    for message in messages:
        if not isinstance(message, AssistantMessage):
            converted.append(message)
            continue
        content: list[Any] = []
        for block in message.content:
            if isinstance(block, ThinkingContent):
                if strip_thinking:
                    continue
                if thinking_to_text and block.thinking:
                    content.append(
                        TextContent(
                            text=f"[thinking]\n{block.thinking}\n[/thinking]"
                        )
                    )
                    continue
            content.append(block)
        converted.append(
            _assistant_with_content(
                message,
                content or [TextContent(text="(no content)")],
            )
        )
    return _repair_tool_boundaries(converted)


def _repair_tool_boundaries(messages: list[Message]) -> list[Message]:
    """Keep only provider-valid ToolCall/ToolResult pairs."""

    remaining_results = Counter(
        message.tool_call_id
        for message in messages
        if isinstance(message, ToolResultMessage) and message.tool_call_id
    )
    pending: dict[str, ToolCall] = {}
    repaired: list[Message] = []
    for message in messages:
        if isinstance(message, ToolResultMessage):
            if message.tool_call_id:
                remaining_results[message.tool_call_id] -= 1
            if message.tool_call_id not in pending:
                continue
            pending.pop(message.tool_call_id, None)
            repaired.append(message)
            continue
        if isinstance(message, AssistantMessage):
            content = [
                block
                for block in message.content
                if not isinstance(block, ToolCall)
                or remaining_results[block.id] > 0
            ]
            if not content:
                continue
            paired = _assistant_with_content(
                message,
                content,
                stop_reason=(
                    message.stop_reason
                    if any(isinstance(block, ToolCall) for block in content)
                    else "stop"
                ),
            )
            pending.update(
                (block.id, block)
                for block in paired.content
                if isinstance(block, ToolCall) and block.id
            )
            repaired.append(paired)
            continue
        repaired.append(message)
    return repaired


def _assistant_with_content(
    message: AssistantMessage,
    content: list[Any],
    *,
    stop_reason: str | None = None,
) -> AssistantMessage:
    return AssistantMessage(
        role=message.role,
        content=content,
        api=message.api,
        provider=message.provider,
        model=message.model,
        usage=message.usage,
        stop_reason=message.stop_reason if stop_reason is None else stop_reason,
        response_id=message.response_id,
        error_message=message.error_message,
        error_info=message.error_info,
        timestamp=message.timestamp,
        metadata=dict(message.metadata),
    )


def _retryable_error(error: object) -> bool:
    if isinstance(error, dict):
        return bool(error.get("retryable", False))
    return bool(getattr(error, "retryable", False))


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
    "RetryingModelPort",
    "RuntimeModel",
    "convert_to_llm",
    "resolve_runtime_model",
]
