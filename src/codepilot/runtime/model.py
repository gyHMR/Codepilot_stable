"""适配 Runtime 模型调用与 Context 摘要模型能力。"""

from __future__ import annotations

"""Resolve the model and credentials for an opened runtime session."""

import os
import asyncio
import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from codepilot.llm.catalog import get_env_api_key_name, get_model
from codepilot.llm.ports import (
    LLMCompleted,
    LLMCorrelation,
    LLMFailed,
    LLMOptions,
    LLMRequest,
    LLMStarted,
    LLMTextDelta,
    ModelDescriptor,
)
from codepilot.protocols import (
    AssistantMessage,
    Message,
    Model,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from codepilot.sessions.context.contracts import (
    CompactSummary,
    ContextSummaryRequest,
    ContextSummaryResult,
)
from codepilot.sessions.memory import MemoryProposal

from .config import ConfigValueSource, RuntimeConfig
from .actions import SessionOpenIntent


@dataclass(frozen=True)
class RuntimeModel:
    """把 LLM 适配器暴露为 Core 所需模型端口。"""
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
    finalization_sink: (
        Callable[[str, tuple[MemoryProposal, ...], str | None], None] | None
    ) = None

    async def stream(self, request):
        if request.correlation.purpose == "finalization":
            async for event in self._stream_finalization(request):
                yield event
            return
        retries = max(0, self.max_retries) if self.enabled else 0
        for attempt in range(retries + 1):
            retry = False
            buffered = []
            async for event in self.base.stream(request):
                if (
                    isinstance(event, LLMFailed)
                    and attempt < retries
                    and _retryable_error(event.error)
                ):
                    retry = True
                    break
                buffered.append(event)
            if not retry:
                attempts = attempt + 1
                for event in buffered:
                    if isinstance(event, (LLMCompleted, LLMFailed)):
                        event = replace(event, attempts=attempts)
                    yield event
                return
            if self.base_delay_ms > 0:
                await asyncio.sleep(self.base_delay_ms / 1000)

    async def _stream_finalization(self, request):
        retries = max(0, self.max_retries) if self.enabled else 0
        for attempt in range(retries + 1):
            retry = False
            buffered = []
            async for event in self.base.stream(request):
                if (
                    isinstance(event, LLMFailed)
                    and attempt < retries
                    and _retryable_error(event.error)
                ):
                    retry = True
                    break
                buffered.append(event)
            if retry:
                if self.base_delay_ms > 0:
                    await asyncio.sleep(self.base_delay_ms / 1000)
                continue
            completed = next(
                (event for event in reversed(buffered) if isinstance(event, LLMCompleted)),
                None,
            )
            if completed is None:
                for event in buffered:
                    yield event
                return
            cleaned, proposals, error = _extract_memory_sidecar(completed.message)
            if self.finalization_sink is not None:
                self.finalization_sink(
                    request.correlation.run_id,
                    proposals,
                    error,
                )
            if any(isinstance(event, LLMStarted) for event in buffered):
                yield LLMStarted()
            visible_text = _assistant_text(cleaned)
            if visible_text:
                yield LLMTextDelta(visible_text)
            yield LLMCompleted(cleaned, completed.usage, attempts=attempt + 1)
            return


@dataclass(frozen=True)
class RuntimeContextSummarizer:
    """为 Context 压缩提供隔离的模型摘要能力。"""
    model_port: Any
    model: ModelDescriptor

    async def summarize(self, request: ContextSummaryRequest) -> ContextSummaryResult:
        payload = {
            "original_goal": request.original_goal,
            "previous_summary": (
                request.previous_summary.to_mapping()
                if request.previous_summary is not None
                else None
            ),
            "messages": [_summary_message(message) for message in request.messages],
        }
        llm_request = LLMRequest(
            model=self.model,
            system_prompt=(
                "Summarize the supplied coding-agent history as one JSON object. "
                "Return only these fields: user_constraints, decisions, completed_work, "
                "files_and_symbols, important_evidence, errors_and_resolutions, "
                "verification_state, open_questions, next_actions. Values except "
                "verification_state must be arrays of concise strings. Do not call tools, "
                "do not propose long-term memory, and do not include raw tool output."
            ),
            messages=(
                UserMessage(
                    content=json.dumps(payload, ensure_ascii=False),
                    metadata={"context_summary_request": True},
                ),
            ),
            tools=(),
            options=LLMOptions(temperature=0.0, max_tokens=1200, reasoning="low"),
            correlation=LLMCorrelation(
                run_id="context_summary",
                session_id="",
                purpose="context_summary",
            ),
        )
        completed: LLMCompleted | None = None
        async for event in self.model_port.stream(llm_request):
            if isinstance(event, LLMFailed):
                raise RuntimeError(f"Context summarizer failed: {event.error}")
            if isinstance(event, LLMCompleted):
                completed = event
        if completed is None:
            raise RuntimeError("Context summarizer returned no completed message")
        raw = _json_object(_assistant_text(completed.message))
        raw["original_goal"] = request.original_goal
        raw["source_refs"] = [
            f"message:{_message_id(message)}" for message in request.messages
        ]
        summary = CompactSummary.from_mapping(raw)
        return ContextSummaryResult(
            summary=summary,
            compacted_until_message_id=_message_id(request.messages[-1]),
        )


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


_MEMORY_SIDECAR = re.compile(
    r"<codepilot-memory-proposals>(.*?)</codepilot-memory-proposals>",
    re.DOTALL,
)


def _extract_memory_sidecar(
    message: AssistantMessage,
) -> tuple[AssistantMessage, tuple[MemoryProposal, ...], str | None]:
    full_text = _assistant_text(message)
    matches = list(_MEMORY_SIDECAR.finditer(full_text))
    cleaned_text = _MEMORY_SIDECAR.sub("", full_text).strip()
    non_text = [block for block in message.content if not isinstance(block, TextContent)]
    cleaned = _assistant_with_content(
        message,
        [*non_text, TextContent(text=cleaned_text)],
    )
    if not matches:
        return cleaned, (), None
    if len(matches) != 1:
        return cleaned, (), "finalization sidecar must appear at most once"
    try:
        payload = json.loads(matches[0].group(1))
        if not isinstance(payload, dict) or set(payload) != {"proposals"}:
            raise ValueError("sidecar must contain only proposals")
        raw_proposals = payload["proposals"]
        if not isinstance(raw_proposals, list) or len(raw_proposals) > 3:
            raise ValueError("sidecar proposals must be a list with at most three items")
        proposals = tuple(
            MemoryProposal(
                scope=raw["scope"],
                type=raw["type"],
                key=raw["key"],
                content=raw["content"],
            )
            for raw in raw_proposals
            if isinstance(raw, dict)
            and set(raw) == {"scope", "type", "key", "content"}
        )
        if len(proposals) != len(raw_proposals):
            raise ValueError("sidecar proposal fields are invalid")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return cleaned, (), str(exc)
    return cleaned, proposals, None


def _assistant_text(message: AssistantMessage) -> str:
    return "".join(
        block.text for block in message.content if isinstance(block, TextContent)
    ).strip()


def _summary_message(message: Message) -> dict[str, object]:
    if isinstance(message, AssistantMessage):
        content = [
            block.text
            if isinstance(block, TextContent)
            else {
                "tool_call_id": block.id,
                "tool_name": block.name,
                "arguments": block.arguments,
            }
            if isinstance(block, ToolCall)
            else "[thinking omitted]"
            for block in message.content
        ]
    elif isinstance(message, ToolResultMessage):
        content = {
            "tool_call_id": message.tool_call_id,
            "tool_name": message.tool_name,
            "status": message.status,
            "text": "".join(
                block.text
                for block in message.content
                if isinstance(block, TextContent)
            )[:1200],
            "affected_paths": list(message.affected_paths),
            "verification": message.verification,
        }
    elif isinstance(message, UserMessage):
        content = (
            message.content
            if isinstance(message.content, str)
            else [
                block.text if isinstance(block, TextContent) else "[image]"
                for block in message.content
            ]
        )
    else:
        content = ""
    return {
        "message_id": _message_id(message),
        "role": message.role,
        "content": content,
    }


def _json_object(text: str) -> dict[str, object]:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("Context summary must be a JSON object")
    return payload


def _message_id(message: Message) -> str:
    value = message.metadata.get("session_message_id")
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError("Context summary messages require session_message_id")
    return text


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
        local_model = config.local_model
        if (
            local_model is not None
            and local_model.provider == restored.provider
            and local_model.model_id == restored.model_id
        ):
            return _with_credentials(
                intent,
                config,
                model=local_model.to_model(),
                get_api_key=intent.get_api_key or local_model.build_api_key_resolver(),
                source=ConfigValueSource("project", ".codepilot/model.local.json"),
            )
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
    "RuntimeContextSummarizer",
    "RuntimeModel",
    "convert_to_llm",
    "resolve_runtime_model",
]
