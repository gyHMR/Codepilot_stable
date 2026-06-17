from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from .errors import LLMErrorInfo


Api = str
Provider = str
StopReason = Literal["stop", "length", "toolUse", "error", "aborted"]
ThinkingLevel = Literal["minimal", "low", "medium", "high", "xhigh"]
LLMStreamEventType = Literal[
    "start",
    "text_start",
    "text_delta",
    "text_end",
    "thinking_start",
    "thinking_delta",
    "thinking_end",
    "toolcall_start",
    "toolcall_delta",
    "toolcall_end",
    "done",
    "error",
]


@dataclass
class Cost:
    """Cost statistics. The unit is usually USD."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


@dataclass
class Usage:
    """Token usage statistics."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    cost: Cost = field(default_factory=Cost)


@dataclass
class ModelCapabilities:
    """Capabilities advertised by a concrete model."""

    tools: bool = True
    vision: bool = False
    json_schema: bool = False
    streaming: bool = True
    reasoning: bool = False
    system_prompt: bool = True
    tool_choice: bool = False
    parallel_tool_calls: bool = False


@dataclass
class Model:
    """
    Concrete model configuration.

    api selects the wire protocol implementation.
    provider identifies the vendor/source for grouping and default auth.
    """

    id: str
    name: str
    api: Api
    provider: Provider
    base_url: str
    reasoning: bool
    input: list[Literal["text", "image"]]
    context_window: int
    max_tokens: int
    cost: Cost = field(default_factory=Cost)
    headers: dict[str, str] | None = None
    compat: dict[str, Any] | None = None
    capabilities: ModelCapabilities | None = None

    def __post_init__(self) -> None:
        if self.capabilities is None:
            self.capabilities = ModelCapabilities(
                vision="image" in self.input,
                reasoning=self.reasoning,
            )


@dataclass
class StreamOptions:
    """Common streaming-call options."""

    temperature: float | None = None
    max_tokens: int | None = None
    api_key: str | None = None
    headers: dict[str, str] | None = None
    timeout_seconds: float | None = None
    session_id: str | None = None


@dataclass
class SimpleStreamOptions(StreamOptions):
    """Simplified streaming options with a reasoning shortcut."""

    reasoning: ThinkingLevel | None = None


class LLMStreamEvent(TypedDict, total=False):
    """Provider-normalized stream event."""

    type: LLMStreamEventType
    partial: Any
    contentIndex: int
    delta: str
    content: str
    toolCall: Any
    reason: str
    message: Any
    error: Any
    errorInfo: LLMErrorInfo
    raw: Any


__all__ = [
    "Api",
    "Cost",
    "LLMStreamEvent",
    "LLMStreamEventType",
    "Model",
    "ModelCapabilities",
    "Provider",
    "SimpleStreamOptions",
    "StopReason",
    "StreamOptions",
    "ThinkingLevel",
    "Usage",
]
