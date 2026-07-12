from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Literal, Protocol

from codepilot.protocols import (
    AssistantMessage,
    LLMErrorInfo,
    Message,
    Tool,
    ToolCall,
    Usage,
)


@dataclass(frozen=True)
class ModelDescriptor:
    provider: str
    model_id: str
    capabilities: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMOptions:
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning: str | None = None
    timeout_seconds: float = 120.0


@dataclass(frozen=True)
class LLMCorrelation:
    run_id: str = ""
    session_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", str(self.run_id).strip())
        object.__setattr__(self, "session_id", str(self.session_id).strip())


@dataclass(frozen=True)
class LLMRequest:
    model: ModelDescriptor
    messages: tuple[Message, ...]
    system_prompt: str = ""
    tools: tuple[Tool, ...] = field(default_factory=tuple)
    options: LLMOptions = field(default_factory=LLMOptions)
    correlation: LLMCorrelation = field(default_factory=LLMCorrelation)


@dataclass(frozen=True)
class LLMStarted:
    kind: Literal["started"] = "started"


@dataclass(frozen=True)
class LLMTextDelta:
    text: str
    kind: Literal["text_delta"] = "text_delta"


@dataclass(frozen=True)
class LLMReasoningDelta:
    text: str
    kind: Literal["reasoning_delta"] = "reasoning_delta"


@dataclass(frozen=True)
class LLMToolCallDelta:
    tool_call: ToolCall
    kind: Literal["tool_call_delta"] = "tool_call_delta"


@dataclass(frozen=True)
class LLMCompleted:
    message: AssistantMessage
    usage: Usage | None = None
    kind: Literal["completed"] = "completed"


@dataclass(frozen=True)
class LLMFailed:
    error: LLMErrorInfo | Exception | dict[str, object]
    kind: Literal["failed"] = "failed"


LLMEvent = (
    LLMStarted
    | LLMTextDelta
    | LLMReasoningDelta
    | LLMToolCallDelta
    | LLMCompleted
    | LLMFailed
)


class ModelPort(Protocol):
    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        ...


__all__ = [
    "LLMCompleted",
    "LLMCorrelation",
    "LLMEvent",
    "LLMFailed",
    "LLMOptions",
    "LLMRequest",
    "LLMReasoningDelta",
    "LLMStarted",
    "LLMTextDelta",
    "LLMToolCallDelta",
    "ModelDescriptor",
    "ModelPort",
]
