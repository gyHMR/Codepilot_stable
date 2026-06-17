from __future__ import annotations

from codepilot.protocols import LLMStreamEvent, LLMStreamEventType


def llm_event(event_type: LLMStreamEventType, **payload: object) -> LLMStreamEvent:
    """Build a normalized LLM stream event."""

    return {"type": event_type, **payload}  # type: ignore[typeddict-item]


__all__ = [
    "LLMStreamEvent",
    "LLMStreamEventType",
    "llm_event",
]
