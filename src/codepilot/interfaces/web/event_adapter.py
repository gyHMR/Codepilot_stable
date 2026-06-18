from __future__ import annotations

"""Adapters from runtime/core events to Web Console event envelopes."""

from dataclasses import asdict, is_dataclass
from typing import Any

from codepilot.protocols.events import AgentEvent

from .schemas import WebEventEnvelope


def agent_event_to_web(event: AgentEvent) -> WebEventEnvelope:
    return WebEventEnvelope(
        type="agent_event",
        session_id=event.get("sessionId"),
        payload=_jsonable(event),
    )


def error_to_web(message: str, *, session_id: str | None = None, code: str = "error") -> WebEventEnvelope:
    return WebEventEnvelope(
        type="error",
        session_id=session_id,
        payload={"code": code, "message": message},
    )


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    return value
