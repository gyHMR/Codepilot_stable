"""
Web Console integration boundary.

This package intentionally contains only contracts for now. A future HTTP or
WebSocket server should adapt browser requests to runtime/sessions/tools here
without duplicating Agent logic.
"""

from .api import WebConsoleBackend, describe_web_contract, web_route_specs
from .app import create_web_app
from .event_adapter import agent_event_to_web, error_to_web
from .schemas import (
    WebCreateSessionRequest,
    WebEventEnvelope,
    WebPromptRequest,
    WebRouteSpec,
    WebSessionRef,
    WebSessionSummary,
    WebToolApproval,
)
from .websocket import WebSocketSessionStream

__all__ = [
    "WebConsoleBackend",
    "WebCreateSessionRequest",
    "WebEventEnvelope",
    "WebPromptRequest",
    "WebRouteSpec",
    "WebSessionRef",
    "WebSessionSummary",
    "WebToolApproval",
    "WebSocketSessionStream",
    "agent_event_to_web",
    "create_web_app",
    "describe_web_contract",
    "error_to_web",
    "web_route_specs",
]
