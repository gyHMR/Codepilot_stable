"""Web Console 集成边界层。

本包目前只包含契约定义和骨架实现。未来的 HTTP/WebSocket 服务器应在此处
将浏览器请求适配到 runtime/sessions/tools，而不重复 Agent 核心逻辑。
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
