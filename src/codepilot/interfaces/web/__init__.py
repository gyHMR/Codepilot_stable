"""
Web Console integration boundary.

This package intentionally contains only contracts for now. A future HTTP or
WebSocket server should adapt browser requests to runtime/sessions/tools here
without duplicating Agent logic.
"""

from .api import describe_web_contract
from .schemas import WebEventEnvelope, WebPromptRequest, WebSessionRef, WebToolApproval

__all__ = [
    "WebEventEnvelope",
    "WebPromptRequest",
    "WebSessionRef",
    "WebToolApproval",
    "describe_web_contract",
]
