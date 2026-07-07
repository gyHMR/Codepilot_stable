"""Runtime boundary for interface actions and agent execution."""

from .gateway import RuntimeGateway
from .opening import SessionOpenIntent

__all__ = [
    "RuntimeGateway",
    "SessionOpenIntent",
]
