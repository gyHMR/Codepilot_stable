from __future__ import annotations

"""Web Console backend application factory.

This is intentionally framework-neutral. A future FastAPI/aiohttp adapter can
wrap WebConsoleBackend without moving runtime logic into the interface layer.
"""

from codepilot.runtime.service import RuntimeService

from .api import WebConsoleBackend, describe_web_contract


def create_web_app(runtime: RuntimeService | None = None) -> WebConsoleBackend:
    return WebConsoleBackend(runtime=runtime)


__all__ = ["WebConsoleBackend", "create_web_app", "describe_web_contract"]
