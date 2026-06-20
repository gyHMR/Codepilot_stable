from __future__ import annotations

"""Web Console 后端应用工厂。

本模块刻意保持框架无关性。未来的 FastAPI/aiohttp 适配器可以包装
WebConsoleBackend，而无需将运行时逻辑移入接口层。
"""

from codepilot.runtime.service import RuntimeService

from .api import WebConsoleBackend, describe_web_contract


def create_web_app(runtime: RuntimeService | None = None) -> WebConsoleBackend:
    """创建 Web Console 后端实例（可选传入已有的 RuntimeService）。"""
    return WebConsoleBackend(runtime=runtime)


__all__ = ["WebConsoleBackend", "create_web_app", "describe_web_contract"]
