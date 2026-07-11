from __future__ import annotations

# 新手导读：llm 顶层不再转发协议 DTO 或 provider registry。
# 关注点：core-facing 契约从 llm.ports 导入；provider 装配从 llm.adapter 导入。

"""
Codepilot LLM layer.

This package owns provider integration, model catalog helpers, and the
ModelPort adapter boundary. Import concrete capabilities from their explicit
modules instead of treating this package root as a compatibility facade.
"""

__all__: list[str] = []
