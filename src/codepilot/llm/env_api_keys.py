from __future__ import annotations

"""
统一读取环境变量中的 API Key。
"""

import os


def get_env_api_key(provider: str) -> str | None:
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY")
    if provider == "deepseek":
        return os.getenv("DEEPSEEK_API_KEY")
    if provider in {"openai", "openai-compatible"}:
        return os.getenv("OPENAI_API_KEY")
    return None
