from __future__ import annotations

"""
统一读取环境变量中的 API Key。
"""

import os


def get_env_api_key_name(provider: str) -> str | None:
    """返回 provider 对应的标准 API Key 环境变量名。"""

    if provider == "anthropic":
        return "ANTHROPIC_API_KEY"
    if provider == "deepseek":
        return "DEEPSEEK_API_KEY"
    if provider in {"openai", "openai-compatible"}:
        return "OPENAI_API_KEY"
    return None


def get_env_api_key(provider: str) -> str | None:
    env_name = get_env_api_key_name(provider)
    return os.getenv(env_name) if env_name else None
