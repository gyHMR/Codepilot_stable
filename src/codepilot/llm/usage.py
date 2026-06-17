from __future__ import annotations

from .types import Usage


def normalize_usage(usage: Usage) -> Usage:
    """Ensure derived token fields are populated consistently."""

    if usage.total_tokens <= 0:
        usage.total_tokens = usage.input + usage.output + usage.cache_read + usage.cache_write
    if usage.cost.total <= 0:
        usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
    return usage


__all__ = ["normalize_usage"]
