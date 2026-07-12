from __future__ import annotations

"""Lifecycle hook composition used while opening a runtime session."""

from typing import Any, Awaitable, Callable


LifecycleHookFn = Callable[[Any], None | Awaitable[None]]


def compose_lifecycle_hooks(
    base: list[LifecycleHookFn] | None,
    loaded: list[LifecycleHookFn],
) -> list[LifecycleHookFn]:
    return [*(base or []), *loaded]


__all__ = ["LifecycleHookFn", "compose_lifecycle_hooks"]
