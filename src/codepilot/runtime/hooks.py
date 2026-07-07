from __future__ import annotations

"""Small hook composition helpers used while opening a runtime session."""

import inspect
from typing import Any, Awaitable, Callable

from codepilot.protocols.commands import (
    AfterToolCallContext,
    AfterToolCallResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
)


BeforeToolHook = Callable[
    [BeforeToolCallContext, Any | None],
    BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None],
]
AfterToolHook = Callable[
    [AfterToolCallContext, Any | None],
    AfterToolCallResult | None | Awaitable[AfterToolCallResult | None],
]
LifecycleHookFn = Callable[[Any], None | Awaitable[None]]


def compose_before_tool_call(
    base: BeforeToolHook | None,
    loaded: list[BeforeToolHook],
):
    chain = [hook for hook in [base, *loaded] if hook is not None]
    if not chain:
        return None

    async def run(ctx: BeforeToolCallContext, signal: Any | None):
        for hook in chain:
            result = hook(ctx, signal)
            if inspect.isawaitable(result):
                result = await result
            if result and result.block:
                return result
        return None

    return run


def compose_after_tool_call(
    base: AfterToolHook | None,
    loaded: list[AfterToolHook],
):
    chain = [hook for hook in [base, *loaded] if hook is not None]
    if not chain:
        return None

    async def run(ctx: AfterToolCallContext, signal: Any | None):
        final = AfterToolCallResult()
        for hook in chain:
            result = hook(ctx, signal)
            if inspect.isawaitable(result):
                result = await result
            if result is None:
                continue
            if result.content is not None:
                ctx.result.content = result.content
                final.content = result.content
            if result.details is not None:
                ctx.result.details = result.details
                final.details = result.details
            if result.is_error is not None:
                ctx.is_error = result.is_error
                final.is_error = result.is_error
        if final.content is None and final.details is None and final.is_error is None:
            return None
        return final

    return run


def compose_lifecycle_hooks(
    base: list[LifecycleHookFn] | None,
    loaded: list[LifecycleHookFn],
) -> list[LifecycleHookFn]:
    return [*(base or []), *loaded]


__all__ = [
    "AfterToolHook",
    "BeforeToolHook",
    "LifecycleHookFn",
    "compose_after_tool_call",
    "compose_before_tool_call",
    "compose_lifecycle_hooks",
]
