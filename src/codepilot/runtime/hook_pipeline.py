from __future__ import annotations

"""Hook composition helpers for runtime assembly."""

import inspect
from typing import Any, Awaitable, Callable

from codepilot.core import (
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
    hooks: list[BeforeToolHook],
):
    chain: list[BeforeToolHook] = []
    if base:
        chain.append(base)
    chain.extend(hooks)
    if not chain:
        return None

    async def _runner(ctx: BeforeToolCallContext, signal: Any | None):
        for hook in chain:
            result = hook(ctx, signal)
            if inspect.isawaitable(result):
                result = await result  # type: ignore[assignment]
            if result and result.block:
                return result
        return None

    return _runner


def compose_after_tool_call(
    base: AfterToolHook | None,
    hooks: list[AfterToolHook],
):
    chain: list[AfterToolHook] = []
    if base:
        chain.append(base)
    chain.extend(hooks)
    if not chain:
        return None

    async def _runner(ctx: AfterToolCallContext, signal: Any | None):
        final = AfterToolCallResult()
        for hook in chain:
            result = hook(ctx, signal)
            if inspect.isawaitable(result):
                result = await result  # type: ignore[assignment]
            if not result:
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

    return _runner


def compose_lifecycle_hooks(
    base_hooks: list[LifecycleHookFn] | None,
    loaded_hooks: list[LifecycleHookFn],
) -> list[LifecycleHookFn]:
    chain: list[LifecycleHookFn] = []
    chain.extend(base_hooks or [])
    chain.extend(loaded_hooks)
    return chain
