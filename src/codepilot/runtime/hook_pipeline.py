from __future__ import annotations

"""
钩子组合（Hook Composition）模块。

负责将调用方提供的钩子与扩展加载的钩子组合成可执行的管道。

三种钩子类型：
1) before_tool_call: 工具调用前的拦截钩子（可用于权限校验、参数修改）
   - 按顺序执行，任一返回 block=True 即拦截
2) after_tool_call: 工具调用后的处理钩子（可用于结果后处理、日志记录）
   - 按顺序执行，每个钩子可修改结果内容
3) lifecycle_hooks: 生命周期钩子（before_prompt / after_prompt）
   - 简单合并为列表，按顺序执行
"""

import inspect
from typing import Any, Awaitable, Callable

from codepilot.core import (
    AfterToolCallContext,
    AfterToolCallResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
)


# 钩子函数类型别名
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
    """组合 before_tool_call 钩子管道。

    将调用方的 base 钩子和扩展加载的 hooks 合并为一个执行器。
    执行时按顺序调用，任一钩子返回 block=True 即立即拦截（短路）。

    Args:
        base: 调用方提供的基础钩子（可选）。
        hooks: 扩展加载的钩子列表。

    Returns:
        组合后的异步执行器函数；无钩子时返回 None。
    """
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
            # 任一钩子返回 block=True 即短路拦截
            if result and result.block:
                return result
        return None

    return _runner


def compose_after_tool_call(
    base: AfterToolHook | None,
    hooks: list[AfterToolHook],
):
    """组合 after_tool_call 钩子管道。

    将调用方的 base 钩子和扩展加载的 hooks 合并为一个执行器。
    执行时按顺序调用，每个钩子可修改上下文中的结果内容、详情和错误标志。

    Args:
        base: 调用方提供的基础钩子（可选）。
        hooks: 扩展加载的钩子列表。

    Returns:
        组合后的异步执行器函数；无钩子时返回 None。
    """
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
            # 钩子可以覆盖结果的 content、details 和 is_error
            if result.content is not None:
                ctx.result.content = result.content
                final.content = result.content
            if result.details is not None:
                ctx.result.details = result.details
                final.details = result.details
            if result.is_error is not None:
                ctx.is_error = result.is_error
                final.is_error = result.is_error
        # 如果没有任何钩子修改了结果，返回 None
        if final.content is None and final.details is None and final.is_error is None:
            return None
        return final

    return _runner


def compose_lifecycle_hooks(
    base_hooks: list[LifecycleHookFn] | None,
    loaded_hooks: list[LifecycleHookFn],
) -> list[LifecycleHookFn]:
    """组合生命周期钩子列表（before_prompt / after_prompt）。

    简单地将调用方钩子和扩展钩子合并为一个列表，按顺序执行。

    Args:
        base_hooks: 调用方提供的钩子列表（可选）。
        loaded_hooks: 扩展加载的钩子列表。

    Returns:
        合并后的钩子函数列表。
    """
    chain: list[LifecycleHookFn] = []
    chain.extend(base_hooks or [])
    chain.extend(loaded_hooks)
    return chain
