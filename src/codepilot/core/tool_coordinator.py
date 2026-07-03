from __future__ import annotations

# 新手导读：ToolCallCoordinator 负责执行模型返回的一批工具调用，并处理 before/after hook。
# 关注点：重点看它如何区分并行安全工具和独占工具，以及如何把结果转成 ToolResultMessage。

"""
工具调用协调器模块。

本模块负责一批工具调用的完整生命周期管理：准备、执行和事件上报。

核心类：
    ToolCallCoordinator: 工具调用协调器，处理一批工具调用的编排

执行流程：
    1. 准备阶段（_prepare）：
       - 检查工具是否在当前上下文中可见
       - 检查非托管工具是否被允许
       - 执行 before_tool_call 钩子（可拦截）
    2. 执行阶段（_execute_prepared）：
       - 调用工具的 execute 方法
       - 处理工具执行过程中的更新事件
    3. 完成阶段（_finalize）：
       - 执行 after_tool_call 钩子（可修改结果）
       - 绑定工具调用身份信息
       - 发射工具执行结束事件
       - 构建 ToolResultMessage

执行模式：
    - 串行模式（sequential）：逐个执行工具调用
    - 并行模式（parallel）：安全的工具放入批次并行执行，不安全的串行执行

辅助函数：
    - error_tool_result: 创建错误状态的工具结果
    - denied_tool_result: 创建策略拒绝状态的工具结果
    - hook_error_tool_result: 将钩子异常转换为结构化工具错误
    - bind_tool_result: 将工具调用身份信息绑定到结果上
    - can_schedule_tool_in_parallel: 判断工具是否可以并行执行
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable

from codepilot.protocols import AssistantMessage, TextContent, ToolCall, ToolResultMessage

from .events import AgentEventEmitter, maybe_await, now_ms
from .types import (
    AfterToolCallContext,
    AgentContext,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
    BeforeToolCallContext,
)


def error_tool_result(message: str, *, approved: bool = True) -> AgentToolResult:
    """创建一个错误状态的工具执行结果。

    用于工具未找到、参数错误等场景，工具实际上并未执行。

    Args:
        message: 错误描述信息。
        approved: 是否已审批（默认 True，因为这是系统生成的错误）。

    Returns:
        AgentToolResult: 错误状态的工具结果。
    """
    return AgentToolResult(
        content=[TextContent(text=message)],
        details={},
        is_error=True,
        status="error",
        approved=approved,
    )


def denied_tool_result(
    message: str,
    *,
    reason: str,
) -> AgentToolResult:
    """创建策略拒绝状态的工具结果。

    denied 表示工具没有执行，因为 core/runtime 策略不允许；
    它不同于工具已经执行但失败的 error。

    Args:
        message: 拒绝描述信息。
        reason: 拒绝原因代码（如 "before_tool_call_blocked"）。

    Returns:
        AgentToolResult: 拒绝状态的工具结果。
    """
    return AgentToolResult(
        content=[TextContent(text=message)],
        details={
            "reason": reason,
            "status": "denied",
            "message": message,
        },
        is_error=True,
        status="denied",
        approved=False,
        error_code=reason,
    )


def hook_error_tool_result(
    phase: str,
    exc: Exception,
    *,
    approved: bool = True,
) -> AgentToolResult:
    """将工具钩子异常转换为结构化工具错误，避免事件流悬挂。

    当 before_tool_call 或 after_tool_call 钩子抛出异常时，
    使用此函数将异常转换为结构化的工具错误结果。

    Args:
        phase: 钩子阶段（"before" 或 "after"）。
        exc: 捕获的异常。
        approved: 是否已审批。

    Returns:
        AgentToolResult: 错误状态的工具结果。
    """
    error_code = f"{phase}_tool_hook_error"
    return AgentToolResult(
        content=[TextContent(text=f"{phase}_tool_call hook failed: {exc}")],
        details={
            "reason": error_code,
            "status": "error",
            "error_kind": type(exc).__name__,
        },
        is_error=True,
        status="error",
        approved=approved,
        error_code=error_code,
    )


def mark_hook_error_result(result: AgentToolResult, phase: str, exc: Exception) -> AgentToolResult:
    """在保留原工具副作用证据的前提下，将 after hook 异常标记为工具错误。

    与 hook_error_tool_result 不同，此函数修改已有的工具结果，
    保留原始的 details 信息（作为 original_details），便于调试。

    Args:
        result: 原始工具执行结果。
        phase: 钩子阶段（通常是 "after"）。
        exc: 捕获的异常。

    Returns:
        AgentToolResult: 修改后的工具结果。
    """
    error_code = f"{phase}_tool_hook_error"
    original_details = result.details
    result.content = [TextContent(text=f"{phase}_tool_call hook failed: {exc}")]
    result.details = {
        "reason": error_code,
        "status": "error",
        "error_kind": type(exc).__name__,
        "original_details": original_details,
    }
    result.is_error = True
    result.status = "error"
    result.error_code = error_code
    return result


def tool_error_reason(result: AgentToolResult, is_error: bool) -> str | None:
    """从工具结果的 details 中提取错误原因字符串。

    Args:
        result: 工具执行结果。
        is_error: 是否出错。

    Returns:
        str | None: 错误原因字符串，无错误时返回 None。
    """
    if not is_error:
        return None
    if isinstance(result.details, dict):
        reason = result.details.get("reason")
        if isinstance(reason, str) and reason:
            return reason
    return None


def bind_tool_result(
    tool_call: ToolCall,
    result: AgentToolResult,
    *,
    is_error: bool,
) -> AgentToolResult:
    """将工具调用身份信息绑定到结果上，并保持 status/error 字段一致性。

    此函数确保工具结果包含正确的 tool_call_id 和 tool_name，
    并根据 is_error 标志调整 status 和 is_error 字段。

    Args:
        tool_call: 工具调用信息。
        result: 工具执行结果。
        is_error: 是否出错。

    Returns:
        AgentToolResult: 绑定后的工具结果。
    """
    result.tool_call_id = tool_call.id
    result.tool_name = tool_call.name
    # 如果标记为错误但 status 仍为 success，修正为 error
    if is_error and result.status == "success":
        result.status = "error"
    # 确保 is_error 与 status 一致
    result.is_error = is_error or result.status != "success"
    return result


def can_schedule_tool_in_parallel(tool: AgentTool) -> bool:
    """判断工具是否可以进入 core 的并行调度批次。

    采用保守规则：只有工具元数据显式声明 concurrency_safe，
    且不要求 exclusive 独占执行时，才允许和同批次工具并行。
    没有元数据的工具一律串行，避免把未知副作用误判为安全。

    Args:
        tool: 工具实例。

    Returns:
        bool: 如果工具可以并行执行则返回 True。
    """
    metadata = tool.metadata
    return bool(metadata and metadata.concurrency_safe and not metadata.exclusive)


@dataclass
class PreparedToolCall:
    """已准备的工具调用：包含原始调用信息、匹配到的工具实例和解析后的参数。

    在准备阶段创建，包含执行所需的所有信息。

    Attributes:
        tool_call: 原始的工具调用信息（来自 LLM 响应）。
        tool: 匹配到的 AgentTool 实例（包含 execute 函数）。
        args: 解析后的参数字典。
    """
    tool_call: ToolCall                   # 原始工具调用信息
    tool: AgentTool                       # 匹配到的工具实例
    args: dict[str, Any]                  # 解析后的参数


@dataclass
class ExecutedToolCall:
    """已执行的工具调用：包含执行结果和是否出错标志。

    在执行阶段创建，传递给完成阶段进行后处理。

    Attributes:
        result: 工具执行结果。
        is_error: 执行是否出错。
    """
    result: AgentToolResult               # 执行结果
    is_error: bool                        # 是否出错


class ToolCallCoordinator:
    """工具调用协调器：负责一批工具调用的准备、执行和事件上报。

    这一层只处理 agent loop 视角的编排边界：
    - 当前上下文中是否存在该工具
    - unmanaged 工具是否允许进入流程
    - before/after hook 的短路或后处理

    真正的权限策略、审批、参数语义和路径安全分别由 ToolRuntime 与具体工具负责。

    执行模式：
        - 串行模式（sequential）：逐个准备、执行、完成
        - 并行模式（parallel）：安全的工具放入批次并行执行，不安全的串行执行

    使用方式：
        coordinator = ToolCallCoordinator(config=config, emitter=emitter)
        results = await coordinator.execute_batch(context, assistant_message)
    """

    def __init__(self, *, config: AgentLoopConfig, emitter: AgentEventEmitter) -> None:
        self._config = config
        self._emitter = emitter

    async def execute_batch(
        self,
        current_context: AgentContext,
        assistant_message: AssistantMessage,
        *,
        signal: Any | None = None,
    ) -> list[ToolResultMessage]:
        """执行一批工具调用：根据配置选择并行或串行执行模式。"""
        tool_calls = [
            content for content in assistant_message.content if isinstance(content, ToolCall)
        ]
        if (
            self._config.max_tool_calls_per_turn is not None
            and len(tool_calls) > self._config.max_tool_calls_per_turn
        ):
            return await self._too_many_tool_calls(tool_calls)

        if self._config.tool_execution == "sequential":
            return await self._execute_sequential(
                current_context,
                assistant_message,
                tool_calls,
                signal=signal,
            )
        return await self._execute_parallel(
            current_context,
            assistant_message,
            tool_calls,
            signal=signal,
        )

    async def _prepare(
        self,
        current_context: AgentContext,
        assistant_message: AssistantMessage,
        tool_call: ToolCall,
        *,
        signal: Any | None,
    ) -> tuple[PreparedToolCall | None, AgentToolResult, bool]:
        """准备单个工具调用。

        这里做的是“调度前检查”：
        - 工具是否在当前 AgentContext 中可见；
        - 非 ToolRuntime 托管的工具是否被允许；
        - before_tool_call 钩子是否要求拦截。

        不在这里做具体参数语义、路径或 shell 安全判断；这些必须留在
        ToolRuntime 和工具自身 execute 中作为可独立生效的运行时防线。
        """
        tool = next((t for t in current_context.tools if t.name == tool_call.name), None)
        if tool is None:
            return None, error_tool_result(
                f"Tool {tool_call.name} not found",
                approved=False,
            ), True
        if not tool.runtime_managed and not self._config.allow_unmanaged_tools:
            result = error_tool_result(
                f"Tool {tool_call.name} is not managed by ToolRuntime",
                approved=False,
            )
            result.status = "denied"
            result.details = {
                "tool": tool_call.name,
                "reason": "unmanaged_tool",
                "status": "denied",
            }
            return None, result, True

        args = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
        if self._config.before_tool_call:
            try:
                before = await maybe_await(
                    self._config.before_tool_call(
                        BeforeToolCallContext(
                            assistant_message=assistant_message,
                            tool_call=tool_call,
                            args=args,
                            context=current_context,
                        ),
                        signal,
                    )
                )
            except Exception as exc:
                return None, hook_error_tool_result(
                    "before",
                    exc,
                    approved=False,
                ), True
            if before and before.block:
                return None, denied_tool_result(
                    before.reason or "Tool execution was blocked",
                    reason="before_tool_call_blocked",
                ), True

        return PreparedToolCall(tool_call=tool_call, tool=tool, args=args), AgentToolResult(content=[]), False

    async def _execute_prepared(
        self,
        prepared: PreparedToolCall,
        *,
        signal: Any | None,
    ) -> ExecutedToolCall:
        """执行已准备的工具调用：调用工具的 execute 方法并处理更新事件。"""
        try:
            updates: list[Awaitable[Any] | Any] = []

            def on_update(partial_result: AgentToolResult) -> None:
                updates.append(
                    self._emitter.emit(
                        {
                            "type": "tool_execution_update",
                            "toolCallId": prepared.tool_call.id,
                            "toolName": prepared.tool_call.name,
                            "args": prepared.tool_call.arguments,
                            "partialResult": partial_result,
                        }
                    )
                )

            raw_result = prepared.tool.execute(
                prepared.tool_call.id,
                prepared.args,
                signal,
                on_update,
            )
            result = await maybe_await(raw_result)

            for update in updates:
                await maybe_await(update)
            return ExecutedToolCall(result=result, is_error=bool(result.is_error))
        except Exception as exc:
            return ExecutedToolCall(result=error_tool_result(str(exc)), is_error=True)

    async def _finalize(
        self,
        current_context: AgentContext,
        assistant_message: AssistantMessage,
        prepared: PreparedToolCall,
        executed: ExecutedToolCall,
        *,
        signal: Any | None,
    ) -> ToolResultMessage:
        """完成工具调用：执行 after_tool_call 钩子，绑定结果身份，发射事件。"""
        result = executed.result
        is_error = executed.is_error or bool(result.is_error)

        if self._config.after_tool_call:
            try:
                after = await maybe_await(
                    self._config.after_tool_call(
                        AfterToolCallContext(
                            assistant_message=assistant_message,
                            tool_call=prepared.tool_call,
                            args=prepared.args,
                            result=result,
                            is_error=is_error,
                            context=current_context,
                        ),
                        signal,
                    )
                )
                if after:
                    if after.content is not None:
                        result.content = after.content
                    if after.details is not None:
                        result.details = after.details
                    if after.is_error is not None:
                        is_error = after.is_error
            except Exception as exc:
                result = mark_hook_error_result(result, "after", exc)
                is_error = True

        bind_tool_result(prepared.tool_call, result, is_error=is_error)
        is_error = result.is_error
        await self._emit_tool_end(prepared.tool_call, result, is_error)
        return await self._emit_tool_result_message(prepared.tool_call, result, is_error)

    # ── 执行策略 ────────────────────────────────────────────────

    async def _execute_sequential(
        self,
        current_context: AgentContext,
        assistant_message: AssistantMessage,
        tool_calls: list[ToolCall],
        *,
        signal: Any | None,
    ) -> list[ToolResultMessage]:
        """串行执行工具调用：逐个准备、执行、完成。"""
        results: list[ToolResultMessage] = []
        for tool_call in tool_calls:
            await self._emit_tool_start(tool_call)
            prepared, immediate, immediate_is_error = await self._prepare(
                current_context,
                assistant_message,
                tool_call,
                signal=signal,
            )
            if prepared is None:
                results.append(
                    await self._finish_immediate(tool_call, immediate, immediate_is_error)
                )
                continue

            executed = await self._execute_prepared(prepared, signal=signal)
            results.append(
                await self._finalize(
                    current_context,
                    assistant_message,
                    prepared,
                    executed,
                    signal=signal,
                )
            )
        return results

    async def _execute_parallel(
        self,
        current_context: AgentContext,
        assistant_message: AssistantMessage,
        tool_calls: list[ToolCall],
        *,
        signal: Any | None,
    ) -> list[ToolResultMessage]:
        """并行执行工具调用：安全的工具放入批次并行执行，不安全的串行执行。"""
        results: dict[int, ToolResultMessage] = {}
        parallel_batch: list[tuple[int, PreparedToolCall]] = []

        async def flush_parallel_batch() -> None:
            """刷新并行批次：将积累的可并行工具一次性用 asyncio.gather 执行。"""
            if not parallel_batch:
                return
            batch = list(parallel_batch)
            parallel_batch.clear()
            executed_results = await asyncio.gather(
                *[
                    self._execute_prepared(prepared, signal=signal)
                    for _, prepared in batch
                ]
            )
            for (index, prepared), executed in zip(batch, executed_results):
                results[index] = await self._finalize(
                    current_context,
                    assistant_message,
                    prepared,
                    executed,
                    signal=signal,
                )

        for index, tool_call in enumerate(tool_calls):
            await self._emit_tool_start(tool_call)
            prepared, immediate, immediate_is_error = await self._prepare(
                current_context,
                assistant_message,
                tool_call,
                signal=signal,
            )
            if prepared is None:
                await flush_parallel_batch()
                results[index] = await self._finish_immediate(
                    tool_call,
                    immediate,
                    immediate_is_error,
                )
                continue
            if can_schedule_tool_in_parallel(prepared.tool):
                parallel_batch.append((index, prepared))
                continue
            await flush_parallel_batch()
            executed = await self._execute_prepared(prepared, signal=signal)
            results[index] = await self._finalize(
                    current_context,
                    assistant_message,
                    prepared,
                    executed,
                    signal=signal,
            )
        await flush_parallel_batch()
        return [results[index] for index in range(len(tool_calls))]

    # ── 辅助方法 ────────────────────────────────────────────────

    async def _too_many_tool_calls(
        self,
        tool_calls: list[ToolCall],
    ) -> list[ToolResultMessage]:
        """处理单轮工具调用数量超限的情况：为每个调用生成错误结果。"""
        results: list[ToolResultMessage] = []
        message = (
            f"Too many tool calls in one turn: {len(tool_calls)} "
            f"(max={self._config.max_tool_calls_per_turn})"
        )
        for tool_call in tool_calls:
            await self._emit_tool_start(tool_call)
            result = error_tool_result(message, approved=False)
            result.status = "error"
            result.details = {
                "reason": "max_tool_calls_per_turn",
                "status": "error",
                "max_tool_calls_per_turn": self._config.max_tool_calls_per_turn,
            }
            results.append(await self._finish_immediate(tool_call, result, True))
        return results

    async def _finish_immediate(
        self,
        tool_call: ToolCall,
        result: AgentToolResult,
        is_error: bool,
    ) -> ToolResultMessage:
        """立即完成工具调用（用于准备阶段就失败的情况）：绑定结果并发射事件。"""
        bind_tool_result(tool_call, result, is_error=is_error)
        is_error = result.is_error
        await self._emit_tool_end(tool_call, result, is_error)
        return await self._emit_tool_result_message(tool_call, result, is_error)

    # ── 事件发射 ────────────────────────────────────────────────

    async def _emit_tool_start(self, tool_call: ToolCall) -> None:
        """发射工具执行开始事件。"""
        await self._emitter.emit(
            {
                "type": "tool_execution_start",
                "toolCallId": tool_call.id,
                "toolName": tool_call.name,
                "args": tool_call.arguments,
            }
        )

    async def _emit_tool_end(
        self,
        tool_call: ToolCall,
        result: AgentToolResult,
        is_error: bool,
    ) -> None:
        """发射工具执行结束事件：包含结果、状态、权限、耗时等完整信息。"""
        await self._emitter.emit(
            {
                "type": "tool_execution_end",
                "toolCallId": tool_call.id,
                "toolName": tool_call.name,
                "result": result,
                "status": result.status,
                "isError": is_error,
                "approved": result.approved,
                "approvalId": result.approval_id,
                "errorReason": tool_error_reason(result, is_error),
                "permission": (
                    result.metadata.get("permission_decision")
                    if isinstance(result.metadata, dict)
                    else None
                ),
                "durationMs": (
                    result.metadata.get("duration_ms")
                    if isinstance(result.metadata, dict)
                    else None
                ),
                "affectedPaths": list(result.affected_paths),
                "workspaceChanged": result.workspace_changed,
                "outputTruncated": bool(
                    isinstance(result.metadata, dict)
                    and (
                        result.metadata.get("truncated")
                        or result.metadata.get("stdout_truncated")
                        or result.metadata.get("stderr_truncated")
                    )
                ),
            }
        )

    async def _emit_tool_result_message(
        self,
        tool_call: ToolCall,
        result: AgentToolResult,
        is_error: bool,
    ) -> ToolResultMessage:
        """构建 ToolResultMessage 并发射 message_start/end 事件。"""
        metadata = dict(result.metadata)
        target_path = (
            tool_call.arguments.get("path")
            if isinstance(tool_call.arguments, dict)
            else None
        )
        if isinstance(target_path, str) and target_path:
            metadata.setdefault("tool_target_path", target_path)
        message = ToolResultMessage(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=result.content,
            details=result.details,
            is_error=is_error,
            status=result.status,
            approved=result.approved,
            approval_id=result.approval_id,
            error_code=result.error_code,
            exit_code=result.exit_code,
            affected_paths=list(result.affected_paths),
            workspace_changed=result.workspace_changed,
            diff_summary=result.diff_summary,
            verification=dict(result.verification) if result.verification else None,
            timestamp=now_ms(),
            metadata=metadata,
        )
        await self._emitter.emit({"type": "message_start", "message": message})
        await self._emitter.emit({"type": "message_end", "message": message})
        return message
