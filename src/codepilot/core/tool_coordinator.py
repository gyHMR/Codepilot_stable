from __future__ import annotations

"""工具调用的准备、执行与事件上报。"""

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
    """创建一个错误状态的工具执行结果。"""
    return AgentToolResult(
        content=[TextContent(text=message)],
        details={},
        is_error=True,
        status="error",
        approved=approved,
    )


def hook_error_tool_result(
    phase: str,
    exc: Exception,
    *,
    approved: bool = True,
) -> AgentToolResult:
    """将工具钩子异常转换为结构化工具错误，避免事件流悬挂。"""
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
    """在保留原工具副作用证据的前提下，将 after hook 异常标记为工具错误。"""
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
    """从工具结果的 details 中提取错误原因字符串。"""
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
    """将工具调用身份信息绑定到结果上，并保持 status/error 字段一致性。"""

    result.tool_call_id = tool_call.id
    result.tool_name = tool_call.name
    if is_error and result.status == "success":
        result.status = "error"
    result.is_error = is_error or result.status != "success"
    return result


@dataclass
class PreparedToolCall:
    """已准备的工具调用：包含原始调用信息、匹配到的工具实例和解析后的参数。"""
    tool_call: ToolCall
    tool: AgentTool
    args: dict[str, Any]


@dataclass
class ExecutedToolCall:
    """已执行的工具调用：包含执行结果和是否出错标志。"""
    result: AgentToolResult
    is_error: bool


class ToolCallCoordinator:
    """工具调用协调器：负责一批工具调用的准备、执行和事件上报（不管理工具权限）。"""

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
        """准备单个工具调用：查找工具、校验权限、执行 before_tool_call 钩子。"""
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
                return None, error_tool_result(
                    before.reason or "Tool execution was blocked",
                    approved=False,
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
            metadata = prepared.tool.metadata
            parallel_safe = bool(
                metadata
                and metadata.concurrency_safe
                and not metadata.exclusive
            )
            if parallel_safe:
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
