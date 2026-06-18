from __future__ import annotations

"""Tool call preparation, execution, and event reporting."""

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable

from codepilot.llm.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage

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
    return AgentToolResult(
        content=[TextContent(text=message)],
        details={},
        is_error=True,
        status="error",
        approved=approved,
    )


def tool_error_reason(result: AgentToolResult, is_error: bool) -> str | None:
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
    """Attach call identity and keep status/error fields consistent."""

    result.tool_call_id = tool_call.id
    result.tool_name = tool_call.name
    if is_error and result.status == "success":
        result.status = "error"
    result.is_error = is_error or result.status != "success"
    return result


@dataclass
class PreparedToolCall:
    tool_call: ToolCall
    tool: AgentTool
    args: dict[str, Any]


@dataclass
class ExecutedToolCall:
    result: AgentToolResult
    is_error: bool


class ToolCallCoordinator:
    """Coordinates a batch of tool calls without owning tool permissions."""

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
        result = executed.result
        is_error = executed.is_error or bool(result.is_error)

        if self._config.after_tool_call:
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

        bind_tool_result(prepared.tool_call, result, is_error=is_error)
        is_error = result.is_error
        await self._emit_tool_end(prepared.tool_call, result, is_error)
        return await self._emit_tool_result_message(prepared.tool_call, result, is_error)

    async def _execute_sequential(
        self,
        current_context: AgentContext,
        assistant_message: AssistantMessage,
        tool_calls: list[ToolCall],
        *,
        signal: Any | None,
    ) -> list[ToolResultMessage]:
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
        immediate_results: list[ToolResultMessage] = []
        prepared_calls: list[PreparedToolCall] = []

        for tool_call in tool_calls:
            await self._emit_tool_start(tool_call)
            prepared, immediate, immediate_is_error = await self._prepare(
                current_context,
                assistant_message,
                tool_call,
                signal=signal,
            )
            if prepared is None:
                immediate_results.append(
                    await self._finish_immediate(tool_call, immediate, immediate_is_error)
                )
            else:
                prepared_calls.append(prepared)

        tasks = [
            asyncio.create_task(self._execute_prepared(prepared, signal=signal))
            for prepared in prepared_calls
        ]
        executed_results = await asyncio.gather(*tasks)

        finalized: list[ToolResultMessage] = []
        for prepared, executed in zip(prepared_calls, executed_results):
            finalized.append(
                await self._finalize(
                    current_context,
                    assistant_message,
                    prepared,
                    executed,
                    signal=signal,
                )
            )
        return [*immediate_results, *finalized]

    async def _too_many_tool_calls(
        self,
        tool_calls: list[ToolCall],
    ) -> list[ToolResultMessage]:
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
        bind_tool_result(tool_call, result, is_error=is_error)
        is_error = result.is_error
        await self._emit_tool_end(tool_call, result, is_error)
        return await self._emit_tool_result_message(tool_call, result, is_error)

    async def _emit_tool_start(self, tool_call: ToolCall) -> None:
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
            }
        )

    async def _emit_tool_result_message(
        self,
        tool_call: ToolCall,
        result: AgentToolResult,
        is_error: bool,
    ) -> ToolResultMessage:
        message = ToolResultMessage(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=result.content,
            details=result.details,
            is_error=is_error,
            status=result.status,
            timestamp=now_ms(),
        )
        await self._emitter.emit({"type": "message_start", "message": message})
        await self._emitter.emit({"type": "message_end", "message": message})
        return message
