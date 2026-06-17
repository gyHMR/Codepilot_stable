from __future__ import annotations

"""
Agent 主循环实现：
用户消息 -> LLM -> 工具调用 -> LLM -> ... -> 结束

核心流程是一个多轮循环：
1. 将用户消息发送给 LLM，获取助手回复。
2. 如果助手回复中包含工具调用请求，则执行工具并将结果反馈给 LLM。
3. 重复步骤 2，直到助手不再请求工具调用，或遇到错误/中止。
"""

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, cast

from codepilot.llm.stream import stream_simple
from codepilot.llm.types import (
    AssistantMessage,
    Context,
    ImageContent,
    Message,
    SimpleStreamOptions,
    TextContent,
    ToolCall,
    ToolResultMessage,
)

from .types import (
    AfterToolCallContext,
    AgentContext,
    AgentEvent,
    AgentEventSink,
    AgentLoopConfig,
    AgentMessage,
    AgentTool,
    AgentToolResult,
    BeforeToolCallContext,
)


# ── 类型别名与工具函数 ──────────────────────────────────────────

# 流式调用函数的类型签名：接收模型、上下文、选项，返回流式响应
StreamFn = Callable[[Any, Context, SimpleStreamOptions | None], Any | Awaitable[Any]]


def _now_ms() -> int:
    """获取当前时间的毫秒时间戳。"""
    return int(time.time() * 1000)


def _error_tool_result(message: str, *, approved: bool = True) -> AgentToolResult:
    """构造一个表示错误的工具执行结果。"""
    return AgentToolResult(
        content=[TextContent(text=message)],
        details={},
        is_error=True,
        approved=approved,
    )


def _tool_error_reason(result: AgentToolResult, is_error: bool) -> str | None:
    """从工具结果中提取可展示/记录的错误原因。"""
    if not is_error:
        return None
    if isinstance(result.details, dict):
        reason = result.details.get("reason")
        if isinstance(reason, str) and reason:
            return reason
    return None


async def _maybe_await(value: Any) -> Any:
    """兼容处理同步/异步返回值：如果是可等待对象则 await，否则直接返回。"""
    if asyncio.isfuture(value) or asyncio.iscoroutine(value):
        return await value
    return value


async def _emit(emit: AgentEventSink, event: dict[str, Any]) -> None:
    """发送一个事件到事件接收器（兼容同步/异步 sink）。"""
    await _maybe_await(emit(cast(AgentEvent, event)))


def _with_event_schema(
    emit: AgentEventSink,
    session_id: str | None,
) -> AgentEventSink:
    """为事件添加统一的元数据字段。

    为每个事件附加以下字段：
    - runId: 本次运行的唯一 ID（格式 run_<12位hex>）
    - turnId: 当前轮次编号（从 1 开始，每轮 turn_start 后 +1）
    - eventId: 本次运行内的递增事件 ID（格式 runId:seq）
    - timestamp: 事件的毫秒时间戳
    - sessionId: 透传上层会话 ID

    Args:
        emit: 原始事件接收器。
        session_id: 上层会话 ID，直接透传到每个事件中。

    Returns:
        包装后的事件接收器，自动为事件注入元数据。
    """
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    turn_id = 0
    event_seq = 0

    async def _wrapped(event: dict[str, Any]) -> None:
        nonlocal turn_id, event_seq
        event_type = event.get("type")
        # 每次 turn_start 时轮次号 +1
        if event_type == "turn_start":
            turn_id += 1

        event_seq += 1
        enriched = {
            **event,
            "runId": run_id,
            "turnId": turn_id,
            "eventId": f"{run_id}:{event_seq}",
            "timestamp": _now_ms(),
            "sessionId": session_id,
        }
        await _maybe_await(emit(cast(AgentEvent, enriched)))

    return _wrapped


# ── 主循环入口 ──────────────────────────────────────────────────

async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    """运行一次全新的 Agent 对话循环。

    流程：将用户提示词追加到上下文，然后进入 LLM <-> 工具 的多轮循环。

    Args:
        prompts: 用户发送的提示消息列表。
        context: 当前的 Agent 上下文（系统提示词、历史消息、工具列表）。
        config: 循环配置（模型、钩子函数、执行模式等）。
        emit: 事件接收器，用于通知外部运行过程中的各类事件。
        signal: 可选的中断信号，传递给工具执行。
        stream_fn: 可选的自定义流式调用函数，默认使用 stream_simple。

    Returns:
        本次运行产生的新 AgentMessage 列表（不含历史消息）。
    """
    # 为事件注入 runId、turnId 等元数据
    emit = _with_event_schema(emit, config.session_id)
    new_messages: list[AgentMessage] = list(prompts)
    # 构建当前上下文：将用户提示追加到历史消息之后
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=[*context.messages, *prompts],
        tools=context.tools,
    )

    # 发送运行开始事件，并为每条用户消息触发 message_start/end 事件
    await _emit(emit, {"type": "agent_start"})
    await _emit(emit, {"type": "turn_start"})
    for prompt in prompts:
        await _emit(emit, {"type": "message_start", "message": prompt})
        await _emit(emit, {"type": "message_end", "message": prompt})

    # 进入核心循环
    await _run_loop(current_context, new_messages, config, emit, signal, stream_fn)
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    """继续上一次未完成的 Agent 运行。

    典型场景：上一轮 LLM 返回了工具调用请求，工具尚未执行完毕，
    调用此方法继续执行工具并将结果反馈给 LLM。

    Args:
        context: 当前的 Agent 上下文（应包含上一轮的完整消息历史）。
        config: 循环配置。
        emit: 事件接收器。
        signal: 可选的中断信号。
        stream_fn: 可选的自定义流式调用函数。

    Returns:
        继续运行产生的新 AgentMessage 列表。

    Raises:
        ValueError: 当上下文为空或最后一条消息是助手消息时抛出
                    （助手消息后面必须跟用户/工具消息才能继续）。
    """
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    if isinstance(context.messages[-1], AssistantMessage):
        raise ValueError("Cannot continue from message role: assistant")

    emit = _with_event_schema(emit, config.session_id)
    new_messages: list[AgentMessage] = []
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages),
        tools=context.tools,
    )

    await _emit(emit, {"type": "agent_start"})
    await _emit(emit, {"type": "turn_start"})

    await _run_loop(current_context, new_messages, config, emit, signal, stream_fn)
    return new_messages


# ── 核心循环逻辑 ────────────────────────────────────────────────

async def _run_loop(
    current_context: AgentContext,
    new_messages: list[AgentMessage],
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None,
    stream_fn: StreamFn | None,
) -> None:
    """Agent 的核心多轮循环。

    循环结构（由内到外）：
    ┌─────────────────────────────────────────────────────┐
    │  外层循环：检查 follow_up 消息                       │
    │  ┌───────────────────────────────────────────────┐  │
    │  │  内层循环：LLM 响应 -> 工具调用 -> LLM 响应    │  │
    │  │  - 获取 steering 消息（中途引导）              │  │
    │  │  - 调用 LLM 获取助手回复                       │  │
    │  │  - 如果有工具调用：执行工具 -> 追加结果 -> 继续 │  │
    │  │  - 如果无工具调用：退出内层循环                │  │
    │  └───────────────────────────────────────────────┘  │
    │  检查 follow_up 消息队列，有则继续外层循环           │
    └─────────────────────────────────────────────────────┘

    Args:
        current_context: 当前上下文（会被原地修改，追加新消息）。
        new_messages: 本次运行产生的新消息列表（会被原地修改）。
        config: 循环配置。
        emit: 事件接收器。
        signal: 中断信号。
        stream_fn: 自定义流式调用函数。
    """
    first_turn = True
    # 预取引导消息（如果有），用于在第一轮开始前注入
    pending_messages = await _maybe_await(config.get_steering_messages()) if config.get_steering_messages else []

    while True:
        has_more_tool_calls = True

        # 内层循环：持续处理 LLM 响应和工具调用，直到没有更多工具调用且无待处理消息
        while has_more_tool_calls or pending_messages:
            if not first_turn:
                await _emit(emit, {"type": "turn_start"})
            else:
                first_turn = False

            # 如果有待处理的引导/后续消息，先注入到上下文中
            if pending_messages:
                for message in pending_messages:
                    await _emit(emit, {"type": "message_start", "message": message})
                    await _emit(emit, {"type": "message_end", "message": message})
                    current_context.messages.append(message)
                    new_messages.append(message)
                pending_messages = []

            # 调用 LLM 获取助手回复（流式）
            assistant = await _stream_assistant_response(current_context, config, emit, signal, stream_fn)
            new_messages.append(assistant)

            # 如果遇到错误或中止，立即结束整个循环
            if assistant.stop_reason in {"error", "aborted"}: 
                await _emit(emit, {"type": "turn_end", "message": assistant, "toolResults": []})
                await _emit(emit, {"type": "agent_end", "messages": new_messages})
                return

            # 提取助手回复中的工具调用请求
            tool_calls = [c for c in assistant.content if isinstance(c, ToolCall)]
            has_more_tool_calls = len(tool_calls) > 0
            tool_results: list[ToolResultMessage] = []

            # 如果有工具调用，执行并将结果追加到上下文
            if has_more_tool_calls:
                tool_results = await _execute_tool_calls(current_context, assistant, config, emit, signal)
                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            await _emit(emit, {"type": "turn_end", "message": assistant, "toolResults": tool_results})
            # 每轮结束后重新检查是否有新的引导消息
            pending_messages = await _maybe_await(config.get_steering_messages()) if config.get_steering_messages else []

        # 内层循环结束，检查是否有后续消息需要处理
        followups = await _maybe_await(config.get_follow_up_messages()) if config.get_follow_up_messages else []
        if followups:
            pending_messages = followups
            continue  # 有后续消息，继续外层循环
        break  # 无后续消息，退出整个循环

    await _emit(emit, {"type": "agent_end", "messages": new_messages})


# ── LLM 流式调用 ───────────────────────────────────────────────

async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None,
    stream_fn: StreamFn | None,
) -> AssistantMessage:
    """调用 LLM 并以流式方式接收助手回复。

    流程：
    1. 可选地对消息上下文进行变换（transform_context）。
    2. 将内部消息格式转换为 LLM 消息格式（convert_to_llm）。
    3. 解析 API Key（如果配置了动态获取）。
    4. 发起流式请求，逐块接收并分发事件。
    5. 返回最终的 AssistantMessage。

    Args:
        context: 当前 Agent 上下文。
        config: 循环配置。
        emit: 事件接收器。
        signal: 中断信号。
        stream_fn: 自定义流式调用函数。

    Returns:
        LLM 生成的最终助手消息。
    """
    messages = context.messages
    # 可选的上下文变换（如裁剪过长消息、脱敏等）
    if config.transform_context:
        messages = await _maybe_await(config.transform_context(messages, signal))

    # 将内部 AgentMessage 转换为 LLM 可接受的 Message 格式
    llm_messages = await _maybe_await(config.convert_to_llm(messages))
    llm_context = Context(
        system_prompt=context.system_prompt,
        messages=llm_messages,
        tools=context.tools,  # AgentTool 与 ai.Tool 字段兼容
    )

    # 动态解析 API Key（按 provider 名称获取）
    resolved_api_key = config.get_api_key and await _maybe_await(config.get_api_key(config.model.provider))
    options = SimpleStreamOptions(reasoning=config.reasoning, api_key=resolved_api_key, session_id=config.session_id)

    # 发起流式请求
    fn = stream_fn or stream_simple
    response_stream = await _maybe_await(fn(config.model, llm_context, options))

    partial: AssistantMessage | None = None
    added_partial = False

    # 逐块处理流式响应事件
    async for event in response_stream:
        t = event.get("type")
        if t == "start":
            # 流式开始：创建初始的 partial 消息并追加到上下文
            partial = event["partial"]
            context.messages.append(partial)
            added_partial = True
            await _emit(emit, {"type": "message_start", "message": partial})
        elif t in {
            "text_start", "text_delta", "text_end",
            "thinking_start", "thinking_delta", "thinking_end",
            "toolcall_start", "toolcall_delta", "toolcall_end",
        }:
            # 流式更新：文本/思考/工具调用的增量内容，更新上下文中的 partial 消息
            if partial is not None:
                partial = event["partial"]
                context.messages[-1] = partial
                await _emit(emit, {"type": "message_update", "message": partial, "assistantMessageEvent": event})
        elif t in {"done", "error"}:
            # 流式结束：获取最终完整消息，替换上下文中的 partial
            final_message = await response_stream.result()
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
                await _emit(emit, {"type": "message_start", "message": final_message})
            await _emit(emit, {"type": "message_end", "message": final_message})
            return final_message

    # 兜底处理：流式迭代器正常退出但未触发 done/error 事件
    final_message = await response_stream.result()
    if added_partial:
        context.messages[-1] = final_message
    else:
        context.messages.append(final_message)
        await _emit(emit, {"type": "message_start", "message": final_message})
    await _emit(emit, {"type": "message_end", "message": final_message})
    return final_message


# ── 工具调用执行 ────────────────────────────────────────────────

@dataclass
class _PreparedToolCall:
    """已准备好的工具调用，包含工具定义和解析后的参数。"""
    tool_call: ToolCall          # 原始的工具调用请求
    tool: AgentTool              # 匹配到的工具定义
    args: dict[str, Any]         # 解析后的参数字典


@dataclass
class _ExecutedToolCall:
    """工具调用的执行结果。"""
    result: AgentToolResult      # 工具返回的结果
    is_error: bool               # 是否为错误结果


async def _execute_tool_calls(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None,
) -> list[ToolResultMessage]:
    """执行助手消息中的所有工具调用。

    根据配置的 tool_execution 模式，选择串行或并行执行。

    Args:
        current_context: 当前 Agent 上下文（用于查找工具定义）。
        assistant_message: 包含工具调用请求的助手消息。
        config: 循环配置。
        emit: 事件接收器。
        signal: 中断信号。

    Returns:
        所有工具调用的结果消息列表。
    """
    tool_calls = [c for c in assistant_message.content if isinstance(c, ToolCall)]
    if config.tool_execution == "sequential":
        return await _execute_tool_calls_sequential(current_context, assistant_message, tool_calls, config, emit, signal)
    return await _execute_tool_calls_parallel(current_context, assistant_message, tool_calls, config, emit, signal)


async def _prepare_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCall,
    config: AgentLoopConfig,
    signal: Any | None,
) -> tuple[_PreparedToolCall | None, AgentToolResult, bool]:
    """准备单个工具调用：查找工具定义、解析参数、执行前置钩子。

    Args:
        current_context: 当前 Agent 上下文。
        assistant_message: 包含工具调用的助手消息。
        tool_call: 单个工具调用请求。
        config: 循环配置。
        signal: 中断信号。

    Returns:
        三元组 (prepared, result, is_error)：
        - prepared: 准备好的工具调用对象，None 表示被拦截或工具未找到。
        - result: 当 prepared 为 None 时，包含错误信息的结果。
        - is_error: 是否为错误（工具未找到或被 before_tool_call 拦截）。
    """
    # 根据工具名称查找工具定义
    tool = next((t for t in current_context.tools if t.name == tool_call.name), None)
    if tool is None:
        return None, _error_tool_result(f"Tool {tool_call.name} not found", approved=False), True

    # 解析工具参数，确保为字典类型
    args = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}

    # 执行前置钩子（before_tool_call），可用于权限校验或拦截工具调用
    if config.before_tool_call:
        before = await _maybe_await(
            config.before_tool_call(
                BeforeToolCallContext(
                    assistant_message=assistant_message,
                    tool_call=tool_call,
                    args=args,
                    context=current_context,
                ),
                signal,
            )
        )
        # 如果钩子返回 block=True，则拦截此次工具调用
        if before and before.block:
            return None, _error_tool_result(before.reason or "Tool execution was blocked", approved=False), True

    return _PreparedToolCall(tool_call=tool_call, tool=tool, args=args), AgentToolResult(content=[]), False


async def _execute_prepared_tool_call(
    prepared: _PreparedToolCall,
    emit: AgentEventSink,
    signal: Any | None,
) -> _ExecutedToolCall:
    """执行一个已准备好的工具调用。

    调用工具的 execute 方法，收集执行过程中的增量更新事件，
    并在执行完成后统一发送。

    Args:
        prepared: 已准备好的工具调用对象。
        emit: 事件接收器。
        signal: 中断信号。

    Returns:
        包含执行结果和错误标志的 _ExecutedToolCall 对象。
    """
    try:
        # 收集工具执行过程中的增量更新事件
        updates: list[Awaitable[Any] | Any] = []

        def _on_update(partial_result: AgentToolResult) -> None:
            """工具执行过程中的增量回调，将更新事件暂存到列表中。"""
            updates.append(
                emit(
                    {
                        "type": "tool_execution_update",
                        "toolCallId": prepared.tool_call.id,
                        "toolName": prepared.tool_call.name,
                        "args": prepared.tool_call.arguments,
                        "partialResult": partial_result,
                    }
                )
            )

        # 执行工具（可能是同步或异步）
        raw_result = prepared.tool.execute(prepared.tool_call.id, prepared.args, signal, _on_update)
        result = await _maybe_await(raw_result)

        # 等待所有增量更新事件发送完毕
        for u in updates:
            await _maybe_await(u)
        return _ExecutedToolCall(result=result, is_error=bool(result.is_error))
    except Exception as exc:
        # 工具执行异常，返回错误结果
        return _ExecutedToolCall(result=_error_tool_result(str(exc)), is_error=True)


async def _finalize_executed_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    prepared: _PreparedToolCall,
    executed: _ExecutedToolCall,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None,
) -> ToolResultMessage:
    """对已执行的工具调用进行收尾处理。

    流程：
    1. 执行后置钩子（after_tool_call），允许修改结果内容或错误标志。
    2. 发送 tool_execution_end 事件。
    3. 构造 ToolResultMessage 并发送 message_start/end 事件。

    Args:
        current_context: 当前 Agent 上下文。
        assistant_message: 包含工具调用的助手消息。
        prepared: 已准备好的工具调用对象。
        executed: 工具执行结果。
        config: 循环配置。
        emit: 事件接收器。
        signal: 中断信号。

    Returns:
        构造好的 ToolResultMessage，可直接追加到消息历史中。
    """
    result = executed.result
    is_error = executed.is_error or bool(result.is_error)

    # 执行后置钩子（after_tool_call），允许对结果进行后处理
    if config.after_tool_call:
        after = await _maybe_await(
            config.after_tool_call(
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
        # 钩子可以覆盖结果的内容、详情和错误标志
        if after:
            if after.content is not None:
                result.content = after.content
            if after.details is not None:
                result.details = after.details
            if after.is_error is not None:
                is_error = after.is_error

    result.is_error = is_error

    # 发送工具执行结束事件
    await _emit(
        emit,
        {
            "type": "tool_execution_end",
            "toolCallId": prepared.tool_call.id,
            "toolName": prepared.tool_call.name,
            "result": result,
            "isError": is_error,
            "approved": result.approved,
            "approvalId": result.approval_id,
            "errorReason": _tool_error_reason(result, is_error),
        },
    )

    # 构造工具结果消息并发送消息事件
    tool_result_message = ToolResultMessage(
        tool_call_id=prepared.tool_call.id,
        tool_name=prepared.tool_call.name,
        content=result.content,
        details=result.details,
        is_error=is_error,
        timestamp=_now_ms(),
    )
    await _emit(emit, {"type": "message_start", "message": tool_result_message})
    await _emit(emit, {"type": "message_end", "message": tool_result_message})
    return tool_result_message


# ── 串行执行 ────────────────────────────────────────────────────

async def _execute_tool_calls_sequential(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None,
) -> list[ToolResultMessage]:
    """串行执行所有工具调用（一个接一个，按顺序依次执行）。

    适用于工具之间存在依赖关系，或需要严格控制执行顺序的场景。

    Args:
        current_context: 当前 Agent 上下文。
        assistant_message: 包含工具调用的助手消息。
        tool_calls: 待执行的工具调用列表。
        config: 循环配置。
        emit: 事件接收器。
        signal: 中断信号。

    Returns:
        所有工具调用的结果消息列表（与输入顺序一致）。
    """
    results: list[ToolResultMessage] = []
    for tool_call in tool_calls:
        # 发送工具执行开始事件
        await _emit(
            emit,
            {
                "type": "tool_execution_start",
                "toolCallId": tool_call.id,
                "toolName": tool_call.name,
                "args": tool_call.arguments,
            },
        )
        # 准备工具调用（查找工具、解析参数、执行前置钩子）
        prepared, immediate, immediate_is_error = await _prepare_tool_call(
            current_context, assistant_message, tool_call, config, signal
        )
        if prepared is None:
            # 工具未找到或被拦截，直接记录错误结果
            immediate.is_error = immediate_is_error
            await _emit(
                emit,
                {
                    "type": "tool_execution_end",
                    "toolCallId": tool_call.id,
                    "toolName": tool_call.name,
                    "result": immediate,
                    "isError": immediate_is_error,
                    "approved": immediate.approved,
                    "approvalId": immediate.approval_id,
                    "errorReason": _tool_error_reason(immediate, immediate_is_error),
                },
            )
            msg = ToolResultMessage(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                content=immediate.content,
                details=immediate.details,
                is_error=immediate_is_error,
                timestamp=_now_ms(),
            )
            await _emit(emit, {"type": "message_start", "message": msg})
            await _emit(emit, {"type": "message_end", "message": msg})
            results.append(msg)
            continue

        # 执行工具并收尾（后置钩子 + 事件发送）
        executed = await _execute_prepared_tool_call(prepared, emit, signal)
        results.append(
            await _finalize_executed_tool_call(
                current_context, assistant_message, prepared, executed, config, emit, signal
            )
        )
    return results


# ── 并行执行 ────────────────────────────────────────────────────

async def _execute_tool_calls_parallel(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None,
) -> list[ToolResultMessage]:
    """并行执行所有工具调用（同时发起，用 asyncio.gather 等待全部完成）。

    适用于工具之间相互独立的场景，可显著提升执行效率。

    执行策略：
    1. 先串行完成所有准备工作（查找工具、执行前置钩子）。
    2. 将准备好的工具调用创建为并发 Task。
    3. 用 asyncio.gather 等待所有 Task 完成。
    4. 串行完成收尾工作（后置钩子 + 事件发送）。

    Args:
        current_context: 当前 Agent 上下文。
        assistant_message: 包含工具调用的助手消息。
        tool_calls: 待执行的工具调用列表。
        config: 循环配置。
        emit: 事件接收器。
        signal: 中断信号。

    Returns:
        所有工具调用的结果消息列表（包含被拦截的错误结果和正常执行结果）。
    """
    immediate_results: list[ToolResultMessage] = []  # 被拦截/未找到的工具的错误结果
    prepared_calls: list[_PreparedToolCall] = []      # 准备好的工具调用

    for tool_call in tool_calls:
        # 发送工具执行开始事件
        await _emit(
            emit,
            {
                "type": "tool_execution_start",
                "toolCallId": tool_call.id,
                "toolName": tool_call.name,
                "args": tool_call.arguments,
            },
        )
        # 准备工具调用
        prepared, immediate, immediate_is_error = await _prepare_tool_call(
            current_context, assistant_message, tool_call, config, signal
        )
        if prepared is None:
            # 工具未找到或被拦截，记录错误结果
            immediate.is_error = immediate_is_error
            await _emit(
                emit,
                {
                    "type": "tool_execution_end",
                    "toolCallId": tool_call.id,
                    "toolName": tool_call.name,
                    "result": immediate,
                    "isError": immediate_is_error,
                    "approved": immediate.approved,
                    "approvalId": immediate.approval_id,
                    "errorReason": _tool_error_reason(immediate, immediate_is_error),
                },
            )
            msg = ToolResultMessage(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                content=immediate.content,
                details=immediate.details,
                is_error=immediate_is_error,
                timestamp=_now_ms(),
            )
            await _emit(emit, {"type": "message_start", "message": msg})
            await _emit(emit, {"type": "message_end", "message": msg})
            immediate_results.append(msg)
        else:
            prepared_calls.append(prepared)

    # 将所有准备好的工具调用并发执行
    tasks = [asyncio.create_task(_execute_prepared_tool_call(pc, emit, signal)) for pc in prepared_calls]
    executed_results = await asyncio.gather(*tasks)

    # 串行完成收尾（后置钩子 + 事件发送），保证结果顺序与调用顺序一致
    finalized: list[ToolResultMessage] = []
    for prepared, executed in zip(prepared_calls, executed_results):
        finalized.append(
            await _finalize_executed_tool_call(
                current_context, assistant_message, prepared, executed, config, emit, signal
            )
        )
    return [*immediate_results, *finalized]
