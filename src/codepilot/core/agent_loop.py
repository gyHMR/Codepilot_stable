"""
Agent 核心执行循环模块。

本模块实现了一次完整的 Agent Run（运行）流程：
用户提示 → 模型推理 → 工具执行 → 返回 RunResult

核心设计:
    1. 事件驱动架构：通过 AgentEventEmitter 发射生命周期事件
    2. 状态机管理：RunState 追踪运行状态和计数器
    3. 错误恢复：支持重试机制和优雅降级
    4. 任务控制：可选的 TaskController 提供任务规划和完成度检查

执行流程:
    run_agent_loop (新任务)
        ↓
    run_agent_loop_continue (继续未完成任务)
        ↓
    _run_safely (异常安全包装)
        ↓
    _run_loop (核心循环)
        ├── LLM 推理 → 获取助手响应
        ├── 工具调用 → 执行并收集结果
        ├── 任务状态更新 → 检查完成度
        └── 循环直到任务完成或出错

关键组件:
    - LLMStreamRunner: 流式调用 LLM 并处理响应
    - ToolCallCoordinator: 协调工具调用的执行
    - TaskController: 可选的任务规划和控制
    - RunState: 运行状态追踪
    - AgentEventEmitter: 事件发射器

停止原因 (stop_reason):
    - "final_answer": 正常完成，模型给出了最终答案
    - "aborted": 用户取消或工具执行被取消
    - "model_error": LLM 调用失败
    - "internal_error": 内部运行时错误
    - "repeated_tool_call": 检测到重复的工具调用
    - "max_iterations": 达到最大工具迭代次数
    - "approval_required": 工具执行需要用户审批
    - "replan_limit": 任务重新规划次数超限
    - "task_blocked": 任务控制器判断需要用户确认或外部指示
    - "task_incomplete": 任务控制器判断完成条件未满足
"""

from __future__ import annotations

import asyncio
from typing import Any

from codepilot.protocols import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage
from codepilot.protocols import (
    AgentEventSink,
    AgentRunResult,
    AgentRunStopReason,
    TaskSummary,
    ErrorInfo,
)

from .events import AgentEventEmitter, maybe_await
from .llm_runner import LLMStreamRunner, StreamFn
from .run_decisions import (
    decide_completion_run,
    decide_model_retry,
    decide_post_tool_run,
    decide_tool_execution_gate,
)
from .run_state import RunState, new_run_id
from .task_controller import TaskController
from .task_planner import TaskPlanner, TaskPlanDraft
from .task_tools import complete_task_step_tool, has_complete_task_step_tool
from .tool_coordinator import ToolCallCoordinator
from .types import AgentContext, AgentLoopConfig, AgentMessage


async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None = None,
    stream_fn: StreamFn | None = None,
    *,
    run_id: str | None = None,
) -> AgentRunResult:
    """
    运行一个新的用户任务并返回结构化结果。

    这是 Agent 执行的主入口函数，接收用户输入（prompts），
    执行完整的推理-工具调用循环，最终返回结构化的运行结果。

    执行流程:
        1. 创建运行状态和事件发射器
        2. 将用户提示追加到上下文消息列表
        3. 发射 agent_start 和 turn_start 事件
        4. 发射每个用户消息的 message_start/message_end 事件
        5. 调用 _run_safely 进入核心执行循环

    Args:
        prompts: 用户输入的消息列表（通常是一条用户消息）
            示例: [UserMessage(content="帮我分析这段代码")]
        context: Agent 上下文，包含系统提示、历史消息、工具列表等
            - system_prompt: 系统提示词，定义 Agent 角色
            - messages: 历史消息列表
            - tools: 可用工具列表
            - current_task: 当前任务上下文（可选）
            - task_recovery_projection: 会话持久化的任务恢复投影（可选）
            - task_signal: 任务控制信号（可选）
        config: Agent 循环配置，包含重试策略、工具执行模式等
            - session_id: 会话标识符
            - retry_enabled: 是否启用重试
            - max_model_retries: 最大重试次数
            - retry_base_delay_ms: 重试基础延迟
            - max_tool_iterations: 最大工具迭代次数
            - repeated_tool_call_limit: 重复工具调用限制
            - task_control_enabled: 是否启用任务控制
            - get_steering_messages: 获取引导消息的回调
            - get_follow_up_messages: 获取后续消息的回调
        emit: 事件接收器，用于接收运行过程中的各种事件
            事件类型包括: agent_start, turn_start, message_start,
            message_end, turn_end, agent_end, error 等
        signal: 可选的取消信号，用于中断运行
            当 signal 被触发时，运行会被优雅地取消
        stream_fn: 可选的流式输出函数，用于实时输出 LLM 生成内容
            接受 token 文本，用于实现打字机效果
        run_id: 可选的运行 ID，不提供则自动生成
            格式示例: "run_abc123def456"

    Returns:
        AgentRunResult: 结构化的运行结果
            - status: 运行状态 (completed/failed/aborted/waiting_approval)
            - stop_reason: 停止原因
            - messages: 本轮新增的消息列表
            - final_message: 最后的助手消息
            - counters: 运行计数器 (model_attempts, tool_iterations 等)
            - error: 错误信息 (仅在失败时)
            - task: 任务摘要 (仅在启用任务控制时)

    示例:
        result = await run_agent_loop(
            prompts=[UserMessage(content="你好")],
            context=AgentContext(
                system_prompt="你是一个有帮助的助手",
                messages=[],
                tools=[],
            ),
            config=my_config,
            emit=my_event_handler,
        )
        print(result.status)  # "completed"
        print(result.final_message.content)  # 助手的回复
    """

    # 创建运行状态对象，用于追踪本次运行的计数器和状态
    state = RunState(run_id=run_id or new_run_id(), session_id=config.session_id)

    # 创建事件发射器，包装 emit 回调并注入 run_id 和 session_id
    emitter = AgentEventEmitter(
        emit,
        run_id=state.run_id,
        session_id=config.session_id,
    )

    # 本轮新增的消息列表（不包含历史消息）
    new_messages: list[AgentMessage] = list(prompts)

    # 构建当前上下文：将历史消息与本次提示合并
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=[*context.messages, *prompts],  # 历史消息 + 本次提示
        tools=context.tools,
        current_task=context.current_task,
        task_recovery_projection=context.task_recovery_projection,
        task_signal=context.task_signal,
    )

    # 发射生命周期事件：运行开始、轮次开始
    await emitter.emit({"type": "agent_start"})
    await emitter.emit({"type": "turn_start"})

    # 为每个用户消息发射 message_start/message_end 事件
    # 这允许上层监听器追踪消息的处理过程
    for prompt in prompts:
        await emitter.emit({"type": "message_start", "message": prompt})
        await emitter.emit({"type": "message_end", "message": prompt})

    # 进入核心执行循环（带异常安全保护）
    return await _run_safely(
        current_context,
        new_messages,
        config,
        emitter,
        state,
        signal=signal,
        stream_fn=stream_fn,
    )


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None = None,
    stream_fn: StreamFn | None = None,
    *,
    run_id: str | None = None,
) -> AgentRunResult:
    """
    继续一个内存中尚未完成的 Run。

    典型使用场景：
        - 用户消息后，助手返回了工具调用请求
        - 工具执行完成后，需要继续推理流程
        - 恢复被中断的运行（如审批后继续）

    前置条件:
        - context.messages 不能为空
        - 最后一条消息不能是助手消息（否则没有继续的意义）

    与 run_agent_loop 的区别:
        - 不接收 prompts 参数（使用现有上下文）
        - 不发射用户消息的 message_start/end 事件
        - 保持原有消息历史，继续推理

    Args:
        context: Agent 上下文，必须包含至少一条消息
            最后一条消息通常是工具结果消息 (ToolResultMessage)
        config: Agent 循环配置
        emit: 事件接收器
        signal: 可选的取消信号
        stream_fn: 可选的流式输出函数
        run_id: 可选的运行 ID（通常与原 Run 相同）

    Returns:
        AgentRunResult: 继续运行的结果

    Raises:
        ValueError: 当上下文为空或最后一条消息是助手消息时
    """

    # 参数校验：确保可以继续运行
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")

    # 如果最后一条是助手消息，说明助手已经完成推理，无需继续
    if isinstance(context.messages[-1], AssistantMessage):
        raise ValueError("Cannot continue from message role: assistant")

    # 创建新的运行状态（可以复用原 run_id）
    state = RunState(run_id=run_id or new_run_id(), session_id=config.session_id)
    emitter = AgentEventEmitter(
        emit,
        run_id=state.run_id,
        session_id=config.session_id,
    )

    # 继续运行不产生新的用户消息
    new_messages: list[AgentMessage] = []

    # 构建上下文：直接使用传入的消息列表
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages),  # 包含工具结果的完整消息历史
        tools=context.tools,
        current_task=context.current_task,
        task_recovery_projection=context.task_recovery_projection,
        task_signal=context.task_signal,
    )

    # 发射开始事件（不发射用户消息事件）
    await emitter.emit({"type": "agent_start"})
    await emitter.emit({"type": "turn_start"})

    # 进入核心执行循环
    return await _run_safely(
        current_context,
        new_messages,
        config,
        emitter,
        state,
        signal=signal,
        stream_fn=stream_fn,
    )


# ── 内部函数 ────────────────────────────────────────────────────


async def _run_safely(
    current_context: AgentContext,
    new_messages: list[AgentMessage],
    config: AgentLoopConfig,
    emitter: AgentEventEmitter,
    state: RunState,
    *,
    signal: Any | None,
    stream_fn: StreamFn | None,
) -> AgentRunResult:
    """
    异常安全的运行包装器。

    功能:
        1. 调用 _run_loop 执行核心循环
        2. 捕获 asyncio.CancelledError → 返回 aborted 状态
        3. 捕获其他异常 → 返回 failed 状态并记录错误信息

    这个函数确保无论发生什么异常，都能返回一个有效的 AgentRunResult，
    而不会让异常向上传播导致调用方崩溃。

    Args:
        current_context: 当前 Agent 上下文
        new_messages: 本轮新增的消息列表
        config: 循环配置
        emitter: 事件发射器
        state: 运行状态
        signal: 取消信号
        stream_fn: 流式输出函数

    Returns:
        AgentRunResult: 运行结果（成功、失败或取消）
    """

    try:
        # 尝试执行核心循环
        return await _run_loop(
            current_context,
            new_messages,
            config,
            emitter,
            state,
            signal=signal,
            stream_fn=stream_fn,
        )
    except asyncio.CancelledError:
        # 运行被取消（通常是用户主动取消或超时）
        # 返回 aborted 状态，不记录错误
        return await _finish_run(
            emitter,
            state.result(
                status="aborted",
                stop_reason="aborted",
                messages=new_messages,
                final_message=_last_assistant(new_messages),
            ),
        )
    except Exception as exc:
        # 发生未预期的内部错误
        # 构建错误信息并发射 error 事件
        error = ErrorInfo(
            code="run.internal_error",
            message=str(exc),
            retryable=False,  # 内部错误通常不可重试
            source="runtime",
            details={"exception": type(exc).__name__},
        )
        await emitter.emit(
            {
                "type": "error",
                "error": error.code,
                "message": error.message,
                "source": error.source,
                "code": error.code,
                "retryable": False,
                "errorInfo": error,
            }
        )
        # 返回 failed 状态
        return await _finish_run(
            emitter,
            state.result(
                status="failed",
                stop_reason="internal_error",
                messages=new_messages,
                final_message=_last_assistant(new_messages),
                error=error,
            ),
        )


async def _run_loop(
    current_context: AgentContext,
    new_messages: list[AgentMessage],
    config: AgentLoopConfig,
    emitter: AgentEventEmitter,
    state: RunState,
    *,
    signal: Any | None,
    stream_fn: StreamFn | None,
) -> AgentRunResult:
    """
    核心执行循环：模型推理 → 工具执行 → 任务状态更新，循环直到任务完成或出错。

    这是整个 Agent 系统的核心函数，实现了完整的推理-执行循环。

    循环流程:
        ┌─────────────────────────────────────────────────────────┐
        │  while True:                                            │
        │    ┌─────────────────────────────────────────────────┐  │
        │    │  while has_more_tool_calls or pending_messages:  │  │
        │    │    1. 注入待处理消息（引导消息、后续消息）       │  │
        │    │    2. 更新任务上下文                             │  │
        │    │    3. 调用 LLM 获取助手响应                     │  │
        │    │    4. 如果有工具调用：                          │  │
        │    │       a. 检查重复调用限制                       │  │
        │    │       b. 检查最大迭代次数                       │  │
        │    │       c. 执行工具调用                           │  │
        │    │       d. 更新任务状态                           │  │
        │    │       e. 检查是否需要停止（审批、取消）         │  │
        │    │    5. 检查是否有新的待处理消息                  │  │
        │    └─────────────────────────────────────────────────┘  │
        │    检查任务完成度                                        │
        │    如果任务未完成且可继续 → 生成引导消息，继续循环     │
        │    检查是否有后续消息                                    │
        │    如果没有 → 返回完成状态                              │
        └─────────────────────────────────────────────────────────┘

    Args:
        current_context: 当前 Agent 上下文
        new_messages: 本轮新增的消息列表（会被修改）
        config: 循环配置
        emitter: 事件发射器
        state: 运行状态（会被修改）
        signal: 取消信号
        stream_fn: 流式输出函数

    Returns:
        AgentRunResult: 运行结果
    """

    # ── 初始化核心组件 ──────────────────────────────────────────

    # LLM 流式运行器：负责调用 LLM 并处理流式响应
    llm_runner = LLMStreamRunner(config=config, emitter=emitter, stream_fn=stream_fn)

    # 工具调用协调器：负责执行工具调用并收集结果
    tool_coordinator = ToolCallCoordinator(config=config, emitter=emitter)

    # 任务控制器（可选）：负责任务规划和完成度检查
    task_controller = TaskController() if config.task_control_enabled else None
    planned_task: TaskPlanDraft | None = None
    if (
        task_controller is not None
        and config.task_planner_enabled
        and current_context.task_recovery_projection is None
        and current_context.messages
    ):
        api_key = (
            await maybe_await(config.get_api_key(config.model.provider))
            if config.get_api_key is not None
            else None
        )
        planned_task = await TaskPlanner().generate(
            model=config.model,
            messages=current_context.messages,
            convert_to_llm=config.convert_to_llm,
            fallback_goal=_latest_user_goal(current_context.messages),
            stream_fn=stream_fn,
            api_key=api_key,
            session_id=config.session_id,
        )

    # 初始化任务（如果启用了任务控制）
    task = (
        task_controller.initialize(
            current_context.messages,
            goal=planned_task.goal if planned_task is not None else None,
            proposed_steps=planned_task.steps if planned_task is not None else None,
            task_recovery_projection=current_context.task_recovery_projection,
        )
        if task_controller is not None
        else None
    )

    # 发射任务计划创建事件
    if task_controller is not None and task is not None:
        if not has_complete_task_step_tool(current_context.tools):
            current_context.tools.append(complete_task_step_tool())
        await emitter.emit(
            {
                "type": "task_plan_created",
                "task": task_controller.event_payload(task),
            }
        )

    # 标记是否为第一次迭代（第一次不发射 turn_start 事件，因为外层已发射）
    first_iteration = True

    # 获取初始的待处理消息（引导消息）
    pending_messages = await _drain(config.get_steering_messages)

    # 模型重试计数器
    model_retries = 0

    # ── 主循环 ──────────────────────────────────────────────────

    while True:
        # 内层循环：处理工具调用和待处理消息
        # has_more_tool_calls: 上一轮是否有工具调用需要继续处理
        # pending_messages: 是否有待注入的引导消息
        has_more_tool_calls = True

        while has_more_tool_calls or pending_messages:
            # 第一次迭代不发射 turn_start（外层已发射）
            if first_iteration:
                first_iteration = False
            else:
                await emitter.emit({"type": "turn_start"})

            # 注入待处理消息（引导消息、后续消息等）
            if pending_messages:
                await _inject_messages(
                    pending_messages,
                    current_context,
                    new_messages,
                    emitter,
                )
                pending_messages = []  # 清空待处理队列

            # 更新任务上下文（供 LLM 参考）
            current_context.current_task = (
                task_controller.render_context(task)
                if task_controller is not None and task is not None
                else None
            )
            # 更新任务控制信号
            current_context.task_signal = (
                task_controller.control_signal(task)
                if task_controller is not None and task is not None
                else None
            )

            # 记录模型尝试次数
            state.counters.model_attempts += 1

            # ── 调用 LLM 获取助手响应 ──────────────────────────
            assistant = await llm_runner.stream_assistant_response(
                current_context,
                signal=signal,
            )
            new_messages.append(assistant)

            # ── 处理 LLM 错误 ─────────────────────────────────
            if assistant.stop_reason == "error":
                await emitter.emit(
                    {"type": "turn_end", "message": assistant, "toolResults": []}
                )

                # 获取错误信息
                error = assistant.error_info or ErrorInfo(
                    code="llm.unknown",
                    message=assistant.error_message or "Unknown model error",
                    retryable=False,
                    source="llm",
                )

                retry = decide_model_retry(
                    error,
                    retries_so_far=model_retries,
                    config=config,
                )
                if retry.should_retry:
                    model_retries = retry.next_retry_count
                    # 发射重试开始事件
                    await emitter.emit(
                        {
                            "type": "model_retry_start",
                            "attempt": model_retries,
                            "maxAttempts": config.max_model_retries + 1,
                            "delayMs": retry.delay_ms,
                            "error": error,
                        }
                    )

                    # 移除失败的助手消息，以便重试
                    if (
                        current_context.messages
                        and current_context.messages[-1] is assistant
                    ):
                        current_context.messages.pop()

                    # 等待延迟时间后重试
                    await asyncio.sleep(retry.delay_ms / 1000.0)
                    continue  # 重试当前轮次

                # 不可重试或达到最大重试次数 → 返回失败
                return await _finish_run(
                    emitter,
                    state.result(
                        status="failed",
                        stop_reason="model_error",
                        messages=new_messages,
                        final_message=assistant,
                        error=error,
                    ),
                )

            # ── 提取工具调用 ──────────────────────────────────
            tool_calls = [
                content for content in assistant.content if isinstance(content, ToolCall)
            ]
            has_more_tool_calls = bool(tool_calls)
            tool_results: list[ToolResultMessage] = []
            decision = None

            # ── 执行工具调用 ──────────────────────────────────
            if has_more_tool_calls:
                gate = decide_tool_execution_gate(tool_calls, state, config)
                if not gate.should_execute:
                    if gate.assistant_stop_reason is not None:
                        assistant.stop_reason = gate.assistant_stop_reason
                    return await _stop_with_error(
                        emitter,
                        state,
                        new_messages,
                        assistant,
                        code=gate.error_code or "run.tool_execution_blocked",
                        message=gate.message or gate.reason,
                        stop_reason=gate.stop_reason or "internal_error",
                    )

                # 记录工具迭代次数
                state.counters.tool_iterations += 1

                # 批量执行工具调用（支持并行或顺序执行）
                tool_results = await tool_coordinator.execute_batch(
                    current_context,
                    assistant,
                    signal=signal,
                )

                # 收集工具结果到状态追踪器
                state.collect_tool_results(tool_results)

                # 通知任务控制器工具执行完成，获取决策
                decision = (
                    task_controller.after_tool_results(task, state, tool_results)
                    if task_controller is not None and task is not None
                    else None
                )

                # 发射任务状态更新事件
                if task_controller is not None and task is not None and decision is not None:
                    await emitter.emit(
                        {
                            "type": "task_step_updated",
                            "task": task_controller.event_payload(task),
                        }
                    )
                    await emitter.emit(
                        {
                            "type": "task_decision",
                            "decision": {
                                "action": decision.action,
                                "reason": decision.reason,
                                "next_action": decision.next_action,
                            },
                            "task": task_controller.event_payload(task),
                        }
                    )

                # 将工具结果追加到消息列表
                current_context.messages.extend(tool_results)
                new_messages.extend(tool_results)

            # 发射轮次结束事件
            await emitter.emit(
                {"type": "turn_end", "message": assistant, "toolResults": tool_results}
            )

            # ── 检查特殊状态 ──────────────────────────────────

            post_tool = decide_post_tool_run(tool_results, task_decision=decision)
            if post_tool.should_stop:
                task_summary = (
                    task_controller.summarize(task)
                    if task_controller is not None and task is not None
                    else None
                )
                if post_tool.error_code is not None:
                    return await _stop_with_error(
                        emitter,
                        state,
                        new_messages,
                        assistant,
                        code=post_tool.error_code,
                        message=post_tool.message or post_tool.reason,
                        stop_reason=post_tool.stop_reason or "internal_error",
                        task=task_summary,
                    )
                return await _finish_run(
                    emitter,
                    state.result(
                        status=post_tool.status or "waiting_user",
                        stop_reason=post_tool.stop_reason or "task_blocked",
                        messages=new_messages,
                        final_message=assistant,
                        task=task_summary,
                    ),
                )

            # 获取新的待处理消息（可能由工具执行触发）
            pending_messages = await _drain(config.get_steering_messages)

        # ── 任务完成度检查 ──────────────────────────────────────

        # 如果启用了任务控制，检查任务是否完成
        if task_controller is not None and task is not None:
            completion = task_controller.check_completion(task, state)

            # 发射完成度检查事件
            await emitter.emit(
                {
                    "type": "completion_checked",
                    "completion": {
                        "satisfied": completion.satisfied,
                        "reason": completion.reason,
                        "missing": list(completion.missing),
                        "can_continue": completion.can_continue,
                        "unverified": completion.unverified,
                    },
                    "task": task_controller.event_payload(task),
                }
            )

            completion_decision = decide_completion_run(completion)
            if completion_decision.action == "continue_with_steering":
                pending_messages = [task_controller.completion_steering(completion)]
                continue
            if completion_decision.should_stop:
                return await _finish_run(
                    emitter,
                    state.result(
                        status=completion_decision.status or "waiting_user",
                        stop_reason=completion_decision.stop_reason or "task_incomplete",
                        messages=new_messages,
                        final_message=_last_assistant(new_messages),
                        task=task_controller.summarize(task),
                    ),
                )

        # ── 检查后续消息 ────────────────────────────────────────

        # 获取后续消息（如用户追加的指令）
        followups = await _drain(config.get_follow_up_messages)
        if followups:
            pending_messages = followups
            continue

        # ── 正常完成 ────────────────────────────────────────────

        # 没有待处理的消息，任务完成
        return await _finish_run(
            emitter,
            state.result(
                status="completed",
                stop_reason="final_answer",
                messages=new_messages,
                final_message=_last_assistant(new_messages),
                task=(
                    task_controller.summarize(task)
                    if task_controller is not None and task is not None
                    else None
                ),
            ),
        )


async def _stop_with_error(
    emitter: AgentEventEmitter,
    state: RunState,
    messages: list[AgentMessage],
    assistant: AssistantMessage,
    *,
    code: str,
    message: str,
    stop_reason: AgentRunStopReason,
    task: TaskSummary | None = None,
) -> AgentRunResult:
    """
    因错误终止运行。

    处理流程:
        1. 设置助手消息的错误信息
        2. 构建 ErrorInfo 对象
        3. 发射 error 事件
        4. 发射 turn_end 事件
        5. 返回 failed 状态的 RunResult

    Args:
        emitter: 事件发射器
        state: 运行状态
        messages: 消息列表
        assistant: 最后的助手消息
        code: 错误代码 (如 "run.repeated_tool_call")
        message: 错误描述信息
        stop_reason: 停止原因
        task: 可选的任务摘要

    Returns:
        AgentRunResult: 失败状态的运行结果
    """

    # 设置助手消息的错误信息
    assistant.error_message = message

    # 构建错误信息对象
    error = ErrorInfo(
        code=code,
        message=message,
        retryable=False,  # 这类错误通常不可重试
        source="runtime",
    )

    # 发射错误事件
    await emitter.emit(
        {
            "type": "error",
            "error": code,
            "message": message,
            "source": "runtime",
            "code": code,
            "retryable": False,
            "errorInfo": error,
        }
    )

    # 发射轮次结束事件
    await emitter.emit({"type": "turn_end", "message": assistant, "toolResults": []})

    # 返回失败状态的结果
    return await _finish_run(
        emitter,
        state.result(
            status="failed",
            stop_reason=stop_reason,
            messages=messages,
            final_message=assistant,
            error=error,
            task=task,
        ),
    )


async def _finish_run(
    emitter: AgentEventEmitter,
    result: AgentRunResult,
) -> AgentRunResult:
    """
    完成运行：发射 agent_end 事件并返回最终结果。

    这是所有运行路径的最终汇聚点，无论是正常完成、错误终止还是取消，
    都会通过这个函数发射结束事件。

    Args:
        emitter: 事件发射器
        result: 运行结果

    Returns:
        AgentRunResult: 原样返回传入的结果
    """

    # 发射运行结束事件，包含完整的运行信息
    await emitter.emit(
        {
            "type": "agent_end",
            "messages": result.messages,
            "status": result.status,
            "stopReason": result.stop_reason,
            "counters": result.counters,
            "result": result,
        }
    )
    return result


async def _inject_messages(
    messages: list[AgentMessage],
    current_context: AgentContext,
    new_messages: list[AgentMessage],
    emitter: AgentEventEmitter,
) -> None:
    """
    将引导/后续消息注入上下文。

    注入流程:
        1. 对每条消息发射 message_start 事件
        2. 对每条消息发射 message_end 事件
        3. 将消息追加到当前上下文
        4. 将消息追加到新增消息列表

    这些消息通常来自:
        - 引导消息 (steering messages): 引导模型行为
        - 后续消息 (follow-up messages): 用户追加的指令
        - 任务控制消息: 任务规划和完成度反馈

    Args:
        messages: 要注入的消息列表
        current_context: 当前 Agent 上下文（会被修改）
        new_messages: 新增消息列表（会被修改）
        emitter: 事件发射器
    """

    for message in messages:
        # 发射消息生命周期事件
        await emitter.emit({"type": "message_start", "message": message})
        await emitter.emit({"type": "message_end", "message": message})
        # 注入到上下文
        current_context.messages.append(message)
        new_messages.append(message)


async def _drain(callback):
    """
    排空回调队列：调用回调并返回结果列表。

    用于获取异步回调的结果，支持同步和异步回调。

    Args:
        callback: 回调函数，可以是 None、同步函数或异步函数

    Returns:
        list: 回调返回的结果列表，callback 为 None 时返回空列表
    """

    if callback is None:
        return []
    return await maybe_await(callback())


def _last_assistant(messages: list[AgentMessage]) -> AssistantMessage | None:
    """
    从消息列表末尾反向查找最近一条助手消息。

    用于获取最后的助手响应，通常用于设置 AgentRunResult.final_message。

    Args:
        messages: 消息列表

    Returns:
        AssistantMessage | None: 最近的助手消息，如果没有则返回 None
    """

    return next(
        (message for message in reversed(messages) if isinstance(message, AssistantMessage)),
        None,
    )


def _latest_user_goal(messages: list[AgentMessage]) -> str:
    for message in reversed(messages):
        if not isinstance(message, UserMessage):
            continue
        if isinstance(message.content, str):
            return message.content.strip() or "完成当前请求"
        text = "".join(
            block.text for block in message.content if isinstance(block, TextContent)
        ).strip()
        return text or "完成当前请求"
    return "继续当前任务"
