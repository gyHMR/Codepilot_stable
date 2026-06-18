from __future__ import annotations

"""
对外 Agent 封装：
提供 prompt/continue、状态管理、事件订阅、串行调度入口。
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from codepilot.protocols import (
    AgentEvent,
    AgentEventSink,
    AgentRunResult,
    ImageContent,
    Message,
    Model,
    TextContent,
    ThinkingLevel,
    UserMessage,
)

from .agent_loop import run_agent_loop_continue_result, run_agent_loop_result
from .types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentLoopConfig,
    AgentMessage,
    AgentState,
    AgentTool,
    BeforeToolCallContext,
    BeforeToolCallResult,
    ToolExecutionMode,
)


# ── 工具函数 ────────────────────────────────────────────────────

def _default_convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    """默认的消息转换函数：直接透传，不做任何修改。"""
    return messages


async def _maybe_await(value: Any) -> Any:
    """兼容处理同步/异步回调：如果是可等待对象则 await，否则直接返回。"""
    if asyncio.isfuture(value) or asyncio.iscoroutine(value):
        return await value
    return value


def _resolve_reasoning(thinking_level: str) -> ThinkingLevel | None:
    """将字符串形式的思考级别映射为 ThinkingLevel 类型。

    Args:
        thinking_level: 思考级别字符串，可选值为
            "off" / "minimal" / "low" / "medium" / "high" / "xhigh"。

    Returns:
        对应的 ThinkingLevel 值；"off" 返回 None 表示关闭推理。
    """
    mapping = {
        "off": None,
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "xhigh",
    }
    return mapping.get(thinking_level)  # type: ignore[return-value]


# ── Agent 配置 ──────────────────────────────────────────────────

@dataclass
class AgentOptions:
    """Agent 实例的配置选项。

    Attributes:
        model: 使用的 LLM 模型信息。
        system_prompt: 系统提示词，定义 Agent 的行为准则。
        tools: 可供 Agent 调用的工具列表。
        messages: 初始消息列表（用于恢复历史上下文）。
        thinking_level: 推理思考级别，控制模型的思维深度。
        tool_execution: 工具执行模式，"parallel"（并行）或 "sequential"（串行）。
        convert_to_llm: 将内部 AgentMessage 转换为 LLM Message 的函数。
        transform_context: 发送前对消息上下文进行变换的钩子（如裁剪、脱敏等）。
        get_api_key: 获取 API Key 的回调函数，用于动态获取密钥。
        before_tool_call: 工具调用前的拦截钩子，可用于权限校验或参数修改。
        after_tool_call: 工具调用后的拦截钩子，可用于结果后处理。
        max_tool_iterations: 单次运行允许的最大工具反馈迭代次数。
        max_tool_calls_per_turn: 单轮允许的最大工具调用数量，None 表示不限制。
        repeated_tool_call_limit: 连续重复相同工具调用的允许次数。
        session_id: 会话标识，用于关联日志和持久化数据。
    """
    model: Model
    system_prompt: str = ""
    tools: list[AgentTool] = field(default_factory=list)
    messages: list[AgentMessage] = field(default_factory=list)
    thinking_level: str = "off"
    tool_execution: ToolExecutionMode = "parallel"
    convert_to_llm: Callable[[list[AgentMessage]], list[Message] | Awaitable[list[Message]]] = _default_convert_to_llm
    transform_context: Optional[
        Callable[[list[AgentMessage], Any | None], list[AgentMessage] | Awaitable[list[AgentMessage]]]
    ] = None
    get_api_key: Optional[Callable[[str], str | None | Awaitable[str | None]]] = None
    before_tool_call: Optional[
        Callable[[BeforeToolCallContext, Any | None], BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None]]
    ] = None
    after_tool_call: Optional[
        Callable[[AfterToolCallContext, Any | None], AfterToolCallResult | None | Awaitable[AfterToolCallResult | None]]
    ] = None
    max_tool_iterations: int = 12
    max_tool_calls_per_turn: Optional[int] = None
    allow_unmanaged_tools: bool = False
    repeated_tool_call_limit: int = 3
    retry_enabled: bool = True
    max_model_retries: int = 2
    retry_base_delay_ms: int = 1200
    session_id: Optional[str] = None


# ── Agent 核心类 ────────────────────────────────────────────────

class Agent:
    """对外暴露的 Agent 核心类。

    职责：
    - 管理 Agent 的运行状态（消息列表、流式状态、错误信息等）。
    - 提供 prompt（发送新提示）和 continue_run（继续未完成运行）两个入口。
    - 通过事件订阅机制向外部通知运行过程中的各类事件。
    - 保证同一时刻只有一个流式任务在运行（串行调度）。
    - 支持运行中的中断（abort）和引导消息注入（steering）。
    """

    def __init__(self, options: AgentOptions) -> None:
        """初始化 Agent 实例。

        Args:
            options: Agent 配置选项，包含模型、提示词、工具等。
        """
        # 初始化 Agent 运行状态
        self._state = AgentState(
            system_prompt=options.system_prompt,
            model=options.model,
            thinking_level=options.thinking_level,  # type: ignore[arg-type]
            tools=list(options.tools),
            messages=list(options.messages),
        )
        self._options = options
        # 事件监听器列表，外部通过 subscribe 注册
        self._listeners: list[AgentEventSink] = []

        # 当前正在运行的流式任务（asyncio.Task），同一时刻最多只有一个
        self._stream_task: asyncio.Task[AgentRunResult] | None = None
        self._last_run_result: AgentRunResult | None = None
        # 引导消息队列：在 Agent 循环的下一轮迭代中注入，用于中途纠正方向
        self._steering_queue: list[AgentMessage] = []
        # 后续消息队列：在当前轮次结束后追加，用于补充上下文
        self._follow_up_queue: list[AgentMessage] = []

    @property
    def state(self) -> AgentState:
        """获取当前 Agent 的运行状态（只读访问）。"""
        return self._state

    @property
    def last_run_result(self) -> AgentRunResult | None:
        return self._last_run_result

    # ── 状态修改方法 ────────────────────────────────────────────

    def set_system_prompt(self, system_prompt: str) -> None:
        """更新系统提示词。"""
        self._state.system_prompt = system_prompt

    def set_tools(self, tools: list[AgentTool]) -> None:
        """更新可用工具列表。"""
        self._state.tools = list(tools)

    def set_messages(self, messages: list[AgentMessage]) -> None:
        """替换全部消息列表（通常用于上下文压缩后重置）。"""
        self._state.messages = list(messages)

    def add_steering_message(self, message: AgentMessage) -> None:
        """向引导消息队列中添加一条消息。

        引导消息会在 Agent 循环的下一轮迭代开始前被注入，
        用于在工具调用过程中向 Agent 提供额外指示或纠正方向。
        """
        self._steering_queue.append(message)

    def add_follow_up_message(self, message: AgentMessage) -> None:
        """向后续消息队列中添加一条消息。

        后续消息会在当前 LLM 响应结束后被追加到上下文中，
        用于补充信息或触发下一轮对话。
        """
        self._follow_up_queue.append(message)

    def clear_error(self) -> None:
        """清除当前的错误状态。"""
        self._state.error = None

    # ── 事件订阅 ────────────────────────────────────────────────

    def subscribe(self, listener: AgentEventSink) -> Callable[[], None]:
        """注册一个事件监听器，订阅 Agent 运行过程中的各类事件。

        Args:
            listener: 事件回调函数，接收 AgentEvent 参数（可以是同步或异步函数）。

        Returns:
            取消订阅的函数，调用后该监听器将不再接收事件。
        """
        self._listeners.append(listener)

        def _unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return _unsubscribe

    # ── 公共运行入口 ────────────────────────────────────────────

    async def prompt(self, message: str | UserMessage, images: list[str] | None = None) -> list[AgentMessage]:
        """向 Agent 发送一条用户提示并等待回复。

        Args:
            message: 用户输入，可以是纯文本字符串或 UserMessage 对象。
            images: 可选的图片列表（base64 编码或 URL），仅当 message 为字符串时生效。

        Returns:
            本次交互产生的新 AgentMessage 列表。

        Raises:
            RuntimeError: 当 Agent 已有流式任务正在运行时抛出。
        """
        # 防止并发调用：同一时刻只允许一个流式任务
        if self._state.is_streaming:
            raise RuntimeError("Agent is already running")

        # 将字符串输入封装为 UserMessage，附带可选图片
        if isinstance(message, str):
            content: list[TextContent | ImageContent] = [TextContent(text=message)]
            for image in images or []:
                content.append(ImageContent(data=image))
            prompt = UserMessage(content=content)
        else:
            prompt = message

        result = await self._start_run_result(prompts=[prompt], continue_mode=False)
        return result.messages

    async def run(
        self,
        message: str | UserMessage,
        images: list[str] | None = None,
    ) -> AgentRunResult:
        """Start a new Run and return its structured result."""

        if self._state.is_streaming:
            raise RuntimeError("Agent is already running")
        if isinstance(message, str):
            content: list[TextContent | ImageContent] = [TextContent(text=message)]
            for image in images or []:
                content.append(ImageContent(data=image))
            prompt = UserMessage(content=content)
        else:
            prompt = message
        return await self._start_run_result(prompts=[prompt], continue_mode=False)

    async def continue_run(self) -> list[AgentMessage]:
        """继续上一次未完成的 Agent 运行。

        典型场景：Agent 返回了工具调用请求但尚未执行完毕，
        调用此方法可继续执行工具并让 Agent 继续推理。

        Returns:
            继续运行产生的新 AgentMessage 列表。

        Raises:
            RuntimeError: 当 Agent 已有流式任务正在运行时抛出。
        """
        if self._state.is_streaming:
            raise RuntimeError("Agent is already running")
        result = await self._start_run_result(prompts=[], continue_mode=True)
        return result.messages

    async def continue_run_result(self) -> AgentRunResult:
        if self._state.is_streaming:
            raise RuntimeError("Agent is already running")
        return await self._start_run_result(prompts=[], continue_mode=True)

    async def wait_for_idle(self) -> None:
        """等待当前流式任务完成，使 Agent 进入空闲状态。"""
        if self._stream_task is not None:
            await self._stream_task

    def abort(self) -> None:
        """中止当前正在运行的流式任务。"""
        if self._stream_task is not None and not self._stream_task.done():
            self._stream_task.cancel()

    # ── 内部方法 ────────────────────────────────────────────────

    async def _start_run_result(
        self,
        prompts: list[AgentMessage],
        continue_mode: bool,
    ) -> AgentRunResult:
        """启动一次 Agent 运行的核心方法。

        流程：
        1. 标记为流式运行状态，清理上一轮的错误和临时消息。
        2. 构建 AgentLoopConfig 和 AgentContext。
        3. 根据 continue_mode 决定调用 run_agent_loop（新对话）
           或 run_agent_loop_continue（继续上一轮）。
        4. 创建 asyncio.Task 执行循环，等待完成后将新消息追加到状态中。
        5. 异常处理：CancelledError 记录为 "aborted"，其他异常记录错误信息。

        Args:
            prompts: 要发送的提示消息列表（continue_mode 时为空列表）。
            continue_mode: True 表示继续上一轮运行，False 表示发起新对话。

        Returns:
            本次运行产生的新 AgentMessage 列表。
        """
        self._state.is_streaming = True
        self._state.stream_message = None
        self._state.error = None

        # 构建循环配置
        cfg = AgentLoopConfig(
            model=self._state.model,
            convert_to_llm=self._options.convert_to_llm,
            transform_context=self._options.transform_context,
            get_api_key=self._options.get_api_key,
            get_steering_messages=self._drain_steering_messages,
            get_follow_up_messages=self._drain_follow_up_messages,
            tool_execution=self._options.tool_execution,
            before_tool_call=self._options.before_tool_call,
            after_tool_call=self._options.after_tool_call,
            reasoning=_resolve_reasoning(self._state.thinking_level),
            session_id=self._options.session_id,
            max_tool_iterations=self._options.max_tool_iterations,
            max_tool_calls_per_turn=self._options.max_tool_calls_per_turn,
            allow_unmanaged_tools=self._options.allow_unmanaged_tools,
            repeated_tool_call_limit=self._options.repeated_tool_call_limit,
            retry_enabled=self._options.retry_enabled,
            max_model_retries=self._options.max_model_retries,
            retry_base_delay_ms=self._options.retry_base_delay_ms,
        )

        # 构建上下文快照（使用副本，避免循环过程中被外部修改）
        context = AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages),
            tools=list(self._state.tools),
        )

        # 根据模式选择对应的循环入口
        if continue_mode:
            coro = run_agent_loop_continue_result(
                context=context,
                config=cfg,
                emit=self._dispatch_event,
            )
        else:
            coro = run_agent_loop_result(
                prompts=prompts,
                context=context,
                config=cfg,
                emit=self._dispatch_event,
            )

        # 创建异步任务并等待完成
        self._stream_task = asyncio.create_task(coro)
        try:
            result = await self._stream_task
            # 将本次产生的新消息追加到 Agent 的消息历史中
            self._state.messages.extend(result.messages)
            self._last_run_result = result
            return result
        except asyncio.CancelledError:
            self._state.error = "aborted"
            raise
        except Exception as exc:
            self._state.error = str(exc)
            raise
        finally:
            # 无论成功或失败，都重置流式状态
            self._state.is_streaming = False
            self._state.stream_message = None
            self._stream_task = None

    async def _dispatch_event(self, event: AgentEvent) -> None:
        """分发 Agent 事件到所有监听器。

        同时根据事件类型更新内部状态：
        - message_start / message_update: 更新当前正在流式生成的消息。
        - message_end: 清除流式消息引用。
        - tool_execution_start / end: 维护待处理工具调用集合。
        - error: 记录错误信息。

        Args:
            event: 要分发的 AgentEvent 事件对象。
        """
        event_type = event.get("type")

        # 根据事件类型更新内部状态
        if event_type == "message_start":
            msg = event.get("message")
            self._state.stream_message = msg
        elif event_type == "message_update":
            self._state.stream_message = event.get("message")
        elif event_type == "message_end":
            self._state.stream_message = None
        elif event_type == "tool_execution_start":
            tool_call_id = event.get("toolCallId")
            if tool_call_id:
                self._state.pending_tool_calls.add(tool_call_id)
        elif event_type == "tool_execution_end":
            tool_call_id = event.get("toolCallId")
            if tool_call_id in self._state.pending_tool_calls:
                self._state.pending_tool_calls.remove(tool_call_id)
        elif event_type == "error":
            self._state.error = event.get("error", "unknown error")
        elif event_type == "agent_end":
            result = event.get("result")
            if result is not None and result.status == "completed":
                self._state.error = None

        # 将事件广播给所有已注册的监听器（兼容同步/异步回调）
        for listener in list(self._listeners):
            await _maybe_await(listener(event))

    async def _drain_steering_messages(self) -> list[AgentMessage]:
        """取出并清空引导消息队列（drain 模式：取一次就清空）。"""
        items = list(self._steering_queue)
        self._steering_queue.clear()
        return items

    async def _drain_follow_up_messages(self) -> list[AgentMessage]:
        """取出并清空后续消息队列（drain 模式：取一次就清空）。"""
        items = list(self._follow_up_queue)
        self._follow_up_queue.clear()
        return items
