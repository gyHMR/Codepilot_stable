"""
Codepilot Agent Run 核心模块
============================

本包负责一次 Agent Run 的完整执行循环：
- Agent / AgentLoop  协调运行生命周期。
- RunState           记录本次运行的机械性执行事实（计数器、受影响路径等）。
- TaskState / TaskController  根据工具反馈跟踪任务进度。
- LLMStreamRunner    将 LLM 流式响应转换为 Agent 消息事件。
- ToolCallCoordinator  准备、执行并上报工具调用。
"""

from .agent import Agent, AgentOptions
from .agent_loop import (
    run_agent_loop,
    run_agent_loop_continue,
)
from .events import AgentEventEmitter
from .llm_runner import LLMStreamRunner, StreamFn
from .message_conversion import convert_to_llm
from .run_state import RunState, new_run_id
from .task_controller import TaskController
from .task_planner import PlannedTaskStep, TaskPlanDraft, TaskPlanner
from .task_tools import COMPLETE_TASK_STEP_TOOL
from .task_state import CompletionCheck, ExecutionDecision, TaskState, TaskStep
from .tool_coordinator import ToolCallCoordinator
from codepilot.protocols import (
    AgentEndEvent,
    AgentEvent,
    AgentEventBase,
    AgentRunCounters,
    AgentRunResult,
    AgentRunStatus,
    AgentRunStopReason,
    AgentStartEvent,
    ErrorEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from codepilot.tools import AgentTool, AgentToolResult
from .types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentLoopConfig,
    AgentMessage,
    AgentState,
    BeforeToolCallContext,
    BeforeToolCallResult,
    ContextPreparationRequest,
    PreparedAgentContext,
    PrepareContextFn,
    ToolExecutionMode,
)

__all__ = [
    "Agent",
    "AgentOptions",
    "run_agent_loop",
    "run_agent_loop_continue",
    "AgentEventEmitter",
    "LLMStreamRunner",
    "StreamFn",
    "convert_to_llm",
    "RunState",
    "new_run_id",
    "ToolCallCoordinator",
    "TaskController",
    "PlannedTaskStep",
    "TaskPlanDraft",
    "TaskPlanner",
    "COMPLETE_TASK_STEP_TOOL",
    "CompletionCheck",
    "ExecutionDecision",
    "TaskState",
    "TaskStep",
    "AfterToolCallContext",
    "AfterToolCallResult",
    "AgentContext",
    "AgentEvent",
    "AgentEventBase",
    "AgentRunCounters",
    "AgentRunResult",
    "AgentRunStatus",
    "AgentRunStopReason",
    "AgentStartEvent",
    "AgentEndEvent",
    "TurnStartEvent",
    "TurnEndEvent",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "MessageEndEvent",
    "ToolExecutionStartEvent",
    "ToolExecutionUpdateEvent",
    "ToolExecutionEndEvent",
    "ErrorEvent",
    "AgentLoopConfig",
    "AgentMessage",
    "AgentState",
    "AgentTool",
    "AgentToolResult",
    "BeforeToolCallContext",
    "BeforeToolCallResult",
    "ContextPreparationRequest",
    "PreparedAgentContext",
    "PrepareContextFn",
    "ToolExecutionMode",
]
