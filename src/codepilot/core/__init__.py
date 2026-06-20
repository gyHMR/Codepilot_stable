"""
Codepilot agent_core
===================

Minimal Agent orchestration core:
- Agent object
- Agent loop
- Tool execution protocol
- Event callback protocol
"""

from .agent import Agent, AgentOptions
from .agent_loop import (
    run_agent_loop,
    run_agent_loop_continue,
)
from .events import AgentEventEmitter
from .llm_runner import LLMStreamRunner, StreamFn
from .message_conversion import convert_to_llm
from .task_controller import TaskController
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
    "ToolCallCoordinator",
    "TaskController",
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
