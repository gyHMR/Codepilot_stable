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
from .agent_loop import run_agent_loop, run_agent_loop_continue
from .events import AgentEventEmitter
from .llm_runner import LLMStreamRunner
from .tool_coordinator import ToolCallCoordinator
from .types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentEvent,
    AgentEventBase,
    AgentStartEvent,
    AgentEndEvent,
    TurnStartEvent,
    TurnEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    MessageEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolExecutionEndEvent,
    ErrorEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentState,
    AgentTool,
    AgentToolResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    ToolExecutionMode,
)

__all__ = [
    "Agent",
    "AgentOptions",
    "run_agent_loop",
    "run_agent_loop_continue",
    "AgentEventEmitter",
    "LLMStreamRunner",
    "ToolCallCoordinator",
    "AfterToolCallContext",
    "AfterToolCallResult",
    "AgentContext",
    "AgentEvent",
    "AgentEventBase",
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
    "ToolExecutionMode",
]
