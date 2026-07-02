"""
Codepilot Agent Run 核心模块
============================

本包是 Codepilot Agent 系统的核心层，负责一次 Agent Run 的完整执行循环。

模块架构：
    ┌─────────────────────────────────────────────────────────────┐
    │                      Agent（对外入口）                       │
    │  提供 run/continue_run、状态管理、事件订阅、串行调度        │
    └─────────────────────────┬───────────────────────────────────┘
                              │
    ┌─────────────────────────▼───────────────────────────────────┐
    │                  agent_loop（核心循环）                       │
    │  用户提示 → 模型推理 → 工具执行 → 任务检查 → 返回结果       │
    └──┬──────────┬──────────┬──────────┬──────────┬──────────────┘
       │          │          │          │          │
    ┌──▼──┐   ┌──▼──┐   ┌───▼───┐  ┌───▼───┐  ┌──▼──┐
    │ LLM │   │工具 │   │任务控 │  │任务规 │  │运行 │
    │Runner│   │Coord│   │制器   │  │划器   │  │状态 │
    └─────┘   └─────┘   └───────┘  └───────┘  └─────┘

主要模块：
    - agent.py: 对外 Agent 封装，提供 run/continue_run 入口
    - agent_loop.py: 核心执行循环，协调 LLM 推理和工具执行
    - llm_runner.py: LLM 流式运行器，处理流式和非流式调用
    - tool_coordinator.py: 工具调用协调器，管理工具执行生命周期
    - task_controller.py: 任务控制器，跟踪任务进度并做出决策
    - task_planner.py: 任务规划器，使用 LLM 生成执行计划
    - task_state.py: 任务状态定义，包含步骤、尝试记录、变更集合等
    - task_tools.py: 任务控制内部工具（如 complete_task_step）
    - run_state.py: 运行状态，记录计数器、受影响路径等事实
    - run_decisions.py: 纯决策函数，将底层事实转换为运行级决策
    - types.py: 类型定义，包含 AgentContext、AgentLoopConfig 等
    - events.py: 事件发射器，为事件注入 run/turn/event 元数据
    - message_conversion.py: 消息转换，将 Agent 消息转为 LLM 格式

注意：协议事件、运行结果和工具定义由 protocols/tools 层拥有。
core 门面只导出核心编排对象，避免调用方误以为跨层数据模型属于 core。
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
from .task_modes import TaskMode, TaskModePolicy, ensure_task_mode, policy_for_mode
from .task_planner import PlannedTaskStep, TaskPlanDraft, TaskPlanner
from .task_tools import COMPLETE_TASK_STEP_TOOL
from .task_state import CompletionCheck, ExecutionDecision, TaskState, TaskStep
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
    "TaskController",
    "TaskMode",
    "TaskModePolicy",
    "ensure_task_mode",
    "policy_for_mode",
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
    "AgentLoopConfig",
    "AgentMessage",
    "AgentState",
    "BeforeToolCallContext",
    "BeforeToolCallResult",
    "ContextPreparationRequest",
    "PreparedAgentContext",
    "PrepareContextFn",
    "ToolExecutionMode",
]
