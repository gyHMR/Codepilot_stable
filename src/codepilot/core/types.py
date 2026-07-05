from __future__ import annotations

# 新手导读：core/types.py 定义 AgentContext 和上下文准备契约。
# 关注点：工具 hook 协议属于 protocols.tool_hooks，core 不再替扩展层拥有 hook DTO。

"""
agent_core 的类型定义模块
========================

本模块是 Codepilot core 层的轻量类型基础，定义上下文快照和上下文准备契约。

设计原则：
    这一层关注"编排"而不是"具体 provider 实现"：
    1) 定义上下文结构（AgentContext）；
    2) 定义模型调用前上下文准备的输入输出；
    3) 引用 protocols 拥有的跨层类型，保持依赖方向清晰。

主要类型：
    - AgentContext: Agent 执行上下文，包含系统提示词、消息列表、工具列表等
    - AgentMessage: 消息类型别名，当前等同于 Message

辅助函数：
    - _ensure_*: 类型校验函数，确保数据类字段在赋值时类型正确
    - _copy_*: 防御性拷贝函数，避免外部修改影响内部状态
    - _clean_* / _optional_*: 文本清理函数
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from codepilot.protocols import (
    AssistantMessage,
    ContextReport,
    Message,
    Tool,
    ToolResultMessage,
    UserMessage,
)


# ── 类型别名与常量 ──────────────────────────────────────────────

# 工具执行模式：串行（sequential）或并行（parallel）
ToolExecutionMode = Literal["sequential", "parallel"]

# 消息类型别名：当前阶段只支持 LLM 消息类型，后续可以扩展 custom message
AgentMessage = Message


@dataclass
class AgentContext:
    """Agent 执行上下文：在一次 Run 的生命周期内，封装所有 LLM 调用所需的数据。

    这是 Agent 核心层与 LLM 之间的数据桥梁，每次调用 LLM 前都会构建一个
    AgentContext 快照，确保循环过程中外部修改不会影响正在进行的推理。

    Attributes:
        system_prompt: 系统提示词，定义 Agent 的角色和行为准则。
            示例: "你是一个有帮助的编程助手。"
        messages: 消息列表，包含历史对话和本次运行的消息。
            消息类型包括: UserMessage（用户消息）、AssistantMessage（助手消息）、
            ToolResultMessage（工具执行结果）。
        tools: 可用工具描述列表，定义 Agent 可以调用的工具及其参数格式。
            每个 Tool 只包含 name、description、parameters，不含执行器。
        current_task: 当前任务上下文（Markdown 格式），由 TaskController 渲染。
            包含任务目标、步骤进度、验收标准等信息，注入系统提示词供 LLM 参考。
        task_recovery_projection: 会话持久化的任务恢复投影。
            当会话中断后恢复时，用于重建 TaskState，避免丢失任务进度。
        task_signal: 任务控制信号，由 TaskController 输出。
            包含当前步骤、阶段、下一步动作等轻量级信息，供上下文和记忆模块使用。
    """
    system_prompt: str                                    # 系统提示词
    messages: list[AgentMessage]                          # 消息列表（历史 + 本次）
    tools: list[Tool] = field(default_factory=list)       # 可用工具描述列表
    current_task: str | None = None                       # 当前任务上下文（Markdown）
    task_recovery_projection: dict[str, object] | None = None  # 任务恢复投影
    task_signal: dict[str, object] | None = None          # 任务控制信号

    def __post_init__(self) -> None:
        """初始化后校验：对所有字段进行类型检查和防御性拷贝。"""
        # 清理系统提示词（确保为字符串）
        self.system_prompt = _clean_core_text(self.system_prompt)
        # 拷贝消息列表（避免外部引用共享可变对象）
        self.messages = _copy_messages(self.messages, field_name="messages")
        # 拷贝工具列表
        self.tools = _copy_tools(self.tools, field_name="tools")
        # 清理可选文本字段
        self.current_task = _optional_core_text(self.current_task)
        # 深拷贝可选字典（避免外部修改影响内部状态）
        self.task_recovery_projection = _copy_optional_dict(
            self.task_recovery_projection,
            field_name="task_recovery_projection",
        )
        self.task_signal = _copy_optional_dict(self.task_signal, field_name="task_signal")


@dataclass(frozen=True)
class ContextPreparationRequest:
    """上下文准备请求：携带模型能力信息，供 prepare_context 回调使用。

    prepare_context 回调可以根据这些信息决定如何裁剪消息列表、
    是否需要压缩上下文等。

    Attributes:
        session_id: 当前会话标识符。
        model_context_window: 模型的上下文窗口大小（token 数）。
        model_max_output_tokens: 模型的最大输出 token 数。
        signal: 可选的取消信号，用于中断长时间的上下文准备操作。
    """
    session_id: str | None                      # 会话标识符
    model_context_window: int                   # 模型上下文窗口大小
    model_max_output_tokens: int                # 模型最大输出 token 数
    signal: Any | None = None                   # 取消信号


@dataclass
class PreparedAgentContext:
    """已准备的上下文：prepare_context 回调的返回值。

    经过上下文准备后，系统提示词、消息列表和工具列表可能已经被裁剪或变换，
    同时附带一份报告（ContextReport）说明做了哪些处理。

    Attributes:
        system_prompt: 处理后的系统提示词。
        messages: 处理后的消息列表（可能已裁剪或压缩）。
        tools: 处理后的工具列表。
        report: 上下文准备报告，记录裁剪了多少消息、节省了多少 token 等。
    """
    system_prompt: str                          # 处理后的系统提示词
    messages: list[AgentMessage]                # 处理后的消息列表
    tools: list[Tool]                           # 处理后的工具描述列表
    report: ContextReport                       # 上下文准备报告


# 上下文准备函数类型：接收原始上下文和准备请求，返回准备后的上下文
# 支持同步和异步两种调用方式
PrepareContextFn = Callable[
    [AgentContext, ContextPreparationRequest],
    PreparedAgentContext | Awaitable[PreparedAgentContext],
]


# ── 文本清理辅助函数 ─────────────────────────────────────────────

def _clean_core_text(value: object) -> str:
    """将任意值转换为字符串，None 转为空字符串。"""
    return str(value) if value is not None else ""


def _optional_core_text(value: object) -> str | None:
    """将任意值转换为可选字符串：空字符串返回 None。"""
    text = _clean_core_text(value).strip()
    return text or None


# ── 防御性拷贝辅助函数 ──────────────────────────────────────────

def _copy_messages(value: object, *, field_name: str) -> list[AgentMessage]:
    """拷贝消息列表：校验每个元素的类型，返回新的列表副本。"""
    if not isinstance(value, list):
        raise TypeError(f"AgentContext {field_name} must be a list")
    messages: list[AgentMessage] = []
    for message in value:
        if not isinstance(message, (UserMessage, AssistantMessage, ToolResultMessage)):
            raise TypeError(f"AgentContext {field_name} entries must be AgentMessage")
        messages.append(message)
    return messages


def _copy_tools(value: object, *, field_name: str) -> list[Tool]:
    """拷贝工具描述列表：校验每个元素的类型，返回新的列表副本。"""
    if not isinstance(value, list):
        raise TypeError(f"AgentContext {field_name} must be a list")
    tools: list[Tool] = []
    for tool in value:
        if not isinstance(tool, Tool):
            raise TypeError(f"AgentContext {field_name} entries must be Tool")
        tools.append(tool)
    return tools


def _copy_optional_dict(
    value: object,
    *,
    field_name: str,
) -> dict[str, object] | None:
    """深拷贝可选字典：None 返回 None，否则返回深拷贝副本。"""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"AgentContext {field_name} must be a dict or None")
    return deepcopy(value)


# ── 类型校验辅助函数 ─────────────────────────────────────────────
# 这些函数用于 dataclass 的 __post_init__ 中，确保字段值的类型正确。
# 采用防御式编程：在赋值时就捕获类型错误，而不是等到使用时才发现。

def _ensure_optional_callable(value: object, field_name: str) -> None:
    """校验可选的可调用对象：None 或可调用对象。"""
    if value is not None and not callable(value):
        raise TypeError(f"{field_name} must be callable or None")
