"""
Session 层的类型定义模块。

本模块定义了 AgentSession 的构造参数类型，将会话编排相关的类型
保留在 sessions 层，使其不依赖 runtime 装配层，实现了良好的分层架构。

主要类型:
    - ConvertToLlmFn: 消息转换函数类型，用于将内部消息格式转换为 LLM 可理解的格式
    - AgentSessionOptions: 会话的完整配置选项，包含所有初始化参数

设计原则:
    - 类型定义与实现分离
    - 通过 dataclass 提供清晰的参数结构
    - 使用 Optional 和默认值提供灵活的配置选项
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from codepilot.core import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentMessage,
    BeforeToolCallContext,
    BeforeToolCallResult,
    PrepareContextFn,
    StreamFn,
    ToolExecutionMode,
)
from codepilot.extensions.types import LifecycleHook, RegisteredCommand
from codepilot.protocols import Message, Model
from codepilot.tools import AgentTool

# 消息转换函数类型定义
# 功能: 将内部的 AgentMessage 列表转换为 LLM API 所需的 Message 列表
# 支持同步和异步两种调用方式
# 使用场景: 在发送消息给 LLM 之前，对消息进行格式转换或预处理
ConvertToLlmFn = Callable[[list[AgentMessage]], list[Message] | Awaitable[list[Message]]]


@dataclass
class AgentSessionOptions:
    """
    AgentSession 的底层初始化参数配置类。

    该数据类由 runtime 装配层构造，包含了创建和配置一个 Agent 会话
    所需的所有参数。通过 dataclass 装饰器，自动生成 __init__、__repr__
    等方法，提供了清晰的参数结构。

    属性分类:
        1. 核心配置 (model, workspace_dir, system_prompt)
        2. 工具相关 (tools, tool_execution, max_tool_calls_per_turn)
        3. 消息管理 (messages, convert_to_llm, max_context_messages)
        4. 上下文控制 (max_context_tokens, retain_recent_messages)
        5. 重试机制 (retry_enabled, max_retries, retry_base_delay_ms)
        6. 扩展功能 (extension_commands, lifecycle hooks)
        7. 流式处理 (stream_fn, prepare_context)

    典型使用:
        options = AgentSessionOptions(
            model=my_model,
            workspace_dir="/path/to/workspace",
            system_prompt="你是一个有帮助的助手",
            tools=[...],
        )
        session = AgentSession(options)
    """

    # ==================== 核心配置 ====================

    # LLM 模型实例，用于与语言模型 API 通信
    model: Model

    # 工作区目录路径，Agent 将在此目录下执行文件操作等任务
    # 支持字符串或 Path 对象
    workspace_dir: str | Path

    # 系统提示词，定义 Agent 的角色和行为规范
    # 默认为空字符串，表示不设置特定的系统提示
    system_prompt: str = ""

    # ==================== 工具相关 ====================

    # Agent 可使用的工具列表
    # 每个工具都是 AgentTool 的实例，定义了工具的名称、描述和执行逻辑
    tools: list[AgentTool] = field(default_factory=list)

    # 会话的唯一标识符
    # 如果不提供，系统会自动生成一个新的会话 ID
    session_id: Optional[str] = None

    # ==================== 消息管理 ====================

    # 初始消息列表，用于恢复历史会话或预设对话上下文
    # 默认为空列表，表示从头开始新会话
    messages: list[AgentMessage] = field(default_factory=list)

    # 思维级别配置，控制 LLM 的推理深度
    # 可选值: "off" (关闭), "low" (低), "medium" (中), "high" (高)
    thinking_level: str = "off"

    # 工具执行模式
    # "parallel": 并行执行多个工具调用（默认，性能更好）
    # "sequential": 顺序执行工具调用（更可控）
    tool_execution: ToolExecutionMode = "parallel"

    # 每轮对话中允许的最大工具调用次数
    # 防止 Agent 在单轮中执行过多工具调用，默认为 8
    max_tool_calls_per_turn: int = 8
    context_governance_enabled: bool = True
    memory_enabled: bool = True
    task_control_enabled: bool = True
    task_planner_enabled: bool = True
    max_task_replans_per_run: int = 2

    # 消息转换函数
    # 在发送消息给 LLM 之前，将内部 AgentMessage 格式转换为 LLM Message 格式
    # 如果为 None，则使用默认的转换逻辑
    convert_to_llm: Optional[ConvertToLlmFn] = None

    # API 密钥获取函数
    # 接受 provider 名称，返回对应的 API 密钥
    # 支持同步和异步调用，返回 None 表示未找到密钥
    get_api_key: Optional[Callable[[str], str | None | Awaitable[str | None]]] = None

    # ==================== 上下文控制 ====================

    # 上下文中保留的最大消息数量
    # 当消息数量超过此限制时，旧消息会被压缩或移除
    # None 表示不限制消息数量
    max_context_messages: Optional[int] = None

    # 上下文的最大 token 数量
    # 用于控制发送给 LLM 的上下文长度，避免超出模型限制
    # None 表示不限制 token 数量
    max_context_tokens: Optional[int] = None

    # 上下文压缩时保留的最近消息数量
    # 确保最近的对话不会被压缩，默认保留 24 条消息
    retain_recent_messages: int = 24

    # 摘要构建函数
    # 用于将旧消息压缩为摘要，减少上下文长度
    # 接受消息列表，返回摘要字符串
    summary_builder: Optional[Callable[[list[Message]], str]] = None

    # ==================== 重试机制 ====================

    # 是否启用重试机制
    # 当 LLM 调用失败时自动重试，默认启用
    retry_enabled: bool = True

    # 最大重试次数
    # 达到此次数后停止重试并抛出异常，默认为 2
    max_retries: int = 2

    # 重试基础延迟时间（毫秒）
    # 实际延迟会采用指数退避策略，默认为 1200ms
    retry_base_delay_ms: int = 1200

    # ==================== 扩展功能 ====================

    # 扩展命令注册表
    # 键为命令名称，值为 RegisteredCommand 实例
    # 用于注册自定义的斜杠命令（如 /help, /clear 等）
    extension_commands: dict[str, RegisteredCommand] = field(default_factory=dict)

    # 提示词执行前的生命周期钩子列表
    # 在用户输入提示词后、LLM 处理前执行
    before_prompt_hooks: list[LifecycleHook] = field(default_factory=list)

    # 提示词执行后的生命周期钩子列表
    # 在 LLM 响应完成后执行
    after_prompt_hooks: list[LifecycleHook] = field(default_factory=list)

    # 工具调用前的回调函数
    # 在执行工具之前调用，可以用于权限检查、参数验证等
    # 返回 BeforeToolCallResult 可以修改或取消工具调用
    before_tool_call: Optional[
        Callable[
            [BeforeToolCallContext, Any | None],
            BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None],
        ]
    ] = None

    # 工具调用后的回调函数
    # 在工具执行完成后调用，可以用于结果处理、日志记录等
    # 返回 AfterToolCallResult 可以修改工具执行结果
    after_tool_call: Optional[
        Callable[
            [AfterToolCallContext, Any | None],
            AfterToolCallResult | None | Awaitable[AfterToolCallResult | None],
        ]
    ] = None

    # ==================== 流式处理 ====================

    # 流式输出函数
    # 用于实时输出 LLM 的生成内容，支持流式显示
    stream_fn: StreamFn | None = None

    # 上下文准备函数
    # 在发送给 LLM 之前，对编译好的上下文进行最后的处理
    prepare_context: PrepareContextFn | None = None
