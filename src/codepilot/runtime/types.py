from __future__ import annotations

"""
Runtime 对外类型定义模块。

定义了创建 Agent 会话所需的配置选项 CreateAgentSessionOptions，
以及运行模式、输出/输入函数等辅助类型。

配置分组（阶段二引入）：
- ModelSelection: 模型选择（provider/model_id）
- AgentPolicy: Agent 策略（thinking_level、retry 等）
- ContextPolicy: 上下文管理策略（消息数、token 数等）
- ToolPolicy: 工具策略（执行模式、权限等）

CreateAgentSessionOptions 保持向后兼容，内部可从分组构建。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Optional

from codepilot.protocols import Message, Model
from codepilot.core import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentMessage,
    BeforeToolCallContext,
    BeforeToolCallResult,
    ToolExecutionMode,
)
from codepilot.extensions.types import LifecycleHook, RegisteredCommand
from codepilot.sessions.types import AgentSessionOptions, ConvertToLlmFn
from codepilot.tools import AgentTool


# ── 配置分组（阶段二引入） ────────────────────────────────────────

@dataclass(frozen=True)
class ModelSelection:
    """模型选择配置。

    描述使用哪个模型以及如何连接。

    Attributes:
        provider: provider 名称（如 deepseek、openai）。
        model_id: 模型 ID（如 deepseek-chat）。
    """
    provider: str | None = None
    model_id: str | None = None

    @property
    def display_id(self) -> str:
        """返回显示用的模型标识（provider/model_id）。"""
        if self.provider and self.model_id:
            return f"{self.provider}/{self.model_id}"
        return self.model_id or "(not configured)"


@dataclass(frozen=True)
class AgentPolicy:
    """Agent 策略配置。

    控制 Agent 的推理、重试等行为。

    Attributes:
        thinking_level: 推理级别（off/minimal/low/medium/high/xhigh）。
        retry_enabled: 是否启用重试。
        max_retries: 最大重试次数。
        retry_base_delay_ms: 重试基础延迟（毫秒）。
    """
    thinking_level: str = "off"
    retry_enabled: bool = True
    max_retries: int = 2
    retry_base_delay_ms: int = 1200


@dataclass(frozen=True)
class ContextPolicy:
    """上下文管理策略。

    控制上下文窗口的溢出检测和压缩行为。

    Attributes:
        max_messages: 消息数量上限（None 表示不限制）。
        max_tokens: token 数量上限（None 表示不限制）。
        retain_recent_messages: 压缩时保留的最近消息数。
    """
    max_messages: int | None = None
    max_tokens: int | None = None
    retain_recent_messages: int = 24


@dataclass(frozen=True)
class ToolPolicy:
    """工具策略配置。

    控制工具的执行模式和权限。

    Attributes:
        execution: 工具执行模式（parallel/sequential）。
        permission_mode: 权限模式（read-only/workspace-write）。
        block_dangerous_bash: 是否阻止危险 bash 命令。
    """
    execution: ToolExecutionMode = "parallel"
    permission_mode: str = "workspace-write"
    block_dangerous_bash: bool = True


@dataclass
class CreateAgentSessionOptions:
    """创建 Agent 会话的友好配置选项。

    支持三种模型指定方式（按优先级）：
    1) 直接传 model 对象（最高优先级）。
    2) 传 provider + model_id（由工厂自动从内置目录解析）。
    3) 传入已有 session_id（工厂从会话元数据恢复 provider/model_id/system_prompt）。

    Attributes:
        workspace_dir: 工作区目录路径。
        model: 直接指定的模型对象（可选）。
        provider: provider 名称（可选，与 model_id 配合使用）。
        model_id: 模型 ID（可选，与 provider 配合使用）。
        get_api_key: API Key 获取函数（可选）。
        system_prompt: 系统提示词（可选，覆盖默认值）。
        tools: 额外的工具列表。
        session_id: 已有会话 ID（用于恢复会话）。
        messages: 初始消息列表。
        thinking_level: 推理思考级别。
        tool_execution: 工具执行模式（parallel/sequential）。
        load_workspace_resources: 是否加载工作区资源文件。
        enabled_builtin_tools: 启用的内置工具名称列表。
        max_context_messages: 上下文消息数量上限。
        max_context_tokens: 上下文 token 数量上限。
        retain_recent_messages: 压缩时保留的最近消息数。
        summary_builder: 自定义摘要构建器。
        retry_enabled: 是否启用重试。
        max_retries: 最大重试次数。
        retry_base_delay_ms: 重试基础延迟（毫秒）。
        read_only_mode: 是否为只读模式（禁止文件修改）。
        block_dangerous_bash: 是否阻止危险的 bash 命令。
        bash_allow_patterns: bash 命令白名单模式。
        bash_block_patterns: bash 命令黑名单模式。
        edit_require_unique_match: edit 工具是否要求唯一匹配。
        prompt_guidelines: 额外的提示词准则。
        append_system_prompt: 追加到系统提示词末尾的文本。
        tool_snippets: 工具说明片段（用于系统提示词）。
        extension_paths: 扩展加载路径列表。
        skill_paths: 技能加载路径列表。
        prompt_debug_sources: 是否在提示词中包含调试来源信息。
        mcp_servers: MCP 服务器配置列表。
        mcp_client: MCP 客户端实例。
        extension_commands: 已注册的扩展命令。
        before_prompt_hooks: 提示词执行前的生命周期钩子。
        after_prompt_hooks: 提示词执行后的生命周期钩子。
        before_tool_call: 工具调用前的拦截钩子。
        after_tool_call: 工具调用后的拦截钩子。
    """

    workspace_dir: str | Path
    model: Optional[Model] = None
    provider: Optional[str] = None
    model_id: Optional[str] = None
    get_api_key: Optional[Callable[[str], str | None | Awaitable[str | None]]] = None
    system_prompt: Optional[str] = None
    tools: list[AgentTool] = field(default_factory=list)
    session_id: Optional[str] = None
    messages: list[AgentMessage] = field(default_factory=list)
    thinking_level: Optional[str] = None
    tool_execution: Optional[ToolExecutionMode] = None
    load_workspace_resources: bool = True
    enabled_builtin_tools: Optional[list[str]] = None
    max_context_messages: Optional[int] = None
    max_context_tokens: Optional[int] = None
    retain_recent_messages: Optional[int] = None
    summary_builder: Optional[Callable[[list[Message]], str]] = None
    retry_enabled: Optional[bool] = None
    max_retries: Optional[int] = None
    retry_base_delay_ms: Optional[int] = None
    read_only_mode: Optional[bool] = None
    block_dangerous_bash: Optional[bool] = None
    bash_allow_patterns: Optional[list[str]] = None
    bash_block_patterns: Optional[list[str]] = None
    edit_require_unique_match: Optional[bool] = None
    prompt_guidelines: Optional[list[str]] = None
    append_system_prompt: Optional[str] = None
    tool_snippets: Optional[dict[str, str]] = None
    extension_paths: Optional[list[str]] = None
    skill_paths: Optional[list[str]] = None
    prompt_debug_sources: Optional[bool] = None
    mcp_servers: Optional[list[dict[str, Any]]] = None
    mcp_client: Any | None = None
    extension_commands: dict[str, RegisteredCommand] = field(default_factory=dict)
    before_prompt_hooks: list[LifecycleHook] = field(default_factory=list)
    after_prompt_hooks: list[LifecycleHook] = field(default_factory=list)
    before_tool_call: Optional[
        Callable[[BeforeToolCallContext, Any | None], BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None]]
    ] = None
    after_tool_call: Optional[
        Callable[[AfterToolCallContext, Any | None], AfterToolCallResult | None | Awaitable[AfterToolCallResult | None]]
    ] = None


# 运行模式：print（单次输出）、interactive（交互式）、rpc（远程调用）
RunMode = Literal["print", "interactive", "rpc"]
# 输出函数类型：接收文本并输出（如打印到终端）
OutputFn = Callable[[str], None]
# 输入函数类型：接收提示文本并返回用户输入
InputFn = Callable[[str], str]
