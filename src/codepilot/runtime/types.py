from __future__ import annotations

"""
Runtime 对外类型定义模块。

定义了创建 Agent 会话所需的配置选项 CreateAgentSessionOptions，
以及运行模式、输出/输入函数等辅助类型。

装配产物（Runtime 装配优化引入）：
- RuntimeDiagnostic: 装配诊断信息
- ResolvedRuntimeProfile: 统一运行时配置
- CapabilityCatalog: 能力目录
- RuntimeAssembly: 完整装配产物
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal, Optional

from codepilot.protocols import Message, Model
from codepilot.core import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentMessage,
    BeforeToolCallContext,
    BeforeToolCallResult,
    StreamFn,
    ToolExecutionMode,
)
from codepilot.extensions.types import LifecycleHook, RegisteredCommand
from codepilot.sessions.types import AgentSessionOptions, ConvertToLlmFn
from codepilot.tools import AgentTool, ToolMetadata
from codepilot.tools.approval import ApprovalProvider

if TYPE_CHECKING:
    from codepilot.sessions.repository_context import RepositoryBootstrap
    from codepilot.sessions.session import AgentSession


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
        stream_fn: Session 级 LLM 流函数，主要用于确定性 Harness Eval。
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
    max_tool_calls_per_turn: Optional[int] = None
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
    tool_permission_mode: Optional[Literal["read-only", "workspace-write", "ask"]] = None
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
    approval_provider: ApprovalProvider | None = None
    shell_timeout_seconds: Optional[int] = None
    shell_max_timeout_seconds: Optional[int] = None
    shell_stdout_limit: Optional[int] = None
    shell_stderr_limit: Optional[int] = None
    shell_allowed_env: Optional[list[str]] = None
    extension_commands: dict[str, RegisteredCommand] = field(default_factory=dict)
    before_prompt_hooks: list[LifecycleHook] = field(default_factory=list)
    after_prompt_hooks: list[LifecycleHook] = field(default_factory=list)
    before_tool_call: Optional[
        Callable[[BeforeToolCallContext, Any | None], BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None]]
    ] = None
    after_tool_call: Optional[
        Callable[[AfterToolCallContext, Any | None], AfterToolCallResult | None | Awaitable[AfterToolCallResult | None]]
    ] = None
    stream_fn: StreamFn | None = None


# 运行模式：print（单次输出）、interactive（交互式）、rpc（远程调用）
RunMode = Literal["print", "interactive", "rpc"]
# 输出函数类型：接收文本并输出（如打印到终端）
OutputFn = Callable[[str], None]
# 输入函数类型：接收提示文本并返回用户输入
InputFn = Callable[[str], str]


# ── 装配产物类型（Runtime 装配优化引入） ──────────────────────────

@dataclass(frozen=True)
class RuntimeDiagnostic:
    """装配诊断信息。

    记录装配过程中的警告、错误和信息，用于 CLI 启动展示和调试。

    Attributes:
        severity: 严重级别（info/warning/error）。
        code: 诊断代码（如 tool.name_conflict）。
        message: 诊断消息。
        source: 来源（可选）。
    """
    severity: Literal["info", "warning", "error"]
    code: str
    message: str
    source: str | None = None


@dataclass(frozen=True)
class ConfigValueSource:
    """配置值来源。

    Attributes:
        kind: 来源类型（cli/session/project/user/default）。
        location: 具体位置（可选，如文件路径）。
    """
    kind: str
    location: str | None = None


@dataclass(frozen=True)
class ResolvedRuntimeProfile:
    """统一运行时配置。

    汇总所有配置的最终值和来源，供 CLI status/config explain 查询。

    Attributes:
        model: 解析后的模型对象。
        credential_source: 凭证来源（env/local-file/missing）。
        credential_location: 凭证位置（可选，如环境变量名）。
        permission_mode: 工具权限模式。
        sources: 每个配置项的来源。
    """
    model: Model
    credential_source: str
    credential_location: str | None = None
    permission_mode: Literal["read-only", "workspace-write", "ask"] = "workspace-write"
    sources: dict[str, ConfigValueSource] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedConfigValue:
    """供接口层展示的单个已解析配置值。"""

    key: str
    value: Any
    source: ConfigValueSource


@dataclass(frozen=True)
class RegisteredTool:
    """已注册的工具信息。

    Attributes:
        name: 工具名称。
        tool: 工具对象。
        metadata: 工具执行元数据。
        source: 来源（builtin/caller/extension/mcp）。
        origin: 具体来源（如扩展路径）。
    """
    name: str
    tool: AgentTool
    metadata: ToolMetadata | None
    source: str
    origin: str | None = None


@dataclass
class CapabilityCatalog:
    """能力目录。

    统一表示工具和命令。

    Attributes:
        tools: 已注册的工具列表。
        commands: 已注册的命令字典。
    """
    tools: list[RegisteredTool] = field(default_factory=list)
    commands: dict[str, RegisteredCommand] = field(default_factory=dict)


@dataclass
class RuntimeAssembly:
    """完整装配产物。

    保存装配过程的所有结果，供 RuntimeService 和 CLI 查询。

    Attributes:
        session_options: 最终的会话选项。
        profile: 统一运行时配置。
        repository: 仓库引导信息。
        capabilities: 能力目录。
        diagnostics: 装配诊断列表。
    """
    session_options: AgentSessionOptions
    profile: ResolvedRuntimeProfile
    repository: RepositoryBootstrap
    capabilities: CapabilityCatalog
    diagnostics: list[RuntimeDiagnostic] = field(default_factory=list)


@dataclass(frozen=True)
class SessionHandle:
    """RuntimeService 创建并注册的会话句柄。"""

    session_id: str
    session: AgentSession
    assembly: RuntimeAssembly


@dataclass(frozen=True)
class UserInput:
    """发送给会话的用户输入。"""

    text: str
    images: list[str] | None = None


@dataclass(frozen=True)
class SessionStatus:
    """供接口层展示的会话状态快照。"""

    session_id: str
    model_id: str
    workspace: str
    permission_mode: str
    message_count: int
    leaf_id: str
    is_running: bool = False
    credential_source: str = "unknown"
    warnings: list[str] | None = None


@dataclass
class ActiveRun:
    """RuntimeService 内部的活动运行记录。"""

    run_id: str
    session_id: str
    task: asyncio.Task[Any] | None = None
    started_at: float = 0
    status: Literal["running", "completed", "failed", "aborted"] = "running"
