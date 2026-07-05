from __future__ import annotations

# ---- assembly input DTO ----

# 新手导读：assembly_input.py 定义 runtime 装配阶段的内部输入。
# 关注点：interfaces 只能提交 SessionOpenIntent；这里的类型只供 runtime assembly/bootstrap 使用。

"""Runtime assembly input types."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Optional

from codepilot.core.task import PlanningBudgetProfile, TaskMode
from codepilot.core.contracts import (
    AgentMessage,
    ToolExecutionMode,
)
from codepilot.llm.provider_types import ProviderSimpleStreamFn
from codepilot.protocols import Model
from codepilot.protocols.commands import (
    AfterToolCallContext,
    AfterToolCallResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    LifecycleHook,
    RegisteredCommand,
)
from codepilot.tools import AgentTool
from codepilot.tools.policy import ApprovalProvider


@dataclass
class RuntimeAssemblyIntent:
    """Runtime-internal intent for assembling a runnable session.

    `SessionOpenIntent` is the public application intent. This richer type is
    produced inside runtime and includes executable tools, hooks, model
    overrides, and resource-loading controls needed by the assembly pipeline.
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
    memory_enabled: bool = True
    task_control_enabled: bool = True
    task_mode: TaskMode | None = None
    planning_budget_profile: PlanningBudgetProfile | None = None
    max_task_replans_per_run: Optional[int] = None
    load_workspace_resources: bool = True
    enabled_builtin_tools: Optional[list[str]] = None
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
        Callable[
            [BeforeToolCallContext, Any | None],
            BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None],
        ]
    ] = None
    after_tool_call: Optional[
        Callable[
            [AfterToolCallContext, Any | None],
            AfterToolCallResult | None | Awaitable[AfterToolCallResult | None],
        ]
    ] = None
    stream_fn: ProviderSimpleStreamFn | None = None


RuntimePermissionMode = Literal["read-only", "workspace-write", "ask"]
