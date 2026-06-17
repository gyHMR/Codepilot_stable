from __future__ import annotations

"""Session-owned type definitions.

This module keeps long-lived AgentSession construction types in the
sessions layer so session orchestration does not depend on runtime assembly.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from codepilot.core import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentMessage,
    AgentTool,
    BeforeToolCallContext,
    BeforeToolCallResult,
    ToolExecutionMode,
)
from codepilot.extensions.types import LifecycleHook, RegisteredCommand
from codepilot.llm.types import Message, Model

ConvertToLlmFn = Callable[[list[AgentMessage]], list[Message] | Awaitable[list[Message]]]


@dataclass
class AgentSessionOptions:
    """Low-level AgentSession initialization parameters."""

    model: Model
    workspace_dir: str | Path
    system_prompt: str = ""
    tools: list[AgentTool] = field(default_factory=list)
    session_id: Optional[str] = None
    messages: list[AgentMessage] = field(default_factory=list)
    thinking_level: str = "off"
    tool_execution: ToolExecutionMode = "parallel"
    convert_to_llm: Optional[ConvertToLlmFn] = None
    max_context_messages: Optional[int] = None
    max_context_tokens: Optional[int] = None
    retain_recent_messages: int = 24
    summary_builder: Optional[Callable[[list[Message]], str]] = None
    retry_enabled: bool = True
    max_retries: int = 2
    retry_base_delay_ms: int = 1200
    read_only_mode: bool = False
    block_dangerous_bash: bool = True
    bash_allow_patterns: Optional[list[str]] = None
    bash_block_patterns: Optional[list[str]] = None
    edit_require_unique_match: bool = True
    prompt_guidelines: Optional[list[str]] = None
    append_system_prompt: Optional[str] = None
    tool_snippets: Optional[dict[str, str]] = None
    extension_paths: Optional[list[str]] = None
    skill_paths: Optional[list[str]] = None
    prompt_debug_sources: bool = False
    mcp_servers: Optional[list[dict[str, Any]]] = None
    mcp_client: Any | None = None
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
