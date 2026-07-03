from __future__ import annotations

# 新手导读：extensions/types.py 描述扩展加载后的统一能力集合。
# 关注点：命令和生命周期契约已经下沉到 sessions.types，避免 sessions 依赖 extensions。

"""扩展层类型定义：钩子、命令、技能规格和加载结果。"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from codepilot.core.types import (
    AfterToolCallContext,
    AfterToolCallResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
)
from codepilot.tools import AgentTool

if TYPE_CHECKING:
    from codepilot.sessions.types import LifecycleHook, RegisteredCommand

# 工具调用前钩子类型
BeforeHook = Callable[[BeforeToolCallContext, Any | None], BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None]]
# 工具调用后钩子类型
AfterHook = Callable[[AfterToolCallContext, Any | None], AfterToolCallResult | None | Awaitable[AfterToolCallResult | None]]


@dataclass
class SkillSpec:
    """技能规格：从 Markdown 技能文件解析出的结构化信息。"""
    name: str             # 技能名称
    command_name: str     # 对应的命令名
    description: str      # 技能描述
    content: str          # 技能内容（Markdown 正文）
    source_path: str      # 源文件路径


@dataclass
class LoadedExtensions:
    """从扩展、技能和 MCP 配置中归一化加载的能力集合。

    扩展来源被刻意归一化为四种简单能力：
    - 工具 → 进入工具安全层
    - 命令 → 进入运行时斜杠命令注册表
    - 提示词文本 → 进入系统提示词
    - 钩子 → 进入生命周期或工具调用管道
    """

    tools: list[AgentTool] = field(default_factory=list)
    before_tool_hooks: list[BeforeHook] = field(default_factory=list)
    after_tool_hooks: list[AfterHook] = field(default_factory=list)
    prompt_guidelines: list[str] = field(default_factory=list)
    append_prompts: list[str] = field(default_factory=list)
    commands: dict[str, "RegisteredCommand"] = field(default_factory=dict)
    before_prompt_hooks: list["LifecycleHook"] = field(default_factory=list)
    after_prompt_hooks: list["LifecycleHook"] = field(default_factory=list)
    skills: list[SkillSpec] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    loaded_paths: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
