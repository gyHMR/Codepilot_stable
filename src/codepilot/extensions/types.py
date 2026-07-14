"""定义扩展加载结果、钩子和能力集合的稳定类型。"""

from __future__ import annotations

# 新手导读：extensions/types.py 描述扩展加载后的统一能力集合。
# 关注点：命令和生命周期契约位于 protocols.commands，extensions 只生产能力描述。

"""扩展层类型定义：钩子、命令、技能规格和加载结果。"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codepilot.protocols.commands import LifecycleHook, RegisteredCommand
from codepilot.tools import ToolRegistration

if TYPE_CHECKING:
    from .skills import SkillPackage


@dataclass
class LoadedExtensions:
    """从扩展、技能和 MCP 配置中归一化加载的能力集合。

    扩展来源被刻意归一化为四种简单能力：
    - 工具 → 进入工具安全层
    - 命令 → 进入运行时斜杠命令注册表
    - 提示词文本 → 进入系统提示词
    - 钩子 → 进入生命周期或工具调用管道
    """

    tools: list[ToolRegistration] = field(default_factory=list)
    prompt_guidelines: list[str] = field(default_factory=list)
    append_prompts: list[str] = field(default_factory=list)
    commands: dict[str, RegisteredCommand] = field(default_factory=dict)
    before_prompt_hooks: list[LifecycleHook] = field(default_factory=list)
    after_prompt_hooks: list[LifecycleHook] = field(default_factory=list)
    skills: list["SkillPackage"] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    loaded_paths: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
