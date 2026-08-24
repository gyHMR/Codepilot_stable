# 新手导读：包门面文件：集中导出本层最常用的类型和入口，降低学习时的导入成本。
# 关注点：extensions 层负责把 Python 扩展、skill 和 MCP 外部能力加载成项目内统一能力。

"""
扩展与技能加载模块。

本包负责从工作区和配置路径加载 Python 扩展和 Skill Package，
将它们的能力（工具、钩子、命令、提示词）统一归一化为 LoadedExtensions。
"""

from .api import ExtensionAPI
from .loader import discover_extension_paths, load_extensions
from .skill_runtime import load_skills
from .skills import (
    SkillCatalog,
    SkillManifest,
    SkillPackage,
    SkillPackageError,
    load_skill_catalog,
)
from codepilot.protocols.commands import (
    CommandHandler,
    CommandOutcome,
    LifecycleHook,
    RegisteredCommand,
    SessionCommandContext,
    SessionCommandView,
    SessionLifecycleContext,
    SessionLifecycleView,
)
from .types import LoadedExtensions

__all__ = [
    "CommandHandler",
    "CommandOutcome",
    "RegisteredCommand",
    "SessionCommandContext",
    "SessionCommandView",
    "SessionLifecycleContext",
    "SessionLifecycleView",
    "LifecycleHook",
    "SkillCatalog",
    "SkillManifest",
    "SkillPackage",
    "SkillPackageError",
    "LoadedExtensions",
    "ExtensionAPI",
    "discover_extension_paths",
    "load_extensions",
    "load_skill_catalog",
    "load_skills",
]
