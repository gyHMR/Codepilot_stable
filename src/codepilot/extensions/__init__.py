"""
扩展与技能加载模块。

本包负责从工作区和配置路径加载 Python 扩展和 Markdown 技能文件，
将它们的能力（工具、钩子、命令、提示词）统一归一化为 LoadedExtensions。
"""

from .api import ExtensionAPI
from .loader import discover_extension_paths, load_extensions
from .skills import discover_skill_paths, load_skills
from .types import (
    AfterHook,
    BeforeHook,
    CommandHandler,
    ExtensionCommandContext,
    ExtensionLifecycleContext,
    LifecycleHook,
    LoadedExtensions,
    RegisteredCommand,
    SkillSpec,
)

__all__ = [
    "BeforeHook",
    "AfterHook",
    "CommandHandler",
    "RegisteredCommand",
    "ExtensionCommandContext",
    "ExtensionLifecycleContext",
    "LifecycleHook",
    "SkillSpec",
    "LoadedExtensions",
    "ExtensionAPI",
    "discover_extension_paths",
    "load_extensions",
    "discover_skill_paths",
    "load_skills",
]
