from __future__ import annotations

# 新手导读：runtime context 汇总工作区、扩展加载结果和组装时诊断，供后续 session 创建使用。
# 关注点：它是装配过程里的共享信息包，不是 sessions/context 的上下文投影。

"""
Runtime 系统提示词启动上下文。

本模块只负责构建创建 AgentSession 时可注入系统提示词的静态启动信息：
- 仓库 bootstrap 概览；
- 配置/扩展/skills 提供的 prompt guidelines；
- 配置/扩展/skills 提供的追加系统提示词段落；
- 工具说明片段。

注意：顶层目录概览虽然会出现在初始系统提示词里，但它不是唯一事实来源。
每次模型调用前，sessions.context.ContextGovernor 会刷新 RepositorySnapshot，
并把动态仓库状态、证据和记忆召回投影到本轮 ContextView 中。
"""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from codepilot.extensions.types import LoadedExtensions
from codepilot.sessions.context.repository_context import (
    GitInfo,
    RepositoryBootstrap,
    build_repository_bootstrap,
    render_repository_context,
)

from .config import ResolvedRuntimeConfig


@dataclass(frozen=True)
class RuntimeContext:
    """供系统提示词渲染使用的启动上下文。

    RuntimeContext is built once during session assembly and then handed to the
    prompt renderer.  It copies and freezes collection fields so later mutation
    of config or extension objects cannot silently change the prompt contract.
    """

    repository_context: str
    prompt_guidelines: tuple[str, ...]
    append_sections: tuple[str, ...]
    tool_snippets: Mapping[str, str]
    memory_text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "repository_context", _strip_text(self.repository_context))
        object.__setattr__(
            self,
            "prompt_guidelines",
            tuple(_clean_text_items(self.prompt_guidelines)),
        )
        object.__setattr__(
            self,
            "append_sections",
            tuple(_clean_text_items(self.append_sections)),
        )
        object.__setattr__(
            self,
            "tool_snippets",
            MappingProxyType(_clean_tool_snippets(self.tool_snippets)),
        )
        object.__setattr__(self, "memory_text", _strip_text(self.memory_text))


def build_runtime_context(
    workspace: Path,
    config: ResolvedRuntimeConfig,
    loaded_extensions: LoadedExtensions,
    loaded_skills: LoadedExtensions,
) -> RuntimeContext:
    """构建系统提示词启动上下文。

    这里不做 active files、recent evidence、memory retrieval 或当前任务投影；
    这些动态内容由 sessions.context.ContextGovernor 在每次模型调用前处理。
    """

    prompt_guidelines = [
        *(config.prompt_guidelines or []),
        *loaded_extensions.prompt_guidelines,
        *loaded_skills.prompt_guidelines,
    ]
    prompt_guidelines.extend([f"[skill-diagnostic] {d}" for d in loaded_skills.diagnostics])

    append_sections = []
    if config.append_system_prompt:
        append_sections.append(config.append_system_prompt)
    append_sections.extend(loaded_extensions.append_prompts)
    append_sections.extend(loaded_skills.append_prompts)
    if config.prompt_debug_sources:
        append_sections.append("\n".join(_build_prompt_debug_lines(loaded_extensions, loaded_skills)))

    return RuntimeContext(
        repository_context=render_repository_context(build_repository_bootstrap(workspace)),
        prompt_guidelines=prompt_guidelines,
        append_sections=append_sections,
        tool_snippets=config.tool_snippets or {},
        memory_text="",
    )


def _build_prompt_debug_lines(
    loaded_extensions: LoadedExtensions,
    loaded_skills: LoadedExtensions,
) -> list[str]:
    """构建提示词来源调试信息（用于 prompt_debug_sources 模式）。"""

    debug_lines: list[str] = ["## Prompt Sources", "### extensions"]
    debug_lines.extend([f"- {p}" for p in loaded_extensions.loaded_paths] or ["- (none)"])
    debug_lines.append("### skills")
    debug_lines.extend([f"- {p}" for p in loaded_skills.loaded_paths] or ["- (none)"])
    if loaded_extensions.errors or loaded_skills.errors:
        debug_lines.append("### errors")
        debug_lines.extend([f"- {e}" for e in [*loaded_extensions.errors, *loaded_skills.errors]])
    if loaded_skills.diagnostics:
        debug_lines.append("### diagnostics")
        debug_lines.extend([f"- {d}" for d in loaded_skills.diagnostics])
    return debug_lines


def _strip_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _clean_text_items(values: object) -> list[str]:
    if values is None:
        return []
    return [
        text
        for text in (_strip_text(value) for value in values)
        if text
    ]


def _clean_tool_snippets(values: Mapping[str, str] | None) -> dict[str, str]:
    if not values:
        return {}
    snippets: dict[str, str] = {}
    for key, value in values.items():
        name = _strip_text(key)
        snippet = _strip_text(value)
        if name and snippet:
            snippets[name] = snippet
    return snippets


__all__ = [
    "GitInfo",
    "RepositoryBootstrap",
    "RuntimeContext",
    "build_repository_bootstrap",
    "build_runtime_context",
    "render_repository_context",
]
