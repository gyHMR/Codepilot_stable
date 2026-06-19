from __future__ import annotations

"""
运行时上下文构建模块。

为系统提示词渲染提供运行时上下文信息，包括：
- 仓库基本信息（项目类型、目录结构、Git 状态等）
- 提示词准则（来自配置和扩展）
- 追加段落（来自配置和扩展）
- 工具说明片段
- 长期记忆文本
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

from codepilot.extensions.types import LoadedExtensions
from codepilot.sessions.memory import load_global_memory

from .config import ResolvedRuntimeConfig

# 清单文件名 -> 项目类型的映射
_MANIFEST_PROJECT_TYPES = {
    "pyproject.toml": "Python",
    "package.json": "JavaScript/TypeScript",
    "Cargo.toml": "Rust",
    "go.mod": "Go",
    "pom.xml": "Java",
}
# 测试目录名称集合
_TEST_DIR_NAMES = {"test", "tests", "spec", "specs"}
# 顶层目录列表的最大条目数
_TOP_LEVEL_LIMIT = 30


@dataclass(frozen=True)
class RepositoryBootstrap:
    """仓库引导信息（从工作区目录扫描得到的静态信息）。

    Attributes:
        workspace_root: 工作区根目录的绝对路径。
        project_type: 项目类型（如 "Python"、"JavaScript/TypeScript"），无法识别时为 None。
        manifest_files: 存在的清单文件名列表（如 ["pyproject.toml"]）。
        top_level_entries: 顶层目录和文件列表（目录带 "/" 后缀）。
        test_directories: 测试目录列表。
        git_branch: 当前 Git 分支名，非 Git 仓库时为 None。
        git_dirty: 工作区是否有未提交变更，非 Git 仓库时为 None。
    """

    workspace_root: str
    project_type: str | None
    manifest_files: list[str]
    top_level_entries: list[str]
    test_directories: list[str]
    git_branch: str | None
    git_dirty: bool | None


@dataclass(frozen=True)
class RuntimeContext:
    """运行时上下文（供系统提示词渲染使用）。

    Attributes:
        repository_context: 渲染后的仓库上下文文本。
        prompt_guidelines: 提示词准则列表。
        append_sections: 追加到系统提示词末尾的段落列表。
        tool_snippets: 工具说明片段字典（工具名 -> 说明文本）。
        memory_text: 长期记忆文本。
    """

    repository_context: str
    prompt_guidelines: list[str]
    append_sections: list[str]
    tool_snippets: dict[str, str]
    memory_text: str


def build_repository_bootstrap(workspace: Path) -> RepositoryBootstrap:
    """扫描工作区目录，构建仓库引导信息。

    检测项目类型（通过清单文件）、目录结构、测试目录和 Git 状态。

    Args:
        workspace: 工作区目录路径。

    Returns:
        RepositoryBootstrap 对象，包含仓库的静态信息。
    """
    root = workspace.resolve()
    entries = _top_level_entries(root)
    manifest_files = [name for name in _MANIFEST_PROJECT_TYPES if (root / name).is_file()]
    project_type = _project_type(manifest_files)
    test_directories = [
        entry
        for entry in entries
        if entry.endswith("/") and entry[:-1] in _TEST_DIR_NAMES
    ]
    git_branch, git_dirty = _git_status(root)
    return RepositoryBootstrap(
        workspace_root=str(root).replace("\\", "/"),
        project_type=project_type,
        manifest_files=manifest_files,
        top_level_entries=entries,
        test_directories=test_directories,
        git_branch=git_branch,
        git_dirty=git_dirty,
    )


def render_repository_context(bootstrap: RepositoryBootstrap) -> str:
    """将仓库引导信息渲染为 Markdown 格式的上下文文本。

    Args:
        bootstrap: 仓库引导信息。

    Returns:
        Markdown 格式的仓库上下文字符串。
    """
    project_type = bootstrap.project_type or "unknown"
    manifests = ", ".join(bootstrap.manifest_files) if bootstrap.manifest_files else "(none)"
    top_level = ", ".join(bootstrap.top_level_entries) if bootstrap.top_level_entries else "(empty)"
    tests = ", ".join(bootstrap.test_directories) if bootstrap.test_directories else "(none)"
    branch = bootstrap.git_branch or "unknown"
    dirty = "unknown" if bootstrap.git_dirty is None else ("modified" if bootstrap.git_dirty else "clean")
    return "\n".join(
        [
            "## Repository Context",
            f"- Workspace: {bootstrap.workspace_root}",
            f"- Project type: {project_type}",
            f"- Manifests: {manifests}",
            f"- Top-level: {top_level}",
            f"- Test directories: {tests}",
            f"- Git branch: {branch}",
            f"- Working tree: {dirty}",
        ]
    )


def build_runtime_context(
    workspace: Path,
    config: ResolvedRuntimeConfig,
    loaded_extensions: LoadedExtensions,
    loaded_skills: LoadedExtensions,
) -> RuntimeContext:
    """构建完整的运行时上下文。

    合并来自配置、扩展和技能的提示词准则、追加段落、工具片段和记忆文本。

    Args:
        workspace: 工作区目录路径。
        config: 已解析的运行时配置。
        loaded_extensions: 已加载的扩展。
        loaded_skills: 已加载的技能。

    Returns:
        RuntimeContext 对象，包含系统提示词渲染所需的所有上下文信息。
    """
    # 合并提示词准则：配置 + 扩展 + 技能
    prompt_guidelines = [
        *(config.prompt_guidelines or []),
        *loaded_extensions.prompt_guidelines,
        *loaded_skills.prompt_guidelines,
    ]
    prompt_guidelines.extend([f"[skill-diagnostic] {d}" for d in loaded_skills.diagnostics])

    # 合并追加段落：配置 + 扩展 + 技能 + 调试来源
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
        memory_text=load_global_memory(workspace),
    )


def _top_level_entries(root: Path) -> list[str]:
    """获取工作区根目录的顶层文件和目录列表（最多 _TOP_LEVEL_LIMIT 条）。"""
    if not root.exists() or not root.is_dir():
        return []
    items = sorted(root.iterdir(), key=lambda item: item.name.lower())
    entries = []
    for item in items[:_TOP_LEVEL_LIMIT]:
        entries.append(f"{item.name}/" if item.is_dir() else item.name)
    return entries


def _project_type(manifest_files: list[str]) -> str | None:
    """根据清单文件判断项目类型。"""
    for manifest in _MANIFEST_PROJECT_TYPES:
        if manifest in manifest_files:
            return _MANIFEST_PROJECT_TYPES[manifest]
    return None


def _git_status(root: Path) -> tuple[str | None, bool | None]:
    """获取 Git 仓库状态（当前分支名和是否有未提交变更）。

    Returns:
        (分支名, 是否有变更) 元组；非 Git 仓库时返回 (None, None)。
    """
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return None, None
    if branch.returncode != 0 or status.returncode != 0:
        return None, None
    branch_text = branch.stdout.strip() or None
    return branch_text, bool(status.stdout.strip())


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
