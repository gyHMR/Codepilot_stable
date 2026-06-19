from __future__ import annotations

"""
运行时上下文构建模块。

为系统提示词渲染提供运行时上下文信息，包括：
- 仓库基本信息（项目类型、目录结构、Git 状态等）
- 提示词准则（来自配置和扩展）
- 追加段落（来自配置和扩展）
- 工具说明片段
- 长期记忆文本

仓库采集结果由 RepositoryBootstrap 统一表示，避免在装配层重复定义。
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
# 指令文件列表
_INSTRUCTION_FILES = ["AGENTS.md", "CLAUDE.md", "COPILOT.md", "INSTRUCTIONS.md"]


@dataclass(frozen=True)
class GitInfo:
    """Git 仓库信息。

    Attributes:
        root: Git 根目录路径。
        branch: 当前分支名（detached HEAD 时为 None）。
        head_sha: HEAD 提交的短 SHA。
        is_dirty: 工作区是否有未提交变更。
        remote_url: 远程仓库 URL（可选）。
    """
    root: Path
    branch: str | None = None
    head_sha: str | None = None
    is_dirty: bool = False
    remote_url: str | None = None


@dataclass(frozen=True)
class RepositoryBootstrap:
    """仓库引导信息（从工作区目录扫描得到的静态信息）。

    Attributes:
        workspace_root: 工作区根目录的绝对路径。
        project_type: 项目类型（如 "Python"、"JavaScript/TypeScript"），无法识别时为 None。
        manifest_files: 存在的清单文件名列表（如 ["pyproject.toml"]）。
        top_level_entries: 顶层目录和文件列表（目录带 "/" 后缀）。
        test_directories: 测试目录列表。
        instruction_files: 指令文件列表（如 AGENTS.md、CLAUDE.md）。
        git: Git 仓库信息（非 Git 仓库时为 None）。
    """
    workspace_root: str
    project_type: str | None
    manifest_files: list[str]
    top_level_entries: list[str]
    test_directories: list[str]
    instruction_files: list[str]
    git: GitInfo | None = None


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

    检测项目类型（通过清单文件）、目录结构、测试目录、指令文件和 Git 状态。

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
    # 检测指令文件
    instruction_files = [name for name in _INSTRUCTION_FILES if (root / name).is_file()]
    # 获取 Git 信息
    git_info = _build_git_info(root)
    return RepositoryBootstrap(
        workspace_root=str(root).replace("\\", "/"),
        project_type=project_type,
        manifest_files=manifest_files,
        top_level_entries=entries,
        test_directories=test_directories,
        instruction_files=instruction_files,
        git=git_info,
    )


def _build_git_info(root: Path) -> GitInfo | None:
    """构建 Git 仓库信息。

    Returns:
        GitInfo 对象，非 Git 仓库时返回 None。
    """
    try:
        # 检查是否为 Git 仓库
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode != 0:
            return None

        # 获取 Git 根目录
        git_root_result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        git_root = Path(git_root_result.stdout.strip()) if git_root_result.returncode == 0 else root

        # 获取当前分支
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        branch = branch_result.stdout.strip() or None

        # 获取 HEAD SHA
        head_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        head_sha = head_result.stdout.strip() or None

        # 检查工作区状态
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        is_dirty = bool(status_result.stdout.strip())

        # 获取远程 URL
        remote_result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        remote_url = remote_result.stdout.strip() or None

        return GitInfo(
            root=git_root,
            branch=branch,
            head_sha=head_sha,
            is_dirty=is_dirty,
            remote_url=remote_url,
        )
    except Exception:
        return None


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
    instructions = ", ".join(bootstrap.instruction_files) if bootstrap.instruction_files else "(none)"

    lines = [
        "## Repository Context",
        f"- Workspace: {bootstrap.workspace_root}",
        f"- Project type: {project_type}",
        f"- Manifests: {manifests}",
        f"- Top-level: {top_level}",
        f"- Test directories: {tests}",
        f"- Instruction files: {instructions}",
    ]

    # Git 信息
    if bootstrap.git:
        branch = bootstrap.git.branch or "detached HEAD"
        dirty = "modified" if bootstrap.git.is_dirty else "clean"
        lines.append(f"- Git branch: {branch}")
        lines.append(f"- HEAD: {bootstrap.git.head_sha or 'unknown'}")
        lines.append(f"- Working tree: {dirty}")
    else:
        lines.append("- Git: not a git repository")

    return "\n".join(lines)


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
