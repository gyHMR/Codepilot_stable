from __future__ import annotations

# 新手导读：repository_context 定义仓库上下文相关的数据结构和渲染辅助。
# 关注点：它服务于上下文投影，不直接扫描全部源码塞进 prompt。

"""
Repository bootstrap helpers shared by runtime prompt assembly and session
context governance.

Runtime uses this module to render static system-prompt bootstrap information
when a session is created. Sessions also use it to refresh dynamic repository
snapshots before every model call, so top-level directory changes do not remain
stale in the model context.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path


_MANIFEST_PROJECT_TYPES = {
    "pyproject.toml": "Python",
    "package.json": "JavaScript/TypeScript",
    "Cargo.toml": "Rust",
    "go.mod": "Go",
    "pom.xml": "Java",
}
_TEST_DIR_NAMES = {"test", "tests", "spec", "specs"}
_TOP_LEVEL_LIMIT = 30
_INSTRUCTION_FILES = ["AGENTS.md", "CLAUDE.md", "COPILOT.md", "INSTRUCTIONS.md"]


@dataclass(frozen=True)
class GitInfo:
    """Git 仓库信息。"""

    root: Path
    branch: str | None = None
    head_sha: str | None = None
    is_dirty: bool = False
    remote_url: str | None = None


@dataclass(frozen=True)
class RepositoryBootstrap:
    """仓库引导信息（从工作区目录扫描得到的静态信息）。"""

    workspace_root: str
    project_type: str | None
    manifest_files: list[str]
    top_level_entries: list[str]
    test_directories: list[str]
    instruction_files: list[str]
    git: GitInfo | None = None


def build_repository_bootstrap(workspace: Path) -> RepositoryBootstrap:
    """扫描工作区目录，构建仓库引导信息。"""

    root = workspace.resolve()
    entries = _top_level_entries(root)
    manifest_files = [name for name in _MANIFEST_PROJECT_TYPES if (root / name).is_file()]
    project_type = _project_type(manifest_files)
    test_directories = [
        entry
        for entry in entries
        if entry.endswith("/") and entry[:-1] in _TEST_DIR_NAMES
    ]
    instruction_files = [name for name in _INSTRUCTION_FILES if (root / name).is_file()]
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


def render_repository_context(bootstrap: RepositoryBootstrap) -> str:
    """将仓库引导信息渲染为 Markdown 格式的上下文文本。"""

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

    if bootstrap.git:
        branch = bootstrap.git.branch or "detached HEAD"
        dirty = "modified" if bootstrap.git.is_dirty else "clean"
        lines.append(f"- Git branch: {branch}")
        lines.append(f"- HEAD: {bootstrap.git.head_sha or 'unknown'}")
        lines.append(f"- Working tree: {dirty}")
    else:
        lines.append("- Git: not a git repository")

    return "\n".join(lines)


def _build_git_info(root: Path) -> GitInfo | None:
    """构建 Git 仓库信息；非 Git 仓库时返回 None。"""

    try:
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

        git_root_result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        git_root = Path(git_root_result.stdout.strip()) if git_root_result.returncode == 0 else root

        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        branch = branch_result.stdout.strip() or None

        head_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        head_sha = head_result.stdout.strip() or None

        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        is_dirty = bool(status_result.stdout.strip())

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


def _top_level_entries(root: Path) -> list[str]:
    """获取工作区根目录的顶层文件和目录列表。"""

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


__all__ = [
    "GitInfo",
    "RepositoryBootstrap",
    "build_repository_bootstrap",
    "render_repository_context",
]
