from __future__ import annotations

# 新手导读：repository.py 定义会话层可公开消费的仓库引导快照。
# 关注点：Runtime 可以用它生成启动提示词；ContextGovernor 会在每轮 run 前刷新更动态的上下文。

"""Repository bootstrap snapshots owned by the sessions layer."""

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
_INTERNAL_TOP_LEVEL_NAMES = {".git", ".codepilot", ".pytest_cache", "__pycache__"}


@dataclass(frozen=True)
class GitInfo:
    """Git repository facts captured for a read-only snapshot."""

    root: Path
    branch: str | None = None
    head_sha: str | None = None
    is_dirty: bool = False
    remote_url: str | None = None


@dataclass(frozen=True)
class RepositoryBootstrap:
    """Static repository facts used when opening a session."""

    workspace_root: str
    project_type: str | None
    manifest_files: list[str]
    top_level_entries: list[str]
    test_directories: list[str]
    instruction_files: list[str]
    git: GitInfo | None = None


def build_repository_bootstrap(workspace: Path) -> RepositoryBootstrap:
    """Scan the workspace root and build a stable repository bootstrap view."""

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
    """Render repository bootstrap facts as Markdown for a system prompt."""

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
        is_dirty = any(
            not _is_internal_status_line(line)
            for line in status_result.stdout.splitlines()
            if line.strip()
        )

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
    if not root.exists() or not root.is_dir():
        return []
    items = sorted(
        (
            item
            for item in root.iterdir()
            if item.name not in _INTERNAL_TOP_LEVEL_NAMES
        ),
        key=lambda item: item.name.lower(),
    )
    entries = []
    for item in items[:_TOP_LEVEL_LIMIT]:
        entries.append(f"{item.name}/" if item.is_dir() else item.name)
    return entries


def _is_internal_status_line(line: str) -> bool:
    if len(line) < 4:
        return False
    path = line[3:].split(" -> ")[-1].replace("\\", "/").strip("/")
    if not path:
        return False
    return path.split("/", 1)[0] in _INTERNAL_TOP_LEVEL_NAMES


def _project_type(manifest_files: list[str]) -> str | None:
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
