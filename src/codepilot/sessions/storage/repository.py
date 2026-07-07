from __future__ import annotations

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
_INSTRUCTION_FILES = ["AGENTS.md", "CLAUDE.md", "COPILOT.md", "INSTRUCTIONS.md"]
_INTERNAL_TOP_LEVEL_NAMES = {".git", ".codepilot", ".pytest_cache", "__pycache__"}
_TOP_LEVEL_LIMIT = 30


@dataclass(frozen=True)
class GitInfo:
    root: Path
    branch: str | None = None
    head_sha: str | None = None
    is_dirty: bool = False
    remote_url: str | None = None


@dataclass(frozen=True)
class RepositoryBootstrap:
    workspace_root: str
    project_type: str | None
    manifest_files: list[str]
    top_level_entries: list[str]
    test_directories: list[str]
    instruction_files: list[str]
    git: GitInfo | None = None


def build_repository_bootstrap(workspace: Path) -> RepositoryBootstrap:
    root = Path(workspace).resolve()
    entries = _top_level_entries(root)
    manifest_files = [name for name in _MANIFEST_PROJECT_TYPES if (root / name).is_file()]
    return RepositoryBootstrap(
        workspace_root=str(root).replace("\\", "/"),
        project_type=_project_type(manifest_files),
        manifest_files=manifest_files,
        top_level_entries=entries,
        test_directories=[
            entry
            for entry in entries
            if entry.endswith("/") and entry[:-1] in _TEST_DIR_NAMES
        ],
        instruction_files=[name for name in _INSTRUCTION_FILES if (root / name).is_file()],
        git=_build_git_info(root),
    )


def render_repository_context(bootstrap: RepositoryBootstrap) -> str:
    lines = [
        "## Repository Context",
        f"- Workspace: {bootstrap.workspace_root}",
        f"- Project type: {bootstrap.project_type or 'unknown'}",
        f"- Manifests: {_joined(bootstrap.manifest_files)}",
        f"- Top-level: {_joined(bootstrap.top_level_entries, empty='(empty)')}",
        f"- Test directories: {_joined(bootstrap.test_directories)}",
        f"- Instruction files: {_joined(bootstrap.instruction_files)}",
    ]
    if bootstrap.git is None:
        lines.append("- Git: not a git repository")
    else:
        lines.extend(
            [
                f"- Git branch: {bootstrap.git.branch or 'detached HEAD'}",
                f"- HEAD: {bootstrap.git.head_sha or 'unknown'}",
                f"- Working tree: {'modified' if bootstrap.git.is_dirty else 'clean'}",
            ]
        )
    return "\n".join(lines)


def _build_git_info(root: Path) -> GitInfo | None:
    if _git(root, "rev-parse", "--git-dir").returncode != 0:
        return None
    git_root = _stdout(_git(root, "rev-parse", "--show-toplevel"))
    return GitInfo(
        root=Path(git_root) if git_root else root,
        branch=_stdout(_git(root, "branch", "--show-current")) or None,
        head_sha=_stdout(_git(root, "rev-parse", "--short", "HEAD")) or None,
        is_dirty=any(
            not _is_internal_status_line(line)
            for line in _stdout(_git(root, "status", "--porcelain")).splitlines()
            if line.strip()
        ),
        remote_url=_stdout(_git(root, "remote", "get-url", "origin")) or None,
    )


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return subprocess.CompletedProcess(["git", *args], 1, "", "")


def _stdout(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout.strip()


def _top_level_entries(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    items = sorted(
        (
            item
            for item in root.iterdir()
            if item.name not in _INTERNAL_TOP_LEVEL_NAMES
        ),
        key=lambda item: item.name.lower(),
    )
    return [f"{item.name}/" if item.is_dir() else item.name for item in items[:_TOP_LEVEL_LIMIT]]


def _is_internal_status_line(line: str) -> bool:
    if len(line) < 4:
        return False
    path = line[3:].split(" -> ")[-1].replace("\\", "/").strip("/")
    return bool(path and path.split("/", 1)[0] in _INTERNAL_TOP_LEVEL_NAMES)


def _project_type(manifest_files: list[str]) -> str | None:
    for manifest, project_type in _MANIFEST_PROJECT_TYPES.items():
        if manifest in manifest_files:
            return project_type
    return None


def _joined(values: list[str], *, empty: str = "(none)") -> str:
    return ", ".join(values) if values else empty


__all__ = [
    "GitInfo",
    "RepositoryBootstrap",
    "build_repository_bootstrap",
    "render_repository_context",
]
