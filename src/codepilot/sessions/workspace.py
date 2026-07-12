from __future__ import annotations

# 新手导读：workspace_state.py 负责会话层需要的文件状态快照。
# 关注点：它服务 freshness、rollback 和 context validation，不依赖 tools 的执行管线。

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import WorkspaceCheckpoint, WorkspaceRecoveryState


@dataclass(frozen=True)
class GitInfo:
    branch: str | None = None
    head_sha: str | None = None
    is_dirty: bool = False


@dataclass(frozen=True)
class RepositoryBootstrap:
    workspace_root: str
    project_type: str | None
    manifest_files: list[str]
    top_level_entries: list[str]
    test_directories: list[str]
    instruction_files: list[str]
    git: GitInfo | None = None


def build_repository_bootstrap(workspace: str | Path) -> RepositoryBootstrap:
    root = Path(workspace).resolve()
    manifests = [
        name
        for name in ("pyproject.toml", "package.json", "Cargo.toml", "go.mod")
        if (root / name).is_file()
    ]
    entries = sorted(
        f"{path.name}/" if path.is_dir() else path.name
        for path in root.iterdir()
        if path.name not in {".codepilot", ".git"}
    )[:30]
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    head = _git(root, "rev-parse", "HEAD")
    return RepositoryBootstrap(
        workspace_root=str(root).replace("\\", "/"),
        project_type=_project_type(manifests),
        manifest_files=manifests,
        top_level_entries=entries,
        test_directories=[item for item in entries if item.rstrip("/") in {"test", "tests"}],
        instruction_files=[
            name
            for name in ("AGENTS.md", "CLAUDE.md", "README.md")
            if (root / name).is_file()
        ],
        git=(
            GitInfo(
                branch=branch,
                head_sha=head,
                is_dirty=bool(_git(root, "status", "--porcelain")),
            )
            if branch or head
            else None
        ),
    )


def render_repository_context(bootstrap: RepositoryBootstrap) -> str:
    return "\n".join(
        [
            "## Repository Context",
            f"- Workspace: {bootstrap.workspace_root}",
            f"- Project type: {bootstrap.project_type or 'unknown'}",
            f"- Manifests: {', '.join(bootstrap.manifest_files) or '(none)'}",
            f"- Top-level: {', '.join(bootstrap.top_level_entries) or '(empty)'}",
            f"- Test directories: {', '.join(bootstrap.test_directories) or '(none)'}",
            f"- Instruction files: {', '.join(bootstrap.instruction_files) or '(none)'}",
            f"- Git branch: {bootstrap.git.branch if bootstrap.git else 'not available'}",
            f"- HEAD: {bootstrap.git.head_sha if bootstrap.git else 'not available'}",
            f"- Working tree: {'modified' if bootstrap.git and bootstrap.git.is_dirty else 'clean'}",
        ]
    )


def file_state_for_path(workspace_dir: str | Path, path: str | Path) -> dict[str, Any]:
    """Return a bounded file-state snapshot for session freshness checks."""

    root = Path(workspace_dir).resolve()
    target = Path(path)
    resolved = target.resolve() if target.is_absolute() else (root / target).resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Path escapes workspace boundary") from exc
    if not resolved.exists() or not resolved.is_file():
        return {
            "path": relative,
            "exists": False,
            "workspace_path": str(root),
        }
    stat = resolved.stat()
    return {
        "path": relative,
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(resolved),
        "workspace_path": str(root),
    }


def capture_workspace_checkpoint(
    workspace_dir: str | Path,
    *,
    tracked_paths: list[str] | tuple[str, ...] = (),
) -> WorkspaceCheckpoint:
    root = Path(workspace_dir).resolve()
    dirty_paths = tuple(_git_status_paths(root))
    paths = sorted(set(dirty_paths) | {Path(path).as_posix() for path in tracked_paths if path})
    hashes = {
        path: str(state["sha256"])
        for path in paths
        if (state := file_state_for_path(root, path)).get("exists")
        and isinstance(state.get("sha256"), str)
    }
    return WorkspaceCheckpoint(
        root=str(root),
        git_head=_git(root, "rev-parse", "HEAD"),
        dirty_paths=dirty_paths,
        tracked_path_hashes=hashes,
    )


def validate_workspace_checkpoint(
    workspace_dir: str | Path,
    checkpoint: WorkspaceCheckpoint | None,
) -> WorkspaceRecoveryState:
    if checkpoint is None:
        return WorkspaceRecoveryState(status="unchanged")
    root = Path(workspace_dir).resolve()
    if Path(checkpoint.root).resolve() != root:
        return WorkspaceRecoveryState(status="changed")
    current = capture_workspace_checkpoint(
        root,
        tracked_paths=tuple(checkpoint.tracked_path_hashes),
    )
    changed: set[str] = set(current.dirty_paths) ^ set(checkpoint.dirty_paths)
    missing: set[str] = set()
    for path, expected_hash in checkpoint.tracked_path_hashes.items():
        state = file_state_for_path(root, path)
        if not state.get("exists"):
            missing.add(path)
        elif state.get("sha256") != expected_hash:
            changed.add(path)
    if checkpoint.git_head != current.git_head:
        changed.add(".git/HEAD")
    return WorkspaceRecoveryState(
        status="changed" if changed or missing else "unchanged",
        changed_paths=tuple(sorted(changed)),
        missing_paths=tuple(sorted(missing)),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_type(manifests: list[str]) -> str | None:
    for name, kind in (
        ("pyproject.toml", "Python"),
        ("package.json", "Node.js"),
        ("Cargo.toml", "Rust"),
        ("go.mod", "Go"),
    ):
        if name in manifests:
            return kind
    return None


def _git(root: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_status_paths(root: Path) -> list[str]:
    output = _git(root, "status", "--porcelain") or ""
    paths: list[str] = []
    for line in output.splitlines():
        value = line[3:].split(" -> ")[-1].strip().strip('"').replace("\\", "/")
        if value and not value.startswith(".codepilot/"):
            paths.append(value)
    return sorted(set(paths))


__all__ = [
    "GitInfo",
    "RepositoryBootstrap",
    "build_repository_bootstrap",
    "capture_workspace_checkpoint",
    "file_state_for_path",
    "render_repository_context",
    "validate_workspace_checkpoint",
]
