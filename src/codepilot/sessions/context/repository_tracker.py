from __future__ import annotations

# 新手导读：RepositoryTracker 跟踪工作区文件指纹和变更 delta。
# 关注点：它为上下文新鲜度和 changed files 提供基础证据。

"""每次模型调用前的低成本动态仓库快照。

仓库快照是动态会话上下文，而非一次性运行时装配状态。
它们刻意将顶层目录条目纳入指纹和差异计算，
以便在下一次模型调用前反映目录的创建/删除。
"""

import hashlib
import subprocess
from pathlib import Path

from codepilot.protocols import RepositoryDelta, RepositorySnapshot

from .repository_context import build_repository_bootstrap


class RepositoryTracker:
    """仓库追踪器：生成低成本的仓库快照并计算前后差异。"""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()

    def snapshot(self) -> RepositorySnapshot:
        bootstrap = build_repository_bootstrap(self.workspace)
        git_status = _git_lines(self.workspace, ["status", "--porcelain"])
        dirty_path_hashes = _dirty_path_hashes(self.workspace, git_status)
        instruction_hashes = {
            path: _sha256(self.workspace / path)
            for path in bootstrap.instruction_files
            if (self.workspace / path).is_file()
        }
        payload = [
            bootstrap.workspace_root,
            bootstrap.project_type or "",
            *(bootstrap.manifest_files),
            *(bootstrap.top_level_entries),
            *(bootstrap.test_directories),
            *(f"{key}:{value}" for key, value in sorted(instruction_hashes.items())),
            *(f"{key}:{value}" for key, value in sorted(dirty_path_hashes.items())),
            bootstrap.git.branch if bootstrap.git and bootstrap.git.branch else "",
            bootstrap.git.head_sha if bootstrap.git and bootstrap.git.head_sha else "",
            *git_status,
        ]
        fingerprint = hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()
        return RepositorySnapshot(
            workspace_root=bootstrap.workspace_root,
            project_type=bootstrap.project_type,
            manifest_files=list(bootstrap.manifest_files),
            top_level_entries=list(bootstrap.top_level_entries),
            test_directories=list(bootstrap.test_directories),
            instruction_files=list(bootstrap.instruction_files),
            branch=bootstrap.git.branch if bootstrap.git else None,
            head_sha=bootstrap.git.head_sha if bootstrap.git else None,
            git_status=git_status,
            fingerprint=fingerprint,
            instruction_hashes=instruction_hashes,
            dirty_path_hashes=dirty_path_hashes,
        )

    def refresh(
        self,
        previous: RepositorySnapshot | None,
    ) -> tuple[RepositorySnapshot, RepositoryDelta]:
        current = self.snapshot()
        return current, compare_snapshots(previous, current)


def compare_snapshots(
    previous: RepositorySnapshot | None,
    current: RepositorySnapshot,
) -> RepositoryDelta:
    if previous is None:
        return RepositoryDelta()
    old_paths = set(previous.top_level_entries)
    new_paths = set(current.top_level_entries)
    old_status = _status_map(previous.git_status)
    new_status = _status_map(current.git_status)
    modified = sorted(
        path
        for path, status in new_status.items()
        if status != "??"
        and (
            old_status.get(path) != status
            or previous.dirty_path_hashes.get(path) != current.dirty_path_hashes.get(path)
        )
    )
    deleted = sorted(
        {
            *[path for path, status in new_status.items() if "D" in status],
            *(old_paths - new_paths),
        }
    )
    return RepositoryDelta(
        added_paths=sorted((new_paths - old_paths) | {p for p, s in new_status.items() if s == "??"}),
        modified_paths=modified,
        deleted_paths=deleted,
        branch_changed=previous.branch != current.branch,
        head_changed=previous.head_sha != current.head_sha,
        instructions_changed=previous.instruction_hashes != current.instruction_hashes,
    )


def render_repository_snapshot(
    snapshot: RepositorySnapshot,
    delta: RepositoryDelta,
) -> str:
    lines = [
        "## Repository Context",
        f"- Repository fingerprint: {snapshot.fingerprint[:12]}",
        f"- Workspace: {snapshot.workspace_root}",
        f"- Project type: {snapshot.project_type or 'unknown'}",
        f"- Manifests: {', '.join(snapshot.manifest_files) or '(none)'}",
        f"- Top-level: {', '.join(snapshot.top_level_entries) or '(empty)'}",
        f"- Test directories: {', '.join(snapshot.test_directories) or '(none)'}",
        f"- Instruction files: {', '.join(snapshot.instruction_files) or '(none)'}",
        f"- Git branch: {snapshot.branch or 'unknown'}",
        f"- HEAD: {snapshot.head_sha or 'unknown'}",
        f"- Working tree changes: {len(snapshot.git_status)}",
    ]
    if delta.changed:
        lines.extend(
            [
                "### Changes since previous model call",
                f"- Added: {', '.join(delta.added_paths) or '(none)'}",
                f"- Modified: {', '.join(delta.modified_paths) or '(none)'}",
                f"- Deleted: {', '.join(delta.deleted_paths) or '(none)'}",
                f"- Branch changed: {delta.branch_changed}",
                f"- HEAD changed: {delta.head_changed}",
                f"- Instructions changed: {delta.instructions_changed}",
            ]
        )
    return "\n".join(lines)


def _git_lines(root: Path, args: list[str]) -> list[str]:
    try:
        result = subprocess.run(
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
        return []
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def _status_map(lines: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in lines:
        if len(line) < 4:
            continue
        status = line[:2]
        path = line[3:].split(" -> ")[-1]
        result[path] = status
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dirty_path_hashes(root: Path, status_lines: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in _status_map(status_lines):
        target = (root / path).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            continue
        if target.is_file():
            hashes[path] = _sha256(target)
        elif target.is_dir():
            hashes[path] = _directory_fingerprint(target)
        elif not target.exists():
            hashes[path] = "<missing>"
    return hashes


def _directory_fingerprint(path: Path, *, limit: int = 100) -> str:
    digest = hashlib.sha256()
    files = sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.as_posix(),
    )
    for item in files[:limit]:
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(_sha256(item).encode("ascii"))
    digest.update(f"count:{len(files)}".encode("ascii"))
    return digest.hexdigest()


__all__ = [
    "RepositoryTracker",
    "compare_snapshots",
    "render_repository_snapshot",
]
