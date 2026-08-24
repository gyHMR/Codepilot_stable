"""工作区状态工具 —— 会话层需要的文件状态快照和仓库信息。

本文件提供三类功能：
1. 仓库引导信息（RepositoryBootstrap）—— 工作区的项目元数据
2. 工作区检查点（capture_workspace_checkpoint）—— 运行开始时的文件快照
3. 检查点验证（validate_workspace_checkpoint）—— 恢复时检查工作区变化

注意：本文件服务 freshness、rollback 和 context validation，
不依赖 tools 的执行管线。
"""

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import WorkspaceCheckpoint, WorkspaceRecoveryState


@dataclass(frozen=True)
class GitInfo:
    """Git 仓库信息。

    参数:
        branch: 当前分支名
        head_sha: HEAD 的提交 SHA
        is_dirty: 工作区是否有未提交的更改
    """
    branch: str | None = None
    head_sha: str | None = None
    is_dirty: bool = False


@dataclass(frozen=True)
class RepositoryBootstrap:
    """仓库引导信息 —— 工作区的项目结构和 Git 状态。

    用于在会话开始时向 Agent 展示工作区的概览信息。

    参数:
        workspace_root: 工作区根目录路径
        project_type: 项目类型（Python / Node.js / Rust / Go / None）
        manifest_files: 发现的清单文件列表
        top_level_entries: 顶级目录和文件名（前 30 个）
        test_directories: 测试目录列表
        instruction_files: 指令文件列表（AGENTS.md / CLAUDE.md / README.md）
        git: Git 仓库信息（可选）
    """
    workspace_root: str
    project_type: str | None
    manifest_files: list[str]
    top_level_entries: list[str]
    test_directories: list[str]
    instruction_files: list[str]
    git: GitInfo | None = None


def build_repository_bootstrap(workspace: str | Path) -> RepositoryBootstrap:
    """构建仓库引导信息 —— 扫描工作区并收集元数据。

    扫描内容：
    - 项目清单文件（pyproject.toml / package.json / Cargo.toml / go.mod）
    - 顶级目录和文件（排除 .codepilot 和 .git）
    - 测试目录
    - 指令文件
    - Git 仓库信息（分支、HEAD、Dirty 状态）

    参数:
        workspace: 工作区根目录

    返回:
        RepositoryBootstrap 包含完整的项目元数据
    """
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
    """将仓库引导信息渲染为系统提示词中的文本块。

    参数:
        bootstrap: RepositoryBootstrap 对象

    返回:
        格式化的多行文本（直接插入到系统提示词中）
    """
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
    """返回工作区内文件的状态快照（用于会话 freshness 检查）。

    与 tools/sandbox.py 中的同名函数类似，但独立实现，
    不依赖工具子系统的沙箱。

    如果文件存在，返回 size、mtime_ns、sha256；
    如果文件不存在，返回 exists=False。

    参数:
        workspace_dir: 工作区根目录
        path: 目标文件路径

    返回:
        文件状态字典（包含 path、exists、size、sha256 等）
    """
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
    """捕获工作区检查点 —— 记录运行开始时的文件系统状态。

    保存的信息包括：
    - Git HEAD 提交哈希
    - 未提交的变更路径（git status --porcelain）
    - 指定跟踪文件的 SHA256 哈希

    参数:
        workspace_dir: 工作区根目录
        tracked_paths: 要跟踪的文件路径列表

    返回:
        WorkspaceCheckpoint 包含文件状态快照
    """
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
    """验证工作区检查点 —— 检查从检查点之后工作区是否发生了变化。

    对比两个方面的变化：
    1. Git dirty paths 的变化（新增或消失的未提交变更）
    2. 跟踪文件的 SHA256 哈希变化（文件内容被修改）
    3. Git HEAD 的变化（切换了分支或提交了代码）

    参数:
        workspace_dir: 工作区根目录
        checkpoint: 要验证的工作区检查点（None = 视为未变化）

    返回:
        WorkspaceRecoveryState 包含变化详情
    """
    if checkpoint is None:
        return WorkspaceRecoveryState(status="unknown")
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


# ── 内部辅助函数 ──────────────────────────────────────────────────────────────


def _sha256_file(path: Path) -> str:
    """计算文件的 SHA256 哈希（流式读取，内存恒定）。"""
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_type(manifests: list[str]) -> str | None:
    """根据清单文件名推断项目类型。"""
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
    """运行 git 命令并返回标准输出（去除尾部换行）。

    如果命令失败（非零退出码），返回 None。
    """
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
    """解析 git status --porcelain 输出，提取变更文件的路径列表。

    排除 .codepilot 目录下的变更。

    返回:
        变更的文件路径列表（排序、去重）
    """
    output = _git(
        root,
        "status",
        "--porcelain",
        "--untracked-files=normal",
        "--relative",
        "--",
        ".",
    ) or ""
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
