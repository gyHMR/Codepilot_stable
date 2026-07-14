"""Git 干净工作树回滚 —— 对单次记录的运行做文件级别回滚。

本文件提供基于 Git 的文件级别回滚功能：
当一个 Agent 运行修改了工作区文件后，可以将其回滚到运行开始前的状态。

回滚策略：
- 对 Git 跟踪的文件：使用 git restore 恢复为 HEAD 版本
- 对未跟踪的新文件：直接删除
- 回滚前会检查文件是否有运行之外的中间修改（阻止回滚防止丢失数据）

核心流程：
capture_git_baseline → （运行结束后）→ plan_run_rollback → revert_run_changes
"""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from codepilot.sessions.workspace import file_state_for_path


RollbackStatus = Literal["reverted", "not_eligible", "conflict", "noop"]
RollbackPlanStatus = Literal["ready", "blocked", "not_eligible", "noop"]
RollbackAction = Literal["restore", "remove", "skip", "block"]
_ROLLBACK_STATUSES = frozenset({"reverted", "not_eligible", "conflict", "noop"})
_ROLLBACK_PLAN_STATUSES = frozenset({"ready", "blocked", "not_eligible", "noop"})
_ROLLBACK_ACTIONS = frozenset({"restore", "remove", "skip", "block"})


@dataclass(frozen=True)
class GitRollbackBaseline:
    """Git 回滚基线 —— 运行开始时的 Git 状态。

    参数:
        eligible: 是否有资格进行回滚
            - True: 运行开始时工作区是干净的（无未提交变更）
            - False: 运行开始时工作区不干净，无法区分运行前后的变化
        reason: 不满足回滚条件的原因
        head: 运行开始时的 HEAD 提交哈希
        branch: 运行开始时的当前分支名
        status_before: 运行前的 git status 输出
    """
    eligible: bool
    reason: str | None = None
    head: str | None = None
    branch: str | None = None
    status_before: str = ""


@dataclass(frozen=True)
class GitRollbackResult:
    """Git 回滚结果。

    参数:
        status: 回滚状态
        run_id: 运行的 ID
        reason: 说明或原因
        restored_paths: 成功恢复的文件路径列表（git restore）
        removed_paths: 成功删除的文件路径列表
        conflicted_paths: 冲突的文件路径列表（无法回滚）
    """
    status: RollbackStatus
    run_id: str
    reason: str | None = None
    restored_paths: list[str] = field(default_factory=list)
    removed_paths: list[str] = field(default_factory=list)
    conflicted_paths: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _ensure_rollback_status(self.status)


@dataclass(frozen=True)
class GitRollbackAction:
    """回滚行动计划 —— 对单个文件的处理决策。

    参数:
        path: 文件路径
        action: 处理动作
        reason: 决策原因
    """
    path: str
    action: RollbackAction
    reason: str | None = None

    def __post_init__(self) -> None:
        _ensure_rollback_action(self.action)


@dataclass(frozen=True)
class GitRollbackPlan:
    """回滚计划 —— 回滚前的完整决策结果。

    参数:
        status: 计划状态
        run_id: 运行 ID
        reason: 状态原因
        actions: 对每个受影响文件的处理行动
        ignored_paths: 被忽略的路径（不属于本次运行的变更）
    """
    status: RollbackPlanStatus
    run_id: str
    reason: str | None = None
    actions: list[GitRollbackAction] = field(default_factory=list)
    ignored_paths: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _ensure_rollback_plan_status(self.status)


# =========================================================================
# 核心函数
# =========================================================================


def capture_git_baseline(workspace_dir: str | Path) -> GitRollbackBaseline:
    """捕获 Git 基线 —— 在运行开始前记录 Git 状态。

    检查工作区是否满足回滚条件：
    - 必须在 Git 仓库中
    - 运行开始时工作区必须是干净的（没有未提交的变更）

    参数:
        workspace_dir: 工作区目录

    返回:
        GitRollbackBaseline 包含基线状态
    """
    root = Path(workspace_dir)
    if _git(root, "rev-parse", "--is-inside-work-tree").returncode != 0:
        return GitRollbackBaseline(eligible=False, reason="not_git_repo")

    head_result = _git(root, "rev-parse", "--short", "HEAD")
    branch_result = _git(root, "branch", "--show-current")
    status = _visible_status(root)
    if status:
        return GitRollbackBaseline(
            eligible=False,
            reason="dirty_worktree_before_run",
            head=_stdout(head_result) or None,
            branch=_stdout(branch_result) or None,
            status_before=status,
        )

    return GitRollbackBaseline(
        eligible=True,
        head=_stdout(head_result) or None,
        branch=_stdout(branch_result) or None,
        status_before="",
    )


def build_rollback_metadata(
    baseline: GitRollbackBaseline,
    *,
    affected_paths: list[str],
    workspace_changed: bool,
    workspace_dir: str | Path,
) -> dict[str, Any]:
    """构建回滚元数据 —— 保存到运行状态中供后续回滚使用。

    参数:
        baseline: Git 基线
        affected_paths: 运行影响的工作区文件路径
        workspace_changed: 工作区是否发生了变更

    返回:
        元数据字典（JSON 可序列化）
    """
    root = Path(workspace_dir).resolve()
    post_run_files = {
        path: file_state_for_path(root, path)
        for path in _normalize_paths(affected_paths)
    }
    return {
        "strategy": "git-clean-worktree",
        "eligible": baseline.eligible,
        "reason": baseline.reason,
        "baseline": {
            "head": baseline.head,
            "branch": baseline.branch,
            "status_before": baseline.status_before,
        },
        "affected_paths": _normalize_paths(affected_paths),
        "post_run_files": post_run_files,
        "workspace_changed": bool(workspace_changed),
    }


def plan_run_rollback(
    workspace_dir: str | Path,
    run_state: dict[str, Any],
) -> GitRollbackPlan:
    """计划运行回滚 —— 对一次运行的变更进行全面分析并制定回滚计划。

    处理流程：
    1. 检查运行是否具备回滚条件（元数据、eligible 标志）
    2. 提取受影响的文件路径
    3. 对每个文件，决定处理方式（restore/remove/skip/block）
    4. 如果有任何文件被 blocked（因为运行后有额外修改），阻止回滚

    restore 条件：Git 跟踪的文件，且自运行后未被修改
    remove 条件：未跟踪的新文件（运行期间创建的）
    block 条件：文件自运行结束后又被修改过（防止数据丢失）
    skip 条件：内部文件（.codepilot/*）、已经删除的文件

    参数:
        workspace_dir: 工作区目录
        run_state: 运行状态的字典表示（含 rollback 元数据）

    返回:
        GitRollbackPlan 包含完整的回滚计划和决策
    """
    root = Path(workspace_dir)
    run_id = str(run_state.get("run_id") or "")
    rollback = run_state.get("rollback")
    if not isinstance(rollback, dict):
        return GitRollbackPlan(status="not_eligible", run_id=run_id, reason="missing_rollback_metadata")
    if rollback.get("strategy") != "git-clean-worktree":
        return GitRollbackPlan(status="not_eligible", run_id=run_id, reason="unsupported_rollback_strategy")
    if not rollback.get("eligible"):
        return GitRollbackPlan(
            status="not_eligible",
            run_id=run_id,
            reason=str(rollback.get("reason") or "rollback_not_eligible"),
        )
    if not rollback.get("workspace_changed"):
        return GitRollbackPlan(status="noop", run_id=run_id, reason="workspace_not_changed")

    affected_paths = _normalize_paths(
        [str(path) for path in rollback.get("affected_paths", []) if isinstance(path, str)]
    )
    if not affected_paths:
        return GitRollbackPlan(status="noop", run_id=run_id, reason="no_affected_paths")
    if not isinstance(rollback.get("post_run_files"), dict):
        return GitRollbackPlan(
            status="not_eligible",
            run_id=run_id,
            reason="missing_post_run_workspace_state",
        )

    affected = set(affected_paths)
    status_entries = _status_entries(root)
    # 收集运行之外的变更（与回滚无关，但有冲突风险的路径）
    ignored_paths = sorted(
        {
            entry.path
            for entry in status_entries
            if not _is_internal_path(entry.path) and entry.path not in affected
        }
    )
    # 检测"受影响的文件被暂存了"的情况
    staged_affected = {
        entry.path
        for entry in status_entries
        if entry.staged and not _is_internal_path(entry.path) and entry.path in affected
    }
    # 获取运行开始时记录的文件状态
    tracked_files = _tracked_file_states(run_state)

    actions: list[GitRollbackAction] = []
    for path in affected_paths:
        if _is_internal_path(path):
            actions.append(GitRollbackAction(path=path, action="skip", reason="internal_path"))
            continue
        if path in staged_affected:
            actions.append(
                GitRollbackAction(
                    path=path,
                    action="block",
                    reason="affected_path_has_staged_changes",
                )
            )
            continue
        actions.append(_plan_path_action(root, path, tracked_files.get(path)))

    blockers = [action for action in actions if action.action == "block"]
    if blockers:
        reasons = sorted({str(action.reason or "blocked_action") for action in blockers})
        return GitRollbackPlan(
            status="blocked",
            run_id=run_id,
            reason=reasons[0] if len(reasons) == 1 else "blocked_actions",
            actions=actions,
            ignored_paths=ignored_paths,
        )

    actionable = [action for action in actions if action.action in {"restore", "remove"}]
    if not actionable:
        return GitRollbackPlan(
            status="noop",
            run_id=run_id,
            reason="nothing_to_revert",
            actions=actions,
            ignored_paths=ignored_paths,
        )

    return GitRollbackPlan(
        status="ready",
        run_id=run_id,
        actions=actions,
        ignored_paths=ignored_paths,
    )


def revert_run_changes(
    workspace_dir: str | Path,
    run_state: dict[str, Any],
) -> GitRollbackResult:
    """执行运行回滚 —— 根据计划执行文件恢复/删除操作。

    先通过 plan_run_rollback 生成计划，然后执行：
    - restore: git restore --worktree
    - remove: 直接删除文件

    参数:
        workspace_dir: 工作区目录
        run_state: 运行状态的字典表示

    返回:
        GitRollbackResult 包含执行结果
    """
    root = Path(workspace_dir)
    plan = plan_run_rollback(root, run_state)
    if plan.status == "not_eligible":
        return GitRollbackResult(status="not_eligible", run_id=plan.run_id, reason=plan.reason)
    if plan.status == "blocked":
        return GitRollbackResult(
            status="conflict",
            run_id=plan.run_id,
            reason=plan.reason,
            conflicted_paths=[action.path for action in plan.actions if action.action == "block"],
        )
    if plan.status == "noop":
        return GitRollbackResult(status="noop", run_id=plan.run_id, reason=plan.reason)

    restored: list[str] = []
    removed: list[str] = []
    for action in plan.actions:
        path = action.path
        if action.action == "restore":
            result = _git(root, "restore", "--worktree", "--", path)
            if result.returncode != 0:
                return GitRollbackResult(
                    status="conflict",
                    run_id=plan.run_id,
                    reason="git_restore_failed",
                    conflicted_paths=[path],
                )
            restored.append(path)
            continue
        if action.action == "remove":
            absolute = (root / path).resolve()
            if absolute.exists():
                absolute.unlink()
                removed.append(path)

    return GitRollbackResult(
        status="reverted" if restored or removed else "noop",
        run_id=plan.run_id,
        reason=None if restored or removed else "nothing_to_revert",
        restored_paths=restored,
        removed_paths=removed,
    )


# =========================================================================
# 内部类型和辅助函数
# =========================================================================


@dataclass(frozen=True)
class _StatusEntry:
    """Git 状态条目。

    参数:
        path: 文件路径
        staged: 是否已暂存（git add 过）
    """
    path: str
    staged: bool


def _plan_path_action(
    root: Path,
    path: str,
    expected_state: dict[str, Any] | None,
) -> GitRollbackAction:
    """规划对单个文件路径的回滚行动。

    决策树:
    1. 如果路径在工作区外 → block
    2. 如果是 Git 跟踪文件：
       a. 如果文件自运行后又发生了变化 → block
       b. 否则 → restore
    3. 如果是未跟踪文件（运行期间创建的）：
       a. 如果有运行时的状态记录：
          - 运行时文件已存在且当前还存在 → 检查哈希是否变化
          - 运行时文件已存在但当前已不存在 → skip
          - 运行时文件不存在但当前存在 → block（说明是之后创建的）
       b. 如果没有状态记录：
          - 如果当前文件存在 → remove
          - 如果当前文件不存在 → skip

    参数:
        root: 工作区根目录
        path: 文件路径
        expected_state: 运行开始时记录的文件状态（可选）

    返回:
        GitRollbackAction 行动决策
    """
    absolute = (root / path).resolve()
    root_resolved = root.resolve()
    if not _is_relative_to(absolute, root_resolved):
        return GitRollbackAction(path=path, action="block", reason="path_outside_workspace")

    if _is_git_tracked(root, path):
        if expected_state is not None and _file_changed_since_state(root, path, expected_state):
            return GitRollbackAction(
                path=path,
                action="block",
                reason="affected_file_changed_after_run",
            )
        return GitRollbackAction(path=path, action="restore", reason="tracked_file")

    if expected_state is not None:
        expected_exists = bool(expected_state.get("exists"))
        if expected_exists:
            if not absolute.exists():
                return GitRollbackAction(path=path, action="skip", reason="already_missing")
            current = file_state_for_path(root, path)
            if current.get("sha256") != expected_state.get("sha256"):
                return GitRollbackAction(
                    path=path,
                    action="block",
                    reason="affected_file_changed_after_run",
                )
        elif absolute.exists():
            return GitRollbackAction(
                path=path,
                action="block",
                reason="affected_file_changed_after_run",
            )

    if not absolute.exists():
        return GitRollbackAction(path=path, action="skip", reason="already_missing")
    if not absolute.is_file():
        return GitRollbackAction(path=path, action="block", reason="untracked_path_not_file")
    return GitRollbackAction(path=path, action="remove", reason="untracked_file")


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """运行 git 命令。"""
    return subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _stdout(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout.strip()


def _visible_status(root: Path) -> str:
    """获取排除 .codepilot 后的 git status。"""
    return "\n".join(
        line
        for line in _git_status_lines(root)
        if not _is_internal_path(_status_path(line))
    )


def _status_entries(root: Path) -> list[_StatusEntry]:
    entries: list[_StatusEntry] = []
    for line in _git_status_lines(root):
        if len(line) < 3:
            continue
        entries.append(_StatusEntry(path=_status_path(line), staged=line[0] not in {" ", "?"}))
    return entries


def _git_status_lines(root: Path) -> list[str]:
    result = _git(root, "status", "--porcelain", "--untracked-files=normal", "--", ".")
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def _status_path(line: str) -> str:
    path = line[3:] if len(line) > 3 else ""
    if " -> " in path:
        path = path.split(" -> ")[-1]
    return _normalize_path(path)


def _is_git_tracked(root: Path, path: str) -> bool:
    return _git(root, "ls-files", "--error-unmatch", "--", path).returncode == 0


def _tracked_file_states(run_state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """从运行元数据中提取运行结束时的文件状态。"""
    tracked: dict[str, dict[str, Any]] = {}
    rollback = run_state.get("rollback")
    values = rollback.get("post_run_files") if isinstance(rollback, dict) else None
    if isinstance(values, dict):
        for path, item in values.items():
            if isinstance(path, str) and isinstance(item, dict):
                tracked[_normalize_path(path)] = item
    return tracked


def _file_changed_since_state(root: Path, path: str, expected: dict[str, Any]) -> bool:
    """检查文件自录制状态后是否发生了变化。"""
    current = file_state_for_path(root, path)
    expected_exists = bool(expected.get("exists"))
    if bool(current.get("exists")) != expected_exists:
        return True
    if not expected_exists:
        return False
    return current.get("sha256") != expected.get("sha256")


def _normalize_paths(paths: list[str]) -> list[str]:
    return sorted({_normalize_path(path) for path in paths if path})


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").strip().strip('"')


def _is_internal_path(path: str) -> bool:
    """检查是否为 .codepilot 内部文件。"""
    normalized = _normalize_path(path)
    return normalized == ".codepilot" or normalized.startswith(".codepilot/")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _ensure_rollback_status(value: object) -> RollbackStatus:
    if value not in _ROLLBACK_STATUSES:
        raise ValueError(f"Unknown rollback status: {value}")
    return cast(RollbackStatus, value)


def _ensure_rollback_plan_status(value: object) -> RollbackPlanStatus:
    if value not in _ROLLBACK_PLAN_STATUSES:
        raise ValueError(f"Unknown rollback plan status: {value}")
    return cast(RollbackPlanStatus, value)


def _ensure_rollback_action(value: object) -> RollbackAction:
    if value not in _ROLLBACK_ACTIONS:
        raise ValueError(f"Unknown rollback action: {value}")
    return cast(RollbackAction, value)


__all__ = [
    "GitRollbackAction",
    "GitRollbackBaseline",
    "GitRollbackPlan",
    "GitRollbackResult",
    "build_rollback_metadata",
    "capture_git_baseline",
    "plan_run_rollback",
    "revert_run_changes",
]
