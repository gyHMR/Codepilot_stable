from __future__ import annotations

# 新手导读：git_rollback 生成轻量回退元数据，帮助用户理解哪些文件可能需要撤销。
# 关注点：它不做隐藏分支或事务快照，只保留学习项目可解释的安全边界。

"""Git clean-worktree 回退。

这个模块刻意只支持一个很小的安全子集：
1. Run 开始前 Git 工作区必须是 clean；
2. 回退时只处理该 Run 记录的 affected_paths；
3. 如果文件在 Run 结束后又被修改，拒绝自动回退。
"""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from codepilot.tools.workspace_safety import file_state_for_path


RollbackStatus = Literal["reverted", "not_eligible", "conflict", "noop"]
RollbackPlanStatus = Literal["ready", "blocked", "not_eligible", "noop"]
RollbackAction = Literal["restore", "remove", "skip", "block"]
_ROLLBACK_STATUSES = frozenset({"reverted", "not_eligible", "conflict", "noop"})
_ROLLBACK_PLAN_STATUSES = frozenset({"ready", "blocked", "not_eligible", "noop"})
_ROLLBACK_ACTIONS = frozenset({"restore", "remove", "skip", "block"})


@dataclass(frozen=True)
class GitRollbackBaseline:
    """Run 开始前的 Git 状态。"""

    eligible: bool
    reason: str | None = None
    head: str | None = None
    branch: str | None = None
    status_before: str = ""


@dataclass(frozen=True)
class GitRollbackResult:
    """一次回退尝试的结果。"""

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
    """Rollback 预览中的单文件动作。"""

    path: str
    action: RollbackAction
    reason: str | None = None

    def __post_init__(self) -> None:
        _ensure_rollback_action(self.action)


@dataclass(frozen=True)
class GitRollbackPlan:
    """一次 run 回退的可执行计划。"""

    status: RollbackPlanStatus
    run_id: str
    reason: str | None = None
    actions: list[GitRollbackAction] = field(default_factory=list)
    ignored_paths: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _ensure_rollback_plan_status(self.status)


def capture_git_baseline(workspace_dir: str | Path) -> GitRollbackBaseline:
    """捕获 Run 开始前的 Git clean-worktree 基线。"""

    root = Path(workspace_dir)
    if _git(root, "rev-parse", "--is-inside-work-tree").returncode != 0:
        return GitRollbackBaseline(
            eligible=False,
            reason="not_git_repo",
        )

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
) -> dict[str, Any]:
    """构建写入 run state 的 rollback 元数据。"""

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
        "workspace_changed": bool(workspace_changed),
    }


def plan_run_rollback(
    workspace_dir: str | Path,
    run_state: dict[str, Any],
) -> GitRollbackPlan:
    """根据 run state 生成回退预览，不修改工作区。"""

    root = Path(workspace_dir)
    run_id = str(run_state.get("run_id") or "")
    rollback = run_state.get("rollback")
    if not isinstance(rollback, dict):
        return GitRollbackPlan(
            status="not_eligible",
            run_id=run_id,
            reason="missing_rollback_metadata",
        )
    if rollback.get("strategy") != "git-clean-worktree":
        return GitRollbackPlan(
            status="not_eligible",
            run_id=run_id,
            reason="unsupported_rollback_strategy",
        )
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

    affected = set(affected_paths)
    status_entries = _status_entries(root)
    ignored_paths = sorted(
        {
            entry.path
            for entry in status_entries
            if not _is_internal_path(entry.path) and entry.path not in affected
        }
    )
    staged_affected = {
        entry.path
        for entry in status_entries
        if entry.staged and not _is_internal_path(entry.path) and entry.path in affected
    }
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
    """根据 run state 撤销本次 Run 对 affected_paths 的修改。"""

    root = Path(workspace_dir)
    plan = plan_run_rollback(root, run_state)
    if plan.status == "not_eligible":
        return GitRollbackResult(
            status="not_eligible",
            run_id=plan.run_id,
            reason=plan.reason,
        )
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


@dataclass(frozen=True)
class _StatusEntry:
    path: str
    staged: bool


def _plan_path_action(
    root: Path,
    path: str,
    expected_state: dict[str, Any] | None,
) -> GitRollbackAction:
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
    lines = [
        line
        for line in _git_status_lines(root)
        if not _is_internal_path(_status_path(line))
    ]
    return "\n".join(lines)


def _status_entries(root: Path) -> list[_StatusEntry]:
    entries: list[_StatusEntry] = []
    for line in _git_status_lines(root):
        if len(line) < 3:
            continue
        entries.append(
            _StatusEntry(
                path=_status_path(line),
                staged=line[0] not in {" ", "?"},
            )
        )
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
    tracked: dict[str, dict[str, Any]] = {}
    for item in run_state.get("tracked_files", []):
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        if isinstance(path, str) and path:
            tracked[_normalize_path(path)] = item
    return tracked


def _file_changed_since_state(root: Path, path: str, expected: dict[str, Any]) -> bool:
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
