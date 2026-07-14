from __future__ import annotations

import subprocess
from pathlib import Path


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_missing_workspace_checkpoint_is_unknown_not_unchanged(tmp_path: Path) -> None:
    from codepilot.sessions.workspace import validate_workspace_checkpoint

    state = validate_workspace_checkpoint(tmp_path, None)

    assert state.status == "unknown"


def test_workspace_checkpoint_ignores_internal_state_in_nested_workspace(tmp_path: Path) -> None:
    from codepilot.sessions.workspace import capture_workspace_checkpoint, validate_workspace_checkpoint

    _git(tmp_path, "init")
    nested = tmp_path / "nested"
    nested.mkdir()
    checkpoint = capture_workspace_checkpoint(nested)
    internal = nested / ".codepilot" / "runs"
    internal.mkdir(parents=True)
    (internal / "run.json").write_text("{}\n", encoding="utf-8")

    state = validate_workspace_checkpoint(nested, checkpoint)

    assert state.status == "unchanged"


def test_rollback_service_restores_tracked_file(tmp_path: Path) -> None:
    from codepilot.sessions.rollback import build_rollback_metadata, capture_git_baseline, revert_run_changes

    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    target = tmp_path / "app.py"
    target.write_text("before\n", encoding="utf-8")
    _git(tmp_path, "add", "app.py")
    _git(tmp_path, "commit", "-m", "baseline")

    baseline = capture_git_baseline(tmp_path)
    target.write_text("after\n", encoding="utf-8")
    run_state = {
        "run_id": "run_test",
        "rollback": build_rollback_metadata(
            baseline,
            affected_paths=["app.py"],
            workspace_changed=True,
            workspace_dir=tmp_path,
        ),
    }
    result = revert_run_changes(tmp_path, run_state)

    assert result.status == "reverted"
    assert target.read_text(encoding="utf-8") == "before\n"


def test_rollback_blocks_unstaged_user_change_after_run(tmp_path: Path) -> None:
    from codepilot.sessions.rollback import build_rollback_metadata, capture_git_baseline, revert_run_changes

    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    target = tmp_path / "app.py"
    target.write_text("before\n", encoding="utf-8")
    _git(tmp_path, "add", "app.py")
    _git(tmp_path, "commit", "-m", "baseline")

    baseline = capture_git_baseline(tmp_path)
    target.write_text("agent change\n", encoding="utf-8")
    metadata = build_rollback_metadata(
        baseline,
        affected_paths=["app.py"],
        workspace_changed=True,
        workspace_dir=tmp_path,
    )
    target.write_text("user change after run\n", encoding="utf-8")

    result = revert_run_changes(tmp_path, {"run_id": "run_test", "rollback": metadata})

    assert result.status == "conflict"
    assert result.reason == "affected_file_changed_after_run"
    assert target.read_text(encoding="utf-8") == "user change after run\n"


def test_rollback_service_rejects_dirty_baseline(tmp_path: Path) -> None:
    from codepilot.sessions.rollback import capture_git_baseline

    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    target = tmp_path / "app.py"
    target.write_text("before\n", encoding="utf-8")
    _git(tmp_path, "add", "app.py")
    _git(tmp_path, "commit", "-m", "baseline")
    target.write_text("dirty\n", encoding="utf-8")
    baseline = capture_git_baseline(tmp_path)

    assert baseline.eligible is False
    assert baseline.reason == "dirty_worktree_before_run"
