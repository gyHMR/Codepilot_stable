from __future__ import annotations

import subprocess
from pathlib import Path


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


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
        ),
    }
    result = revert_run_changes(tmp_path, run_state)

    assert result.status == "reverted"
    assert target.read_text(encoding="utf-8") == "before\n"


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
