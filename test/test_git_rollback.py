from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _init_repo(root: Path) -> None:
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test User")


def _model():
    from codepilot.protocols import Model

    return Model(
        id="test-model",
        name="Test Model",
        api="openai-compatible",
        provider="test",
        base_url="http://localhost",
        reasoning=False,
        input=["text"],
        context_window=1000,
        max_tokens=100,
    )


def _session(root: Path):
    from codepilot.sessions import AgentSession
    from codepilot.sessions.types import AgentSessionOptions

    return AgentSession(
        AgentSessionOptions(
            model=_model(),
            workspace_dir=root,
            session_id="session_git",
            memory_enabled=False,
        )
    )


def _append_run(session, run_id: str, affected_paths: list[str]) -> None:
    from codepilot.protocols import (
        AgentRunCounters,
        AgentRunResult,
        AssistantMessage,
        TextContent,
    )

    final = AssistantMessage(content=[TextContent(text="done")])
    result = AgentRunResult(
        run_id=run_id,
        session_id=session.session_id,
        status="completed",
        stop_reason="final_answer",
        counters=AgentRunCounters(tool_calls=1),
        messages=[final],
        final_message=final,
        affected_paths=affected_paths,
        workspace_changed=True,
    )
    session.store.append_run_result(result)


def test_git_rollback_result_rejects_unknown_status() -> None:
    from codepilot.sessions.history.git_rollback import GitRollbackResult

    with pytest.raises(ValueError, match="Unknown rollback status"):
        GitRollbackResult(status="partial", run_id="run_1")  # type: ignore[arg-type]


def test_git_rollback_reverts_tracked_file_and_removes_new_file(tmp_path: Path) -> None:
    from codepilot.sessions.history.git_rollback import (
        build_rollback_metadata,
        capture_git_baseline,
    )

    _init_repo(tmp_path)
    tracked = tmp_path / "app.py"
    tracked.write_text("print('before')\n", encoding="utf-8")
    _git(tmp_path, "add", "app.py")
    _git(tmp_path, "commit", "-m", "baseline")

    baseline = capture_git_baseline(tmp_path)
    assert baseline.eligible is True

    session = _session(tmp_path)
    tracked.write_text("print('after')\n", encoding="utf-8")
    new_file = tmp_path / "generated.txt"
    new_file.write_text("generated\n", encoding="utf-8")
    _append_run(session, "run_rollback", ["app.py", "generated.txt"])
    session.store.write_rollback_metadata(
        "run_rollback",
        build_rollback_metadata(
            baseline,
            affected_paths=["app.py", "generated.txt"],
            workspace_changed=True,
        ),
    )

    result = session.revert_last_run()

    assert result.status == "reverted"
    assert result.restored_paths == ["app.py"]
    assert result.removed_paths == ["generated.txt"]
    assert tracked.read_text(encoding="utf-8") == "print('before')\n"
    assert not new_file.exists()


def test_git_rollback_marks_dirty_baseline_not_eligible(tmp_path: Path) -> None:
    from codepilot.sessions.history.git_rollback import capture_git_baseline

    _init_repo(tmp_path)
    tracked = tmp_path / "app.py"
    tracked.write_text("print('before')\n", encoding="utf-8")
    _git(tmp_path, "add", "app.py")
    _git(tmp_path, "commit", "-m", "baseline")
    tracked.write_text("print('user change')\n", encoding="utf-8")

    baseline = capture_git_baseline(tmp_path)

    assert baseline.eligible is False
    assert baseline.reason == "dirty_worktree_before_run"


def test_git_rollback_rejects_affected_file_changed_after_run(tmp_path: Path) -> None:
    from codepilot.sessions.history.git_rollback import (
        build_rollback_metadata,
        capture_git_baseline,
    )

    _init_repo(tmp_path)
    tracked = tmp_path / "app.py"
    tracked.write_text("print('before')\n", encoding="utf-8")
    _git(tmp_path, "add", "app.py")
    _git(tmp_path, "commit", "-m", "baseline")

    baseline = capture_git_baseline(tmp_path)
    session = _session(tmp_path)
    tracked.write_text("print('agent change')\n", encoding="utf-8")
    _append_run(session, "run_conflict", ["app.py"])
    session.store.write_rollback_metadata(
        "run_conflict",
        build_rollback_metadata(
            baseline,
            affected_paths=["app.py"],
            workspace_changed=True,
        ),
    )
    tracked.write_text("print('manual change')\n", encoding="utf-8")

    result = session.revert_last_run()

    assert result.status == "conflict"
    assert result.reason == "affected_file_changed_after_run"
    assert result.conflicted_paths == ["app.py"]
    assert tracked.read_text(encoding="utf-8") == "print('manual change')\n"
