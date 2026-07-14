from __future__ import annotations

import subprocess
from pathlib import Path

from codepilot.protocols import TextContent, ToolResultMessage
from codepilot.sessions.context.state import ContextState, RepositoryTracker


def test_repository_tracker_detects_external_dirty_file_changes(tmp_path: Path) -> None:
    tracked = tmp_path / "app.py"
    tracked.write_text("value = 1\n", encoding="utf-8", newline="\n")
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "add", "app.py")
    _git(tmp_path, "commit", "-m", "initial")
    tracker = RepositoryTracker(tmp_path)
    first = tracker.snapshot()

    tracked.write_text("value = 2\n", encoding="utf-8", newline="\n")
    second, first_delta = tracker.refresh(first)
    tracked.write_text("value = 3\n", encoding="utf-8", newline="\n")
    third, second_delta = tracker.refresh(second)

    assert first.fingerprint != second.fingerprint != third.fingerprint
    assert "app.py" in first_delta.modified_paths
    assert "app.py" in second_delta.modified_paths


def test_repository_tracker_ignores_codepilot_artifacts(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    tracker = RepositoryTracker(tmp_path)
    first = tracker.snapshot()
    artifact = tmp_path / ".codepilot" / "runs" / "run_1" / "artifacts" / "output.txt"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("output", encoding="utf-8")

    second, delta = tracker.refresh(first)

    assert first.fingerprint == second.fingerprint
    assert not delta.changed


def test_context_state_links_tool_evidence_to_committed_message(tmp_path: Path) -> None:
    state = ContextState(workspace_dir=tmp_path)
    result = ToolResultMessage(
        tool_call_id="read_1",
        tool_name="read",
        content=[TextContent(text="print('hello')")],
        status="success",
        metadata={
            "session_message_id": "msg_result",
            "read_paths": ["src/app.py"],
            "file_state": {"path": "src/app.py", "sha256": "abc"},
        },
    )

    state.observe_messages((result,), repository_fingerprint="fp_1")

    assert state.evidence["message:msg_result"].source_tool_call_id == "read_1"
    assert state.active_files["src/app.py"].role == "target"
    assert state.active_files["src/app.py"].source_hash == "abc"


def test_context_state_caps_active_files_without_dropping_targets(tmp_path: Path) -> None:
    state = ContextState(workspace_dir=tmp_path, max_active_files=2)
    state.touch_file("docs/reference.md", role="reference", reason="read")
    state.touch_file("src/current.py", role="target", reason="edit")
    state.touch_file("docs/other.md", role="reference", reason="read")

    assert "src/current.py" in state.active_files
    assert len(state.active_files) == 2


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
