from __future__ import annotations

from pathlib import Path


def test_repository_tracker_detects_external_dirty_file_changes(tmp_path: Path) -> None:
    from codepilot.sessions.context.repository_tracker import RepositoryTracker

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

    assert first.fingerprint != second.fingerprint
    assert second.fingerprint != third.fingerprint
    assert "app.py" in first_delta.modified_paths
    assert "app.py" in second_delta.modified_paths


def test_repository_tracker_ignores_codepilot_internal_artifacts(tmp_path: Path) -> None:
    from codepilot.sessions.context.repository_tracker import RepositoryTracker

    tracked = tmp_path / "app.py"
    tracked.write_text("value = 1\n", encoding="utf-8", newline="\n")
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "add", "app.py")
    _git(tmp_path, "commit", "-m", "initial")

    tracker = RepositoryTracker(tmp_path)
    first = tracker.snapshot()
    artifact = (
        tmp_path
        / ".codepilot"
        / "sessions"
        / "session_1"
        / "artifacts"
        / "tool_outputs"
        / "call_1.txt"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text("tool output\n", encoding="utf-8")
    second, delta = tracker.refresh(first)

    assert first.fingerprint == second.fingerprint
    assert not delta.changed
    assert ".codepilot/" not in second.top_level_entries


def test_context_state_records_reject_unknown_enum_values() -> None:
    import pytest
    from codepilot.sessions.context.state import ActiveFile, ContextEvidence, FileSummary

    with pytest.raises(ValueError, match="Unknown active file role"):
        ActiveFile(path="src/app.py", role="scratch", reason="bad role")

    with pytest.raises(ValueError, match="Unknown context freshness"):
        FileSummary(
            path="src/app.py",
            summary="summary",
            source_hash="hash",
            freshness="expired",
        )

    with pytest.raises(ValueError, match="Unknown context trust"):
        ContextEvidence(
            kind="tool_result",
            content="content",
            trust="maybe",
            source="read",
        )

    with pytest.raises(ValueError, match="Unknown context evidence kind"):
        ContextEvidence(
            kind="mystery",
            content="content",
            trust="observed",
            source="read",
        )


def test_context_protocol_records_reject_unknown_enum_values() -> None:
    import pytest
    from codepilot.protocols import ContextItem, DroppedContextItem

    with pytest.raises(ValueError, match="Unknown context trust"):
        ContextItem(
            id="bad-trust",
            kind="active_file",
            content="content",
            source="test",
            trust="certain",
            priority=1,
            estimated_tokens=1,
        )

    with pytest.raises(ValueError, match="Unknown context freshness"):
        ContextItem(
            id="bad-freshness",
            kind="active_file",
            content="content",
            source="test",
            trust="observed",
            priority=1,
            estimated_tokens=1,
            freshness="expired",
        )

    with pytest.raises(ValueError, match="Unknown dropped context reason"):
        DroppedContextItem(
            item_id="item-1",
            section="memory",
            reason="too_old",
            source="test",
        )


def test_context_freshness_notice_summarizes_stale_run_files(
    tmp_path: Path,
) -> None:
    from codepilot.protocols import TextContent, UserMessage
    from codepilot.sessions.context.freshness import build_context_freshness_notice
    from codepilot.sessions.persistence import FreshnessResult

    result = FreshnessResult(
        status="stale",
        checked_paths=["src/app.py", "src/missing.py"],
        changed_paths=["src/app.py"],
        missing_paths=["src/missing.py"],
        workspace_path=str(tmp_path),
    )

    notice = build_context_freshness_notice(result)

    assert isinstance(notice, UserMessage)
    assert notice.metadata == {"context_freshness": result.to_event_payload()}
    assert len(notice.content) == 1
    block = notice.content[0]
    assert isinstance(block, TextContent)
    assert "[Context Freshness]" in block.text
    assert "status=stale" in block.text
    assert "changed_files=src/app.py" in block.text
    assert "missing_files=src/missing.py" in block.text
    assert "旧工具结果可能已过期" in block.text


def test_context_freshness_notice_is_absent_for_valid_state(tmp_path: Path) -> None:
    from codepilot.sessions.context.freshness import build_context_freshness_notice
    from codepilot.sessions.persistence import FreshnessResult

    result = FreshnessResult(status="valid", workspace_path=str(tmp_path))

    assert build_context_freshness_notice(result) is None


def test_session_context_state_caps_verification_only_evidence(
    tmp_path: Path,
) -> None:
    from codepilot.protocols import ToolResultMessage
    from codepilot.sessions.context.state import SessionContextState

    state = SessionContextState(workspace_dir=tmp_path)
    for index in range(90):
        state.observe_tool_result(
            ToolResultMessage(
                tool_call_id=f"verify_{index}",
                tool_name="bash",
                verification={
                    "status": "passed",
                    "command": f"pytest #{index}",
                    "exit_code": 0,
                },
            ),
            repository_fingerprint="fp",
        )

    assert len(state.evidence) == 80


def test_session_context_state_caps_active_files_by_relevance(tmp_path: Path) -> None:
    from codepilot.sessions.context.state import SessionContextState

    state = SessionContextState(workspace_dir=tmp_path, max_active_files=3)
    state.touch_file("docs/old.md", role="reference", reason="read")
    state.touch_file("src/service.py", role="target", reason="edit", source_hash="abc")
    state.touch_file("test/test_service.py", role="test", reason="verify", source_hash="def")
    state.touch_file("src/dependency.py", role="dependency", reason="read")

    assert list(state.active_files) == [
        "src/service.py",
        "test/test_service.py",
        "src/dependency.py",
    ]


def test_session_context_state_never_prunes_recent_target_file(tmp_path: Path) -> None:
    from codepilot.sessions.context.state import SessionContextState

    state = SessionContextState(workspace_dir=tmp_path, max_active_files=2)
    state.touch_file("docs/reference.md", role="reference", reason="read")
    state.touch_file("src/current.py", role="target", reason="edit")
    state.touch_file("docs/other.md", role="reference", reason="read")

    assert "src/current.py" in state.active_files
    assert len(state.active_files) == 2


def _git(root: Path, *args: str) -> None:
    import subprocess

    subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
