from __future__ import annotations

import asyncio
from pathlib import Path

from codepilot.protocols import UserMessage
from codepilot.sessions.context.compaction import ContextCompactor
from codepilot.sessions.context.contracts import (
    CompactSummary,
    ContextSummaryResult,
)


class _Summarizer:
    def summarize(self, request):
        cursor = str(request.messages[-1].metadata["session_message_id"])
        return ContextSummaryResult(
            summary=CompactSummary(
                original_goal="Goal",
                completed_work=("Work",),
                next_actions=("Next",),
                source_refs=(f"message:{cursor}",),
            ),
            compacted_until_message_id=cursor,
        )


def test_context_compactor_rejects_legacy_inline_summary_checkpoint(tmp_path: Path) -> None:
    compactor = ContextCompactor(
        workspace_dir=tmp_path,
        session_id="session_1",
        summarizer=None,
    )

    try:
        compactor.restore_checkpoint_state(
            {
                "compact_summary": "legacy summary",
                "compacted_until_message_id": "msg_1",
            }
        )
    except ValueError:
        pass
    else:  # pragma: no cover - guards removal of the compatibility path
        raise AssertionError("legacy inline Context checkpoints must be rejected")


def test_checkpoint_restores_snapshot_and_missing_artifact_degrades_to_empty(
    tmp_path: Path,
) -> None:
    messages = tuple(
        UserMessage(
            content=f"message {index} " + ("detail " * 60),
            metadata={"session_message_id": f"msg_{index}"},
        )
        for index in range(8)
    )
    original = ContextCompactor(
        workspace_dir=tmp_path,
        session_id="session_1",
        summarizer=_Summarizer(),
    )
    snapshot = asyncio.run(
        original.compact(run_id="run_1", messages=messages, original_goal="Goal")
    )
    assert snapshot is not None
    checkpoint = original.checkpoint_state()

    restored = ContextCompactor(
        workspace_dir=tmp_path,
        session_id="session_1",
        summarizer=_Summarizer(),
    )
    restored.restore_checkpoint_state(checkpoint)
    assert restored.current_summary is not None
    assert restored.current_snapshot == snapshot

    changed = list(messages)
    changed[0] = UserMessage(
        content="changed source message",
        metadata={"session_message_id": "msg_0"},
    )
    assert restored.validate_messages(tuple(changed)) is False
    assert restored.current_summary is None

    restored.restore_checkpoint_state(checkpoint)

    (tmp_path / snapshot.path).unlink()
    restored.restore_checkpoint_state(checkpoint)
    assert restored.current_summary is None
    assert restored.checkpoint_state() == {
        "compact_snapshot_ref": None,
        "compacted_until_message_id": None,
    }
