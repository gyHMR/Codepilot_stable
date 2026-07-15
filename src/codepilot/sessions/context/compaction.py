"""在上下文高压时生成结构化摘要、artifact 与 checkpoint。"""

from __future__ import annotations

import hashlib
import inspect
import json
import uuid
from pathlib import Path

from codepilot.core.transcript import (
    is_closed_tool_batch,
    latest_unconsumed_tool_batch,
    message_groups,
    tool_batch_call_ids,
)
from codepilot.llm.estimation import estimate_context_tokens, estimate_text_tokens
from codepilot.protocols import AssistantMessage, Message, ToolResultMessage

from .contracts import (
    CompactSnapshotRef,
    CompactSummary,
    ContextCheckpointState,
    ContextSummarizerPort,
    ContextSummaryRequest,
    ContextSummaryResult,
)


class ContextCompactionError(RuntimeError):
    """无法生成满足结构约束的上下文压缩结果。"""
    pass


class ContextCompactor:
    """在压力阈值触发时生成摘要、artifact 和 checkpoint。"""
    def __init__(
        self,
        *,
        workspace_dir: str | Path,
        session_id: str,
        summarizer: ContextSummarizerPort | None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.session_id = session_id
        self.summarizer = summarizer
        self.current_snapshot: CompactSnapshotRef | None = None
        self.current_summary: CompactSummary | None = None

    async def compact(
        self,
        *,
        run_id: str,
        messages: tuple[Message, ...],
        original_goal: str,
    ) -> CompactSnapshotRef | None:
        previous_snapshot = self.current_snapshot
        previous_summary = self.current_summary
        try:
            if self.summarizer is None:
                raise ContextCompactionError("Context summarizer is not configured")
            pending = _messages_after_cursor(
                messages,
                previous_snapshot.compacted_until_message_id
                if previous_snapshot is not None
                else None,
            )
            groups = message_groups(pending)
            _validate_message_groups(groups)
            if len(groups) < 4:
                if previous_snapshot is not None:
                    return previous_snapshot
                raise ContextCompactionError("no legal message prefix is large enough to compact")
            keep_groups = max(3, (len(groups) * 3 + 9) // 10)
            compact_group_count = max(1, len(groups) - keep_groups)
            protected_group = latest_unconsumed_tool_batch(groups)
            if protected_group is not None:
                compact_group_count = min(compact_group_count, protected_group)
            if compact_group_count < 1:
                if previous_snapshot is not None:
                    return previous_snapshot
                raise ContextCompactionError(
                    "no legal message prefix exists before the unconsumed tool batch"
                )
            compact_groups = groups[:compact_group_count]
            compact_messages = tuple(message for group in compact_groups for message in group)
            if any(_message_id(message) is None for message in compact_messages):
                raise ContextCompactionError(
                    "compaction requires committed message ids for the complete prefix"
                )
            expected_cursor = _message_id(compact_messages[-1])
            if expected_cursor is None:
                raise ContextCompactionError("compaction cursor requires committed message ids")
            request = ContextSummaryRequest(
                messages=compact_messages,
                original_goal=original_goal,
                previous_summary=previous_summary,
            )
            result = self.summarizer.summarize(request)
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, ContextSummaryResult):
                raise ContextCompactionError("summarizer returned an invalid result")
            if result.compacted_until_message_id != expected_cursor:
                raise ContextCompactionError("summarizer returned an invalid compaction cursor")
            summary_source_text = _source_text(compact_messages, previous_summary)
            rendered = result.summary.render()
            if len(rendered) >= max(1, int(len(summary_source_text) * 0.85)):
                raise ContextCompactionError("summary did not reach the minimum compression ratio")
            source_messages = _prefix_through_cursor(messages, expected_cursor)
            source_digest = hashlib.sha256(
                _source_text(source_messages, None).encode("utf-8")
            ).hexdigest()
            compact_id = f"compact_{uuid.uuid4().hex[:12]}"
            relative = (
                Path(".codepilot")
                / "runs"
                / run_id
                / "artifacts"
                / "context"
                / f"{compact_id}.json"
            )
            before_tokens = estimate_context_tokens(list(source_messages), "")
            after_tokens = estimate_text_tokens(rendered).total
            snapshot = CompactSnapshotRef(
                compact_id=compact_id,
                path=relative.as_posix(),
                compacted_until_message_id=expected_cursor,
                source_digest=source_digest,
                estimated_tokens_before=before_tokens,
                estimated_tokens_after=after_tokens,
            )
            self._write_snapshot(snapshot, result.summary)
            self.current_snapshot = snapshot
            self.current_summary = result.summary
            return snapshot
        except Exception as exc:
            self.current_snapshot = previous_snapshot
            self.current_summary = previous_summary
            if previous_snapshot is not None:
                return previous_snapshot
            if isinstance(exc, ContextCompactionError):
                raise
            raise ContextCompactionError(str(exc)) from exc

    def checkpoint_state(self) -> dict[str, object]:
        return ContextCheckpointState(
            compact_snapshot_ref=(
                self.current_snapshot.path if self.current_snapshot is not None else None
            ),
            compacted_until_message_id=(
                self.current_snapshot.compacted_until_message_id
                if self.current_snapshot is not None
                else None
            ),
        ).to_mapping()

    def validate_messages(self, messages: tuple[Message, ...]) -> bool:
        _validate_message_groups(message_groups(messages))
        snapshot = self.current_snapshot
        if snapshot is None:
            return True
        try:
            source = _prefix_through_cursor(
                messages,
                snapshot.compacted_until_message_id,
            )
        except ContextCompactionError:
            self._clear()
            return False
        digest = hashlib.sha256(_source_text(source, None).encode("utf-8")).hexdigest()
        if digest != snapshot.source_digest:
            self._clear()
            return False
        return True

    def restore_checkpoint_state(self, state: dict[str, object]) -> None:
        checkpoint = ContextCheckpointState.from_mapping(state)
        if checkpoint.compact_snapshot_ref is None:
            self._clear()
            return
        try:
            payload = json.loads(
                (self.workspace_dir / checkpoint.compact_snapshot_ref).read_text(
                    encoding="utf-8"
                )
            )
            if not isinstance(payload, dict):
                raise ValueError("compact snapshot must be an object")
            allowed = {"snapshot", "summary"}
            if set(payload) != allowed:
                raise ValueError("compact snapshot has unknown fields")
            snapshot_raw = payload["snapshot"]
            summary_raw = payload["summary"]
            if not isinstance(snapshot_raw, dict) or not isinstance(summary_raw, dict):
                raise ValueError("compact snapshot payload is invalid")
            snapshot = _snapshot_from_mapping(snapshot_raw)
            summary = CompactSummary.from_mapping(summary_raw)
            if snapshot.path != checkpoint.compact_snapshot_ref:
                raise ValueError("compact snapshot path mismatch")
            if snapshot.compacted_until_message_id != checkpoint.compacted_until_message_id:
                raise ValueError("compact snapshot cursor mismatch")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            self._clear()
            return
        self.current_snapshot = snapshot
        self.current_summary = summary

    def _write_snapshot(
        self,
        snapshot: CompactSnapshotRef,
        summary: CompactSummary,
    ) -> None:
        target = self.workspace_dir / snapshot.path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "snapshot": {
                "compact_id": snapshot.compact_id,
                "path": snapshot.path,
                "compacted_until_message_id": snapshot.compacted_until_message_id,
                "source_digest": snapshot.source_digest,
                "estimated_tokens_before": snapshot.estimated_tokens_before,
                "estimated_tokens_after": snapshot.estimated_tokens_after,
            },
            "summary": summary.to_mapping(),
        }
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
            newline="\n",
        )
        temp.replace(target)

    def _clear(self) -> None:
        self.current_snapshot = None
        self.current_summary = None


def _snapshot_from_mapping(raw: dict[str, object]) -> CompactSnapshotRef:
    allowed = {
        "compact_id",
        "path",
        "compacted_until_message_id",
        "source_digest",
        "estimated_tokens_before",
        "estimated_tokens_after",
    }
    if set(raw) != allowed:
        raise ValueError("compact snapshot reference fields are invalid")
    return CompactSnapshotRef(
        compact_id=str(raw["compact_id"]),
        path=str(raw["path"]),
        compacted_until_message_id=str(raw["compacted_until_message_id"]),
        source_digest=str(raw["source_digest"]),
        estimated_tokens_before=int(raw["estimated_tokens_before"]),
        estimated_tokens_after=int(raw["estimated_tokens_after"]),
    )


def _validate_message_groups(groups: tuple[tuple[Message, ...], ...]) -> None:
    for group in groups:
        if is_closed_tool_batch(group):
            continue
        if len(group) != 1:
            raise ContextCompactionError("tool call batch is not closed")
        message = group[0]
        if isinstance(message, ToolResultMessage):
            raise ContextCompactionError("orphan tool result is not compactable")
        if isinstance(message, AssistantMessage) and tool_batch_call_ids(group):
            raise ContextCompactionError("tool call batch is not closed")


def _messages_after_cursor(
    messages: tuple[Message, ...],
    cursor: str | None,
) -> tuple[Message, ...]:
    if not cursor:
        return messages
    for index, message in enumerate(messages):
        if _message_id(message) == cursor:
            return messages[index + 1 :]
    return messages


def _prefix_through_cursor(
    messages: tuple[Message, ...],
    cursor: str,
) -> tuple[Message, ...]:
    for index, message in enumerate(messages):
        if _message_id(message) == cursor:
            return messages[: index + 1]
    raise ContextCompactionError("compaction cursor is not present in the message chain")


def _source_text(
    messages: tuple[Message, ...],
    previous_summary: CompactSummary | None,
) -> str:
    parts = [previous_summary.render()] if previous_summary is not None else []
    for message in messages:
        parts.append(repr(message))
    return "\n".join(parts)


def _message_id(message: Message) -> str | None:
    value = message.metadata.get("session_message_id")
    text = str(value).strip() if value is not None else ""
    return text or None


__all__ = [
    "ContextCompactionError",
    "ContextCompactor",
]
