from __future__ import annotations

import hashlib
import inspect
import json
import uuid
from dataclasses import replace
from pathlib import Path

from codepilot.llm.estimation import estimate_context_tokens, estimate_text_tokens
from codepilot.protocols import AssistantMessage, Message, ToolCall, ToolResultMessage

from .contracts import (
    CompactSnapshotRef,
    CompactSummary,
    ContextCheckpointState,
    ContextSummarizerPort,
    ContextSummaryRequest,
    ContextSummaryResult,
)


class ContextCompactionError(RuntimeError):
    pass


_LEGACY_SOURCE_DIGEST = "legacy:unverified"


class ContextCompactor:
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
            groups = _message_groups(pending)
            if len(groups) < 4:
                if previous_snapshot is not None:
                    return previous_snapshot
                raise ContextCompactionError("no legal message prefix is large enough to compact")
            keep_groups = max(3, (len(groups) * 3 + 9) // 10)
            compact_groups = groups[: max(1, len(groups) - keep_groups)]
            compact_messages = tuple(message for group in compact_groups for message in group)
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
        if snapshot.source_digest == _LEGACY_SOURCE_DIGEST:
            updated = replace(
                snapshot,
                source_digest=digest,
                estimated_tokens_before=estimate_context_tokens(list(source), ""),
            )
            if self.current_summary is None:
                self._clear()
                return False
            self._write_snapshot(updated, self.current_summary)
            self.current_snapshot = updated
            return True
        if digest != snapshot.source_digest:
            self._clear()
            return False
        return True

    def restore_checkpoint_state(self, state: dict[str, object]) -> None:
        if set(state) == {"compact_summary", "compacted_until_message_id"}:
            self._restore_legacy_checkpoint(state)
            return
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

    def _restore_legacy_checkpoint(self, state: dict[str, object]) -> None:
        summary_text = str(state.get("compact_summary") or "").strip()
        cursor = str(state.get("compacted_until_message_id") or "").strip()
        if not summary_text or not cursor:
            raise ValueError("legacy context checkpoint requires summary and cursor")
        digest = hashlib.sha256(
            f"{self.session_id}\n{cursor}\n{summary_text}".encode("utf-8")
        ).hexdigest()
        compact_id = f"compact_legacy_{digest[:12]}"
        relative = (
            Path(".codepilot")
            / "sessions"
            / self.session_id
            / "artifacts"
            / "context"
            / f"{compact_id}.json"
        )
        tokens = estimate_text_tokens(summary_text).total
        snapshot = CompactSnapshotRef(
            compact_id=compact_id,
            path=relative.as_posix(),
            compacted_until_message_id=cursor,
            source_digest=_LEGACY_SOURCE_DIGEST,
            estimated_tokens_before=tokens,
            estimated_tokens_after=tokens,
        )
        summary = CompactSummary(
            original_goal="Restored legacy session context",
            important_evidence=(summary_text,),
            source_refs=(f"message:{cursor}",),
        )
        self._write_snapshot(snapshot, summary)
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


def _message_groups(messages: tuple[Message, ...]) -> list[tuple[Message, ...]]:
    groups: list[tuple[Message, ...]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if isinstance(message, AssistantMessage):
            call_ids = {
                block.id for block in message.content if isinstance(block, ToolCall)
            }
            if call_ids:
                group: list[Message] = [message]
                cursor = index + 1
                while cursor < len(messages):
                    candidate = messages[cursor]
                    if not isinstance(candidate, ToolResultMessage):
                        break
                    if candidate.tool_call_id not in call_ids:
                        break
                    group.append(candidate)
                    cursor += 1
                groups.append(tuple(group))
                index = cursor
                continue
        groups.append((message,))
        index += 1
    return groups


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
