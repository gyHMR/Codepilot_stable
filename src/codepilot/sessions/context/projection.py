"""Build a provider-safe, lossless projection of canonical session messages."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from codepilot.core.transcript import (
    is_closed_tool_batch,
    latest_unconsumed_tool_batch,
    message_groups,
)
from codepilot.protocols import (
    AssistantMessage,
    Message,
    TextContent,
    ToolCall,
    ToolResultMessage,
)

from .contracts import ProjectedMessage, ProjectionPlan
from .state import ContextEvidence, ContextState


class ContextProjector:
    """Normalize message history without pressure-driven semantic deletion."""

    def __init__(self, *, workspace_dir: str | Path, session_id: str) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.session_id = session_id

    def build(
        self,
        *,
        messages: tuple[Message, ...],
        state: ContextState,
        run_id: str | None = None,
        pressure: str | None = None,
        compacted_until_message_id: str | None = None,
    ) -> ProjectionPlan:
        del run_id, pressure
        visible = _messages_after_cursor(messages, compacted_until_message_id)
        groups = message_groups(visible)
        protected_group = latest_unconsumed_tool_batch(groups)
        read_replacements, omitted_groups = _exact_read_duplicates(
            groups,
            protected_group=protected_group,
        )
        known_call_ids = {
            block.id
            for message in visible
            if isinstance(message, AssistantMessage)
            for block in message.content
            if isinstance(block, ToolCall)
        }
        projected: list[ProjectedMessage] = []
        for index, group in enumerate(groups):
            group_id = f"group:{index}"
            if index in omitted_groups:
                projected.extend(
                    ProjectedMessage(
                        source_ref=_message_source_ref(message, index),
                        message=message,
                        action="covered_by_newer_read",
                        group_id=group_id,
                    )
                    for message in group
                )
                continue
            for message in group:
                source_ref = _message_source_ref(message, index)
                if not isinstance(message, ToolResultMessage):
                    projected.append(
                        ProjectedMessage(source_ref, message, "keep_full", group_id)
                    )
                    continue
                if message.tool_call_id not in known_call_ids:
                    projected.append(
                        ProjectedMessage(source_ref, message, "discard_orphan", group_id)
                    )
                    continue
                marker = read_replacements.get((index, message.tool_call_id))
                if marker is not None and index != protected_group:
                    projected.append(
                        ProjectedMessage(
                            source_ref,
                            replace(message, content=[TextContent(text=marker)]),
                            "keep_projected",
                            group_id,
                        )
                    )
                    continue
                evidence = state.evidence.get(source_ref)
                projected.append(
                    ProjectedMessage(
                        source_ref,
                        _with_freshness_warning(message, evidence),
                        "keep_full",
                        group_id,
                    )
                )
        return ProjectionPlan(messages=tuple(projected))


def read_result_identity(
    message: ToolResultMessage,
) -> tuple[str, str, str, tuple[int, int]] | None:
    """Return normalized path, display path, file hash, and inclusive line range."""

    if message.tool_name != "read" or not isinstance(message.details, dict):
        return None
    path = str(message.details.get("path") or "").strip()
    sha256 = str(message.details.get("sha256") or "").strip()
    try:
        offset = int(message.details.get("offset"))
        returned_lines = int(message.details.get("returned_lines"))
    except (TypeError, ValueError):
        return None
    if not path or not sha256 or offset < 1 or returned_lines < 0:
        return None
    end = offset + max(0, returned_lines - 1)
    normalized_path = os.path.normcase(os.path.normpath(path))
    return normalized_path, Path(path).as_posix(), sha256, (offset, end)


def message_source_ref(message: Message, index: int) -> str:
    return _message_source_ref(message, index)


def tool_text(message: ToolResultMessage) -> str:
    return "\n".join(
        block.text
        for block in message.content
        if isinstance(block, TextContent) and block.text
    )


def _exact_read_duplicates(
    groups: tuple[tuple[Message, ...], ...],
    *,
    protected_group: int | None,
) -> tuple[dict[tuple[int, str], str], frozenset[int]]:
    replacements: dict[tuple[int, str], str] = {}
    omitted_groups: set[int] = set()
    seen: set[tuple[str, str, tuple[int, int]]] = set()

    for index in range(len(groups) - 1, -1, -1):
        group = groups[index]
        if not is_closed_tool_batch(group):
            continue
        results = [
            message for message in group[1:] if isinstance(message, ToolResultMessage)
        ]
        duplicate_ids: set[str] = set()
        for message in results:
            identity = read_result_identity(message)
            if identity is None:
                continue
            normalized_path, display_path, sha256, read_range = identity
            key = (normalized_path, sha256, read_range)
            if key in seen and index != protected_group:
                start, end = read_range
                replacements[(index, message.tool_call_id)] = (
                    f"read result duplicate: path={display_path} range={start}:{end} "
                    "full_content_available_in_newer_result"
                )
                duplicate_ids.add(message.tool_call_id)
            seen.add(key)
        if results and len(duplicate_ids) == len(results):
            omitted_groups.add(index)
    return replacements, frozenset(omitted_groups)


def _with_freshness_warning(
    message: ToolResultMessage,
    evidence: ContextEvidence | None,
) -> ToolResultMessage:
    if evidence is None or evidence.freshness not in {"stale", "missing"}:
        return message
    warning = (
        f"[Historical tool result: freshness={evidence.freshness}. "
        "Keep as historical context; do not treat it as current workspace state.]"
    )
    return replace(message, content=[TextContent(text=warning), *message.content])


def _messages_after_cursor(
    messages: tuple[Message, ...],
    cursor: str | None,
) -> tuple[Message, ...]:
    if not cursor:
        return messages
    for index, message in enumerate(messages):
        if _session_message_id(message) == cursor:
            return messages[index + 1 :]
    return messages


def _message_source_ref(message: Message, index: int) -> str:
    message_id = _session_message_id(message)
    if message_id:
        return f"message:{message_id}"
    if isinstance(message, ToolResultMessage) and message.tool_call_id:
        return f"tool:{message.tool_call_id}"
    return f"transient:{index}:{message.role}"


def _session_message_id(message: Message) -> str | None:
    value = message.metadata.get("session_message_id")
    text = str(value).strip() if value is not None else ""
    return text or None


__all__ = [
    "ContextProjector",
    "message_source_ref",
    "read_result_identity",
    "tool_text",
]
