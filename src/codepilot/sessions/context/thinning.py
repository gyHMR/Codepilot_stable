"""Deterministic, request-scoped Context thinning for tight pressure."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from codepilot.core.transcript import latest_unconsumed_tool_batch, message_groups
from codepilot.llm.estimation import estimate_text_tokens
from codepilot.protocols import Message, TextContent, ToolResultMessage

from .projection import message_source_ref, read_result_identity, tool_text
from .state import ContextState


_MAX_RECENT_TOOL_TOKENS = 40_000
_EXCERPT_CHARS = 500
_THINNABLE_TOOLS = frozenset(
    {
        "bash",
        "command",
        "find",
        "grep",
        "ls",
        "shell",
        "workspace_status",
    }
)


@dataclass(frozen=True)
class ThinningResult:
    messages: tuple[Message, ...]
    actions: tuple[str, ...] = ()


class ContextThinner:
    """Shrink old recoverable outputs while preserving current coding evidence."""

    def __init__(self, *, workspace_dir: str | Path) -> None:
        self.workspace_dir = Path(workspace_dir)

    def thin(
        self,
        messages: tuple[Message, ...],
        *,
        state: ContextState,
        run_id: str,
        target_tokens: int,
        estimate_tokens: Callable[[tuple[Message, ...]], int],
        recent_tool_tokens: int = _MAX_RECENT_TOOL_TOKENS,
    ) -> ThinningResult:
        if estimate_tokens(messages) <= target_tokens:
            return ThinningResult(messages)

        groups = message_groups(messages)
        protected_group = latest_unconsumed_tool_batch(groups)
        protected_calls = _recent_tool_calls(groups, max(0, recent_tool_tokens))
        covered_reads = _covered_read_calls(groups, state)
        actions: list[str] = []
        output = list(messages)
        positions = {id(message): index for index, message in enumerate(messages)}

        for group_index, group in enumerate(groups):
            if group_index == protected_group:
                continue
            for message in group:
                if not isinstance(message, ToolResultMessage):
                    continue
                if message.tool_call_id in protected_calls:
                    continue
                reason = _thinning_reason(message, covered_reads)
                if reason is None:
                    continue
                text = tool_text(message)
                if not text:
                    continue
                artifact_ref = self._archive(run_id, message, text)
                replacement = _thinned_result(message, text, artifact_ref, reason)
                output[positions[id(message)]] = replacement
                actions.append(f"{reason}:{message.tool_call_id}")
                if estimate_tokens(tuple(output)) <= target_tokens:
                    return ThinningResult(tuple(output), tuple(actions))
        return ThinningResult(tuple(output), tuple(actions))

    def _archive(
        self,
        run_id: str,
        message: ToolResultMessage,
        text: str,
    ) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        call_id = _safe_stem(message.tool_call_id)
        relative = (
            Path(".codepilot")
            / "runs"
            / run_id
            / "artifacts"
            / "tool_outputs"
            / f"{call_id}_{digest[:12]}.txt"
        )
        target = self.workspace_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(text, encoding="utf-8", newline="\n")
        return relative.as_posix()


def _thinning_reason(
    message: ToolResultMessage,
    covered_reads: frozenset[str],
) -> str | None:
    if message.tool_call_id in covered_reads:
        return "read_covered"
    if message.tool_name not in _THINNABLE_TOOLS:
        return None
    if message.status != "success" or message.is_error or message.workspace_changed:
        return None
    if message.verification:
        return None
    return "old_tool_output"


def _covered_read_calls(
    groups: tuple[tuple[Message, ...], ...],
    state: ContextState,
) -> frozenset[str]:
    newer: dict[str, list[tuple[str, tuple[int, int]]]] = {}
    covered: set[str] = set()
    for group_index in range(len(groups) - 1, -1, -1):
        for message in groups[group_index]:
            if not isinstance(message, ToolResultMessage):
                continue
            identity = read_result_identity(message)
            if identity is None:
                continue
            path, _display, sha256, read_range = identity
            source_ref = message_source_ref(message, group_index)
            evidence = state.evidence.get(source_ref)
            candidates = newer.get(path, [])
            current_ranges = [item_range for _hash, item_range in candidates]
            has_new_version = any(item_hash != sha256 for item_hash, _range in candidates)
            if (
                candidates
                and _range_covered(read_range, current_ranges)
                and (
                    evidence is None
                    or evidence.freshness in {"stale", "missing"}
                    or has_new_version
                )
            ):
                covered.add(message.tool_call_id)
            if evidence is None or evidence.freshness not in {"stale", "missing"}:
                newer.setdefault(path, []).append((sha256, read_range))
    return frozenset(covered)


def _range_covered(
    target: tuple[int, int],
    ranges: list[tuple[int, int]],
) -> bool:
    if not ranges:
        return False
    cursor = target[0]
    for start, end in sorted(ranges):
        if end < cursor:
            continue
        if start > cursor:
            return False
        cursor = max(cursor, end + 1)
        if cursor > target[1]:
            return True
    return cursor > target[1]


def _recent_tool_calls(
    groups: tuple[tuple[Message, ...], ...],
    token_budget: int,
) -> frozenset[str]:
    protected: set[str] = set()
    used = 0
    for group in reversed(groups):
        results = [message for message in group if isinstance(message, ToolResultMessage)]
        if not results:
            continue
        group_tokens = sum(estimate_text_tokens(tool_text(message)).total for message in results)
        if protected and used + group_tokens > token_budget:
            break
        protected.update(message.tool_call_id for message in results)
        used += group_tokens
    return frozenset(protected)


def _thinned_result(
    message: ToolResultMessage,
    text: str,
    artifact_ref: str,
    reason: str,
) -> ToolResultMessage:
    first, last = _excerpt(text)
    parts = [
        "[Older tool output omitted under tight Context pressure]",
        f"tool={message.tool_name} status={message.status} reason={reason}",
    ]
    if message.exit_code is not None:
        parts.append(f"exit_code={message.exit_code}")
    if message.affected_paths:
        parts.append("paths=" + ", ".join(message.affected_paths[:8]))
    parts.append(f"artifact_ref={artifact_ref}")
    if first:
        parts.extend(["first_excerpt:", first])
    if last and last != first:
        parts.extend(["last_excerpt:", last])
    metadata = dict(message.metadata)
    metadata["artifact_ref"] = artifact_ref
    metadata["context_thinned"] = True
    return replace(
        message,
        content=[TextContent(text="\n".join(parts))],
        metadata=metadata,
    )


def _excerpt(text: str) -> tuple[str, str]:
    if len(text) <= _EXCERPT_CHARS * 2:
        return text, text
    return text[:_EXCERPT_CHARS].rstrip(), text[-_EXCERPT_CHARS:].lstrip()


def _safe_stem(value: str) -> str:
    stem = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )
    return stem[:80] or "tool"


__all__ = ["ContextThinner", "ThinningResult"]
