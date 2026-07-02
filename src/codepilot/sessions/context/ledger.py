from __future__ import annotations

"""工具输出 ledger：保存大输出并为 prompt 生成轻量引用。"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from codepilot.llm.overflow import estimate_context_tokens
from codepilot.protocols import ContextArtifactRef, TextContent, ToolResultMessage


@dataclass(frozen=True)
class ToolLedgerEntry:
    tool_call_id: str
    run_id: str | None
    tool_name: str
    status: str
    artifact: ContextArtifactRef
    affected_paths: list[str]
    verification: dict[str, object] | None
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["artifact"] = asdict(self.artifact)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolLedgerEntry":
        artifact_data = data.get("artifact", {})
        artifact = ContextArtifactRef(
            kind=str(artifact_data.get("kind", "tool_output")),
            path=str(artifact_data.get("path", "")),
            source_hash=(
                str(artifact_data.get("source_hash"))
                if artifact_data.get("source_hash") is not None
                else None
            ),
            summary=str(artifact_data.get("summary", "")),
            original_tokens=int(artifact_data.get("original_tokens", 0)),
            visible_tokens=int(artifact_data.get("visible_tokens", 0)),
        )
        verification = data.get("verification")
        return cls(
            tool_call_id=str(data.get("tool_call_id", "")),
            run_id=data.get("run_id") if isinstance(data.get("run_id"), str) else None,
            tool_name=str(data.get("tool_name", "")),
            status=str(data.get("status", "success")),
            artifact=artifact,
            affected_paths=[
                str(item)
                for item in data.get("affected_paths", [])
                if isinstance(item, str)
            ],
            verification=verification if isinstance(verification, dict) else None,
            error_code=(
                str(data.get("error_code"))
                if data.get("error_code") is not None
                else None
            ),
        )


class ToolArtifactLedger:
    """Session 级工具输出索引与 artifact 存储。"""

    def __init__(self, *, workspace_dir: str | Path, session_id: str) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.session_id = session_id
        self.root = self.workspace_dir / ".codepilot" / "sessions" / session_id
        self.artifact_dir = self.root / "artifacts" / "tool_outputs"
        self.ledger_file = self.root / "tool_ledger.jsonl"

    def record_tool_result(
        self,
        *,
        run_id: str | None,
        message: ToolResultMessage,
    ) -> ToolLedgerEntry:
        text = _tool_text(message)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        existing = self._entry_for_call(message.tool_call_id)
        if existing is not None and existing.artifact.source_hash == digest:
            return existing
        relative = (
            Path(".codepilot")
            / "sessions"
            / self.session_id
            / "artifacts"
            / "tool_outputs"
            / f"{_artifact_stem(message.tool_call_id, digest)}.txt"
        )
        target = self.workspace_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
        summary = _summary_for_tool_result(message, text)
        visible = _projection_text(message, relative.as_posix(), summary)
        artifact = ContextArtifactRef(
            kind="tool_output",
            path=relative.as_posix(),
            source_hash=digest,
            summary=summary,
            original_tokens=estimate_context_tokens([message], ""),
            visible_tokens=estimate_context_tokens(
                [
                    ToolResultMessage(
                        tool_call_id=message.tool_call_id,
                        tool_name=message.tool_name,
                        content=[TextContent(text=visible)],
                        status=message.status,
                    )
                ],
                "",
            ),
        )
        entry = ToolLedgerEntry(
            tool_call_id=message.tool_call_id,
            run_id=run_id,
            tool_name=message.tool_name,
            status=message.status,
            artifact=artifact,
            affected_paths=list(message.affected_paths),
            verification=dict(message.verification) if message.verification else None,
            error_code=message.error_code,
        )
        self._append(entry)
        return entry

    def project_tool_result(
        self,
        message: ToolResultMessage,
        *,
        preserve_full: bool,
    ) -> ToolResultMessage:
        if preserve_full:
            return message
        entry = self._entry_for_call(message.tool_call_id)
        if entry is None:
            entry = self.record_tool_result(run_id=None, message=message)
        text = _projection_text(message, entry.artifact.path, entry.artifact.summary)
        return ToolResultMessage(
            tool_call_id=message.tool_call_id,
            tool_name=message.tool_name,
            content=[TextContent(text=text)],
            status=message.status,
            is_error=message.is_error,
            approved=message.approved,
            approval_id=message.approval_id,
            error_code=message.error_code,
            exit_code=message.exit_code,
            affected_paths=list(message.affected_paths),
            workspace_changed=message.workspace_changed,
            diff_summary=message.diff_summary,
            verification=dict(message.verification) if message.verification else None,
            details=message.details,
            timestamp=message.timestamp,
            metadata={**message.metadata, "artifact_ref": entry.artifact.path},
        )

    def load_entries(self) -> list[ToolLedgerEntry]:
        if not self.ledger_file.exists():
            return []
        entries: list[ToolLedgerEntry] = []
        for line in self.ledger_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                entries.append(ToolLedgerEntry.from_dict(payload))
        return entries

    def artifact_refs(self) -> list[ContextArtifactRef]:
        return [entry.artifact for entry in self.load_entries()]

    def _append(self, entry: ToolLedgerEntry) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.ledger_file.open("a", encoding="utf-8", newline="\n") as fp:
            fp.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def _entry_for_call(self, tool_call_id: str) -> ToolLedgerEntry | None:
        for entry in reversed(self.load_entries()):
            if entry.tool_call_id == tool_call_id:
                return entry
        return None


def _tool_text(message: ToolResultMessage) -> str:
    return "".join(getattr(block, "text", "") for block in message.content)


def _artifact_stem(tool_call_id: str, digest: str) -> str:
    raw = tool_call_id or "tool"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)
    return f"{safe[:64]}_{digest[:12]}"


def _summary_for_tool_result(message: ToolResultMessage, text: str) -> str:
    compact = " ".join(text.strip().split())
    line_count = len(text.splitlines())
    if len(compact) > 400:
        compact = f"{line_count} lines, {len(text)} chars archived"
    elif len(compact) > 240:
        compact = compact[:240].rstrip() + "..."
    paths = ", ".join(message.affected_paths)
    status = f"{message.tool_name} status={message.status}"
    if paths:
        status = f"{status} paths={paths}"
    return f"{status}: {compact}" if compact else status


def _projection_text(
    message: ToolResultMessage,
    artifact_path: str,
    summary: str,
) -> str:
    verification = (
        f"\nverification={message.verification}"
        if message.verification
        else ""
    )
    return (
        "[Tool output archived]\n"
        f"tool={message.tool_name}\n"
        f"status={message.status}\n"
        f"artifact={artifact_path}\n"
        f"summary={summary}"
        f"{verification}"
    )


__all__ = ["ToolArtifactLedger", "ToolLedgerEntry"]
