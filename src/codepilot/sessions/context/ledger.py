from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from codepilot.llm.estimation import estimate_context_tokens
from codepilot.protocols import ContextArtifactRef, ContextReport, TextContent, ToolResultMessage
from codepilot.sessions.storage import SessionLayout


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
    def from_dict(cls, payload: dict[str, Any]) -> "ToolLedgerEntry":
        artifact = payload.get("artifact") if isinstance(payload.get("artifact"), dict) else {}
        verification = payload.get("verification")
        return cls(
            tool_call_id=str(payload.get("tool_call_id") or ""),
            run_id=payload.get("run_id") if isinstance(payload.get("run_id"), str) else None,
            tool_name=str(payload.get("tool_name") or ""),
            status=str(payload.get("status") or "success"),
            artifact=ContextArtifactRef(
                kind=str(artifact.get("kind") or "tool_output"),
                path=str(artifact.get("path") or ""),
                source_hash=artifact.get("source_hash") if isinstance(artifact.get("source_hash"), str) else None,
                summary=str(artifact.get("summary") or ""),
                original_tokens=_int(artifact.get("original_tokens")),
                visible_tokens=_int(artifact.get("visible_tokens")),
            ),
            affected_paths=[
                str(path)
                for path in payload.get("affected_paths", [])
                if isinstance(path, str)
            ],
            verification=verification if isinstance(verification, dict) else None,
            error_code=payload.get("error_code") if isinstance(payload.get("error_code"), str) else None,
        )


class ContextLedger:
    """Append a compact audit record for every prepared context."""

    def __init__(self, *, workspace_dir: str | Path, session_id: str) -> None:
        self.file = SessionLayout.for_workspace(workspace_dir, session_id).context_ledger_file

    def append_projection(self, report: ContextReport, *, run_id: str | None) -> None:
        payload = {
            "type": "context_projection",
            "context_id": report.context_id,
            "run_id": run_id,
            "pressure": asdict(report.pressure) if report.pressure else None,
            "tokens_by_layer": dict(report.tokens_by_layer),
            "selected_items": list(report.selected_items),
            "dropped_items": [asdict(item) for item in report.dropped_items],
            "memory_ids": list(report.retrieved_memory_ids),
            "artifact_refs": [asdict(item) for item in report.artifact_refs],
            "prefix_hash": report.prefix_hash,
            "dynamic_hash": report.dynamic_hash,
        }
        self.file.parent.mkdir(parents=True, exist_ok=True)
        with self.file.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class ToolArtifactLedger:
    """Archive large tool outputs and expose short references to context prep."""

    def __init__(self, *, workspace_dir: str | Path, session_id: str) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.session_id = session_id
        self.layout = SessionLayout.for_workspace(self.workspace_dir, self.session_id)
        self.ledger_file = self.layout.context_ledger_file
        self.artifact_dir = self.layout.tool_outputs_dir

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

        artifact_path = (
            Path(".codepilot")
            / "sessions"
            / self.session_id
            / "artifacts"
            / "tool_outputs"
            / f"{_safe_stem(message.tool_call_id)}_{digest[:12]}.txt"
        )
        target = self.workspace_dir / artifact_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")

        summary = _summary_for_tool_result(message, text)
        visible = _projection_text(message, artifact_path.as_posix(), summary)
        entry = ToolLedgerEntry(
            tool_call_id=message.tool_call_id,
            run_id=run_id,
            tool_name=message.tool_name,
            status=message.status,
            artifact=ContextArtifactRef(
                kind="tool_output",
                path=artifact_path.as_posix(),
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
            ),
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
            if isinstance(payload, dict) and payload.get("type") == "tool_artifact":
                entries.append(ToolLedgerEntry.from_dict(payload))
        return entries

    def artifact_refs(self) -> list[ContextArtifactRef]:
        return [entry.artifact for entry in self.load_entries()]

    def _entry_for_call(self, tool_call_id: str) -> ToolLedgerEntry | None:
        for entry in reversed(self.load_entries()):
            if entry.tool_call_id == tool_call_id:
                return entry
        return None

    def _append(self, entry: ToolLedgerEntry) -> None:
        self.ledger_file.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_file.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"type": "tool_artifact", **entry.to_dict()}, ensure_ascii=False) + "\n")


def _tool_text(message: ToolResultMessage) -> str:
    return "".join(getattr(block, "text", "") for block in message.content)


def _safe_stem(value: str) -> str:
    raw = value or "tool"
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)[:64]


def _summary_for_tool_result(message: ToolResultMessage, text: str) -> str:
    compact = " ".join(text.strip().split())
    if len(compact) > 280:
        compact = f"{len(text.splitlines())} lines, {len(text)} chars archived"
    paths = ", ".join(message.affected_paths)
    prefix = f"{message.tool_name} status={message.status}"
    if paths:
        prefix += f" paths={paths}"
    return f"{prefix}: {compact}" if compact else prefix


def _projection_text(message: ToolResultMessage, artifact_path: str, summary: str) -> str:
    lines = [
        "[Tool output archived]",
        f"tool={message.tool_name}",
        f"status={message.status}",
        f"artifact={artifact_path}",
        f"summary={summary}",
    ]
    if message.verification:
        lines.append(f"verification={message.verification}")
    return "\n".join(lines)


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


__all__ = ["ContextLedger", "ToolArtifactLedger", "ToolLedgerEntry"]
