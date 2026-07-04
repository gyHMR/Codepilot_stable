from __future__ import annotations

# 新手导读：audit.py 只记录钉钉入口层的远程控制审计，不参与 Agent 决策。
# 关注点：审计内容必须脱敏、短小、可排障，不能保存完整 prompt、stdout 或密钥。

"""DingTalk interface audit logging."""

import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

from codepilot.observability import redact_artifact

from .schemas import DingTalkInboundMessage


AUDIT_EVENTS = frozenset(
    {
        "message_received",
        "message_rejected",
        "run_accepted",
        "run_finished",
        "approval_requested",
        "approval_received",
        "approval_finished",
        "cancel_requested",
        "status_requested",
        "bridge_error",
    }
)

AUDIT_FIELDS = (
    "event",
    "timestamp_ms",
    "message_id",
    "sender_id",
    "conversation_id",
    "session_id",
    "run_id",
    "approval_id",
    "command",
    "status",
    "reason",
    "workspace_state",
)


class DingTalkAuditLogger:
    """Append DingTalk bridge audit records as JSONL under the workspace."""

    def __init__(self, workspace_dir: str | Path) -> None:
        self.path = Path(workspace_dir) / ".codepilot" / "dingtalk" / "audit.jsonl"

    def record(
        self,
        event: str,
        *,
        inbound: DingTalkInboundMessage | None = None,
        command: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        approval_id: str | None = None,
        status: str | None = None,
        reason: object | None = None,
        workspace_state: str | None = None,
    ) -> None:
        """Write one audit record and never interrupt the bridge on failure."""

        try:
            record = _record(
                event,
                inbound=inbound,
                command=command,
                session_id=session_id,
                run_id=run_id,
                approval_id=approval_id,
                status=status,
                reason=reason,
                workspace_state=workspace_state,
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        except Exception as exc:  # pragma: no cover - best-effort local audit
            warning = redact_artifact(str(exc))
            print(f"codepilot-dingtalk audit warning: {warning}", file=sys.stderr)


def _record(
    event: str,
    *,
    inbound: DingTalkInboundMessage | None,
    command: str | None,
    session_id: str | None,
    run_id: str | None,
    approval_id: str | None,
    status: str | None,
    reason: object | None,
    workspace_state: str | None,
) -> dict[str, object | None]:
    if event not in AUDIT_EVENTS:
        raise ValueError(f"Unknown DingTalk audit event: {event}")
    record: dict[str, object | None] = {
        "event": event,
        "timestamp_ms": int(time.time() * 1000),
        "message_id": inbound.message_id if inbound else None,
        "sender_id": _hash_sender(inbound.sender_id) if inbound else None,
        "conversation_id": inbound.conversation_id if inbound else None,
        "session_id": _clean(session_id),
        "run_id": _clean(run_id),
        "approval_id": _clean(approval_id),
        "command": _clean(command),
        "status": _clean(status),
        "reason": _short_reason(reason),
        "workspace_state": _clean(workspace_state),
    }
    redacted = redact_artifact(record)
    if not isinstance(redacted, dict):
        raise TypeError("DingTalk audit record redaction returned non-dict value")
    return {field: redacted.get(field) for field in AUDIT_FIELDS}


def _hash_sender(sender_id: str) -> str:
    digest = hashlib.sha256(sender_id.encode("utf-8")).hexdigest()[:16]
    return f"sha256:{digest}"


def _clean(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _short_reason(value: object | None) -> str | None:
    text = _clean(value)
    if text is None:
        return None
    if len(text) <= 240:
        return text
    return text[:226].rstrip() + "...[truncated]"


__all__ = ["AUDIT_EVENTS", "AUDIT_FIELDS", "DingTalkAuditLogger"]
