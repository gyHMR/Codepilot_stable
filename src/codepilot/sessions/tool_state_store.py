from __future__ import annotations

"""Persistent canonical ToolStateStore owned by one session directory."""

import json
import os
import time
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path
from threading import RLock
from uuid import uuid4

from codepilot.tools.contracts import ToolExecutionRequest
from codepilot.tools.results import (
    ArtifactContent,
    ArtifactRef,
    ImageContent,
    TextContent,
    ToolError,
    ToolResult,
    ToolTiming,
)
from codepilot.tools.security import (
    ApprovalChallenge,
    ApprovalGrant,
    ToolAccessRequest,
    ToolEffect,
    ToolResource,
)
from codepilot.tools.state import (
    InteractionRequest,
    ToolAttemptRecord,
    ToolAttemptState,
    ToolStateConflictError,
)

from .store import SessionLayout, SessionStore


_SCHEMA_VERSION = 1


class SessionToolStateStore:
    """JSON-backed attempt store with atomic compare-and-set transitions."""

    def __init__(self, workspace_dir: str | Path, session_id: str) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.session_id = session_id
        self.path = SessionLayout.for_workspace(workspace_dir, session_id).tool_state_file
        self.lock_path = self.path.with_suffix(".lock")
        self._lock = RLock()

    def create(self, record: ToolAttemptRecord) -> None:
        with self._locked_state() as state:
            attempts = state["attempts"]
            if record.attempt_id in attempts:
                raise ToolStateConflictError(f"Tool attempt already exists: {record.attempt_id}")
            attempts[record.attempt_id] = _record_to_json(record)

    def get(self, attempt_id: str) -> ToolAttemptRecord | None:
        with self._lock:
            payload = self._read_state()["attempts"].get(attempt_id)
        return _record_from_json(payload) if isinstance(payload, dict) else None

    def find_by_approval_id(self, approval_id: str) -> ToolAttemptRecord | None:
        return self._find(lambda item: item.challenge is not None and item.challenge.approval_id == approval_id)

    def find_by_interaction_id(self, interaction_id: str) -> ToolAttemptRecord | None:
        return self._find(
            lambda item: item.interaction is not None
            and item.interaction.interaction_id == interaction_id
        )

    def find_reusable_grant(
        self,
        request: ToolExecutionRequest,
        access: ToolAccessRequest,
    ) -> ApprovalGrant | None:
        wanted_resources = tuple(resource.uri for resource in access.resources)
        with self._lock:
            records = [
                _record_from_json(item)
                for item in self._read_state()["attempts"].values()
                if isinstance(item, dict)
            ]
        for record in reversed(records):
            grant = record.grant
            if grant is None or grant.scope == "once" or record.grant_consumed or grant.expired():
                continue
            if record.request.registration_id != request.registration_id:
                continue
            if grant.scope == "session" and record.request.session_id != request.session_id:
                continue
            if grant.actions != access.actions:
                continue
            if tuple(resource.uri for resource in grant.resources) != wanted_resources:
                continue
            return grant
        return None

    def compare_and_set(
        self,
        attempt_id: str,
        expected_state: ToolAttemptState,
        record: ToolAttemptRecord,
    ) -> None:
        with self._locked_state() as state:
            payload = state["attempts"].get(attempt_id)
            current = _record_from_json(payload) if isinstance(payload, dict) else None
            if current is None or current.state != expected_state:
                actual = current.state if current is not None else "missing"
                raise ToolStateConflictError(
                    f"Tool attempt state conflict: expected {expected_state}, got {actual}"
                )
            if record.attempt_id != attempt_id:
                raise ValueError("replacement attempt_id must match")
            state["attempts"][attempt_id] = _record_to_json(record)

    def pending_challenges(self) -> tuple[ApprovalChallenge, ...]:
        with self._lock:
            records = [
                _record_from_json(item)
                for item in self._read_state()["attempts"].values()
                if isinstance(item, dict)
            ]
        return tuple(
            record.challenge
            for record in records
            if record.state == "awaiting_approval" and record.challenge is not None
        )

    def _find(self, predicate) -> ToolAttemptRecord | None:
        with self._lock:
            values = tuple(self._read_state()["attempts"].values())
        for payload in reversed(values):
            if not isinstance(payload, dict):
                continue
            record = _record_from_json(payload)
            if predicate(record):
                return record
        return None

    @contextmanager
    def _locked_state(self):
        with self._lock, _file_lock(self.lock_path):
            state = self._read_state()
            yield state
            self._write_state(state)

    def _read_state(self) -> dict[str, object]:
        if not self.path.exists():
            session_store = SessionStore(self.workspace_dir, self.session_id)
            if session_store.session_file.exists():
                session_store.read_meta()
            return {"schema_version": _SCHEMA_VERSION, "attempts": {}}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("Unsupported tool state schema; legacy session tool state is not compatible")
        attempts = value.get("attempts")
        if not isinstance(attempts, dict):
            raise ValueError("Invalid canonical tool state: attempts must be an object")
        return {"schema_version": _SCHEMA_VERSION, "attempts": dict(attempts)}

    def _write_state(self, state: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, self.path)


@contextmanager
def _file_lock(path: Path, *, timeout_seconds: float = 2.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise ToolStateConflictError("Timed out acquiring tool state lock")
            time.sleep(0.01)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)


def _record_to_json(record: ToolAttemptRecord) -> dict[str, object]:
    return {
        "attempt_id": record.attempt_id,
        "request": _plain(record.request),
        "state": record.state,
        "challenge": _plain(record.challenge) if record.challenge is not None else None,
        "interaction": _plain(record.interaction) if record.interaction is not None else None,
        "interaction_consumed": record.interaction_consumed,
        "grant": _plain(record.grant) if record.grant is not None else None,
        "grant_consumed": record.grant_consumed,
        "result": _result_to_json(record.result) if record.result is not None else None,
        "cleanup_errors": list(record.cleanup_errors),
    }


def _record_from_json(payload: dict[str, object]) -> ToolAttemptRecord:
    request = ToolExecutionRequest(**dict(payload["request"]))
    challenge_payload = payload.get("challenge")
    interaction_payload = payload.get("interaction")
    grant_payload = payload.get("grant")
    return ToolAttemptRecord(
        attempt_id=str(payload["attempt_id"]),
        request=request,
        state=str(payload.get("state") or "received"),
        challenge=_challenge_from_json(challenge_payload) if isinstance(challenge_payload, dict) else None,
        interaction=InteractionRequest(**interaction_payload) if isinstance(interaction_payload, dict) else None,
        interaction_consumed=bool(payload.get("interaction_consumed")),
        grant=_grant_from_json(grant_payload) if isinstance(grant_payload, dict) else None,
        grant_consumed=bool(payload.get("grant_consumed")),
        result=_result_from_json(payload["result"]) if isinstance(payload.get("result"), dict) else None,
        cleanup_errors=tuple(str(item) for item in payload.get("cleanup_errors", [])),
    )


def _challenge_from_json(payload: dict[str, object]) -> ApprovalChallenge:
    return ApprovalChallenge(
        **{
            **payload,
            "resources": tuple(ToolResource(**item) for item in payload.get("resources", [])),
            "effects": frozenset(payload.get("effects", [])),
            "allowed_scopes": frozenset(payload.get("allowed_scopes", [])),
        }
    )


def _grant_from_json(payload: dict[str, object]) -> ApprovalGrant:
    return ApprovalGrant(
        **{
            **payload,
            "resources": tuple(ToolResource(**item) for item in payload.get("resources", [])),
            "actions": tuple(payload.get("actions", [])),
        }
    )


def _result_to_json(result: ToolResult) -> dict[str, object]:
    return {
        "tool_call_id": result.tool_call_id,
        "tool_name": result.tool_name,
        "status": result.status,
        "content": [_plain(item) for item in result.content],
        "data": _plain(result.data),
        "error": _plain(result.error) if result.error is not None else None,
        "effects": [_plain(item) for item in result.effects],
        "artifacts": [_plain(item) for item in result.artifacts],
        "approval": _plain(result.approval) if result.approval is not None else None,
        "interaction": _plain(result.interaction),
        "timing": _plain(result.timing),
        "registration_id": result.registration_id,
        "output_validation": result.output_validation,
        "content_trust": result.content_trust,
    }


def _result_from_json(payload: dict[str, object]) -> ToolResult:
    content = tuple(_content_from_json(item) for item in payload.get("content", []))
    effects = tuple(_effect_from_json(item) for item in payload.get("effects", []))
    artifacts = tuple(ArtifactRef(**item) for item in payload.get("artifacts", []))
    error_payload = payload.get("error")
    approval_payload = payload.get("approval")
    return ToolResult(
        tool_call_id=str(payload["tool_call_id"]),
        tool_name=str(payload["tool_name"]),
        status=str(payload["status"]),
        content=content,
        data=dict(payload.get("data") or {}),
        error=ToolError(**error_payload) if isinstance(error_payload, dict) else None,
        effects=effects,
        artifacts=artifacts,
        approval=_challenge_from_json(approval_payload) if isinstance(approval_payload, dict) else None,
        interaction=dict(payload["interaction"]) if isinstance(payload.get("interaction"), dict) else None,
        timing=ToolTiming(**dict(payload.get("timing") or {})),
        registration_id=str(payload["registration_id"]),
        output_validation=str(payload.get("output_validation") or "schema_validated"),
        content_trust=str(payload.get("content_trust") or "trusted"),
    )


def _content_from_json(payload: dict[str, object]):
    content_type = payload.get("type")
    if content_type == "text":
        return TextContent(text=str(payload.get("text") or ""))
    if content_type == "image":
        return ImageContent(
            data=str(payload.get("data") or ""),
            mime_type=str(payload.get("mime_type") or "application/octet-stream"),
            name=payload.get("name"),
        )
    artifact = payload.get("artifact")
    if content_type == "artifact" and isinstance(artifact, dict):
        return ArtifactContent(ArtifactRef(**artifact))
    raise ValueError(f"Unknown persisted tool content type: {content_type}")


def _effect_from_json(payload: dict[str, object]) -> ToolEffect:
    resource = payload.get("resource")
    if not isinstance(resource, dict):
        raise ValueError("Persisted tool effect requires resource")
    return ToolEffect(**{**payload, "resource": ToolResource(**resource)})


def _plain(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    return value


__all__ = ["SessionToolStateStore"]
