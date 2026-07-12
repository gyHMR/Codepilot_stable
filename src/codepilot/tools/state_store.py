from __future__ import annotations

"""Tool-owned checkpoint state and workspace-scoped approval grants."""

import json
import os
from pathlib import Path
from threading import RLock
from typing import Mapping, cast
from uuid import uuid4

from .contracts import ToolExecutionRequest
from .security import (
    ApprovalChallenge,
    ApprovalGrant,
    ToolAccessRequest,
    ToolEffect,
    ToolResource,
)
from .results import (
    ArtifactContent,
    ArtifactRef,
    ImageContent,
    TextContent,
    ToolError,
    ToolResult,
    ToolTiming,
)
from .state import (
    InMemoryToolStateStore,
    InteractionRequest,
    ToolAttemptRecord,
)

TOOL_CHECKPOINT_SCHEMA_VERSION = 1
TOOL_GRANT_SCHEMA_VERSION = 1
_CHECKPOINT_KEYS = frozenset({"schema_version", "session_id", "attempts", "intent"})
_GRANT_ROOT_KEYS = frozenset({"schema_version", "grants"})
_TERMINAL_STATES = frozenset(
    {"succeeded", "failed", "denied", "timed_out", "cancelled", "interrupted"}
)
_RISK_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
_GRANT_FILE_LOCKS: dict[Path, RLock] = {}
_GRANT_FILE_LOCKS_GUARD = RLock()


class FileToolGrantStore:
    """Persist reusable session/project grants as Tools security state."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).resolve()
        self._lock = _grant_file_lock(self._path)

    def remember(
        self,
        *,
        session_id: str,
        registration_id: str,
        grant: ApprovalGrant,
    ) -> None:
        if grant.scope not in {"session", "project"}:
            return
        session_key = session_id if grant.scope == "session" else None
        entry = {
            "session_id": session_key,
            "registration_id": _required_text(registration_id, "registration_id"),
            "grant": _grant_to_dict(grant),
        }
        with self._lock:
            entries = self._load()
            entries[grant.grant_id] = entry
            self._write(entries)

    def find_reusable(
        self,
        *,
        session_id: str,
        request: ToolExecutionRequest,
        access: ToolAccessRequest,
    ) -> ApprovalGrant | None:
        wanted_resources = tuple(resource.uri for resource in access.resources)
        with self._lock:
            entries = self._load()
        for item in reversed(tuple(entries.values())):
            registration_id = _required_text(item.get("registration_id"), "registration_id")
            candidate = _grant_from_dict(_mapping(item.get("grant"), "grant"))
            entry_session_id = _optional_text(item.get("session_id"))
            if candidate.expired() or registration_id != request.registration_id:
                continue
            if candidate.scope == "session" and entry_session_id != session_id:
                continue
            if candidate.scope == "project" and entry_session_id is not None:
                continue
            if candidate.actions != access.actions:
                continue
            if tuple(resource.uri for resource in candidate.resources) != wanted_resources:
                continue
            if not access.effects <= candidate.effects:
                continue
            if _RISK_RANK[access.risk] > _RISK_RANK[candidate.risk]:
                continue
            return candidate
        return None

    def _load(self) -> dict[str, dict[str, object]]:
        payload = _read_json_object(self._path)
        if payload is None:
            return {}
        _require_exact_keys(payload, set(_GRANT_ROOT_KEYS), "tool grants")
        if payload.get("schema_version") != TOOL_GRANT_SCHEMA_VERSION:
            raise ValueError("Unsupported tool grant schema version")
        raw_grants = payload.get("grants")
        if not isinstance(raw_grants, list):
            raise ValueError("Tool grants must be a list")
        entries: dict[str, dict[str, object]] = {}
        for item in raw_grants:
            raw = _mapping(item, "tool grant")
            _require_exact_keys(raw, {"session_id", "registration_id", "grant"}, "tool grant")
            grant = _grant_from_dict(_mapping(raw.get("grant"), "grant"))
            session_id = _optional_text(raw.get("session_id"))
            if grant.scope == "session" and session_id is None:
                raise ValueError("Session grants require session_id")
            if grant.scope == "project" and session_id is not None:
                raise ValueError("Project grants cannot contain session_id")
            if grant.scope not in {"session", "project"}:
                raise ValueError("Tool grant file cannot contain once grants")
            entries[grant.grant_id] = dict(raw)
        return entries

    def _write(self, entries: Mapping[str, Mapping[str, object]]) -> None:
        active = [
            dict(item)
            for item in entries.values()
            if not _grant_from_dict(_mapping(item.get("grant"), "grant")).expired()
        ]
        _atomic_write_json(
            self._path,
            {"schema_version": TOOL_GRANT_SCHEMA_VERSION, "grants": active},
        )


class CheckpointToolStateStore(InMemoryToolStateStore):
    """Keep live attempt state in memory and expose opaque checkpoint payloads."""

    def __init__(
        self,
        *,
        session_id: str,
        grant_store: FileToolGrantStore | None = None,
    ) -> None:
        super().__init__()
        self._session_id = _required_text(session_id, "session_id")
        self._grant_store = grant_store

    def create(self, record: ToolAttemptRecord) -> None:
        self._validate_record_session(record)
        super().create(record)

    def compare_and_set(self, attempt_id, expected_state, record) -> None:
        self._validate_record_session(record)
        super().compare_and_set(attempt_id, expected_state, record)
        if self._grant_store is not None and record.grant is not None:
            self._grant_store.remember(
                session_id=self._session_id,
                registration_id=record.request.registration_id,
                grant=record.grant,
            )

    def find_reusable_grant(self, request, access):
        grant = super().find_reusable_grant(request, access)
        if grant is not None or self._grant_store is None:
            return grant
        return self._grant_store.find_reusable(
            session_id=self._session_id,
            request=request,
            access=access,
        )

    def checkpoint_state(
        self,
        *,
        intent: Mapping[str, object] | None = None,
    ) -> dict[str, object] | None:
        with self._lock:
            attempts = [
                _record_to_dict(item)
                for item in self._attempts.values()
                if item.state not in _TERMINAL_STATES
            ]
        if not attempts and not intent:
            return None
        return {
            "schema_version": TOOL_CHECKPOINT_SCHEMA_VERSION,
            "session_id": self._session_id,
            "attempts": attempts,
            "intent": _plain_json(intent) if intent is not None else None,
        }

    def restore_checkpoint_state(self, payload: Mapping[str, object]) -> None:
        _require_exact_keys(payload, set(_CHECKPOINT_KEYS), "tool checkpoint")
        if payload.get("schema_version") != TOOL_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("Unsupported tool checkpoint schema version")
        if payload.get("session_id") != self._session_id:
            raise ValueError("Tool checkpoint session_id does not match runtime session")
        attempts = payload.get("attempts")
        if not isinstance(attempts, list):
            raise ValueError("Tool checkpoint attempts must be a list")
        intent = payload.get("intent")
        if intent is not None and not isinstance(intent, Mapping):
            raise ValueError("Tool checkpoint intent must be an object or null")
        with self._lock:
            if self._attempts:
                raise ValueError("Tool state can only be restored into an empty store")
            for raw in attempts:
                record = _record_from_dict(_mapping(raw, "tool attempt"))
                self._validate_record_session(record)
                if record.state in _TERMINAL_STATES:
                    raise ValueError("Tool checkpoint cannot contain terminal attempts")
                super().create(record)

    def _validate_record_session(self, record: ToolAttemptRecord) -> None:
        if record.request.session_id != self._session_id:
            raise ValueError("Tool attempt session_id does not match state store")


def _record_to_dict(record: ToolAttemptRecord) -> dict[str, object]:
    return {
        "attempt_id": record.attempt_id,
        "request": _request_to_dict(record.request),
        "state": record.state,
        "challenge": _challenge_to_dict(record.challenge) if record.challenge else None,
        "interaction": _interaction_to_dict(record.interaction) if record.interaction else None,
        "interaction_consumed": record.interaction_consumed,
        "grant": _grant_to_dict(record.grant) if record.grant else None,
        "grant_consumed": record.grant_consumed,
        "result": _result_to_dict(record.result) if record.result else None,
        "cleanup_errors": list(record.cleanup_errors),
    }


def _record_from_dict(raw: Mapping[str, object]) -> ToolAttemptRecord:
    expected = {
        "attempt_id", "request", "state", "challenge", "interaction",
        "interaction_consumed", "grant", "grant_consumed", "result", "cleanup_errors",
    }
    _require_exact_keys(raw, expected, "tool attempt")
    request = _mapping(raw.get("request"), "request")
    challenge_raw = raw.get("challenge")
    interaction_raw = raw.get("interaction")
    grant_raw = raw.get("grant")
    result_raw = raw.get("result")
    cleanup_errors = raw.get("cleanup_errors")
    if not isinstance(cleanup_errors, list) or any(not isinstance(item, str) for item in cleanup_errors):
        raise ValueError("cleanup_errors must be a list of strings")
    return ToolAttemptRecord(
        attempt_id=_required_text(raw.get("attempt_id"), "attempt_id"),
        request=_request_from_dict(request),
        state=cast(object, _required_text(raw.get("state"), "state")),
        challenge=(
            _challenge_from_dict(_mapping(challenge_raw, "challenge"))
            if challenge_raw is not None else None
        ),
        interaction=(
            _interaction_from_dict(_mapping(interaction_raw, "interaction"))
            if interaction_raw is not None else None
        ),
        interaction_consumed=_required_bool(raw.get("interaction_consumed"), "interaction_consumed"),
        grant=(
            _grant_from_dict(_mapping(grant_raw, "grant"))
            if grant_raw is not None else None
        ),
        grant_consumed=_required_bool(raw.get("grant_consumed"), "grant_consumed"),
        result=(
            _result_from_dict(_mapping(result_raw, "result"))
            if result_raw is not None else None
        ),
        cleanup_errors=tuple(cleanup_errors),
    )


def _request_to_dict(value: ToolExecutionRequest) -> dict[str, object]:
    return {
        "run_id": value.run_id,
        "session_id": value.session_id,
        "tool_call_id": value.tool_call_id,
        "tool_name": value.tool_name,
        "arguments": _plain_json(value.arguments),
        "mode": value.mode,
        "registration_id": value.registration_id,
        "deadline_at_ms": value.deadline_at_ms,
        "raw_arguments": value.raw_arguments,
        "argument_parse_error": value.argument_parse_error,
    }


def _request_from_dict(raw: Mapping[str, object]) -> ToolExecutionRequest:
    _require_exact_keys(raw, {
        "run_id", "session_id", "tool_call_id", "tool_name", "arguments", "mode",
        "registration_id", "deadline_at_ms", "raw_arguments",
        "argument_parse_error",
    }, "tool request")
    return ToolExecutionRequest(
        run_id=_required_text(raw.get("run_id"), "run_id"),
        session_id=_required_text(raw.get("session_id"), "session_id"),
        tool_call_id=_required_text(raw.get("tool_call_id"), "tool_call_id"),
        tool_name=_required_text(raw.get("tool_name"), "tool_name"),
        arguments=_mapping(raw.get("arguments"), "arguments"),
        mode=cast(object, _required_text(raw.get("mode"), "mode")),
        registration_id=_required_text(raw.get("registration_id"), "registration_id"),
        deadline_at_ms=_optional_int(raw.get("deadline_at_ms"), "deadline_at_ms"),
        raw_arguments=_optional_text(raw.get("raw_arguments")),
        argument_parse_error=_optional_text(raw.get("argument_parse_error")),
    )


def _resource_to_dict(value: ToolResource) -> dict[str, object]:
    return {"uri": value.uri, "metadata": _plain_json(value.metadata)}


def _resource_from_dict(raw: Mapping[str, object]) -> ToolResource:
    _require_exact_keys(raw, {"uri", "metadata"}, "tool resource")
    return ToolResource(
        uri=_required_text(raw.get("uri"), "resource uri"),
        metadata=_mapping(raw.get("metadata"), "resource metadata"),
    )


def _challenge_to_dict(value: ApprovalChallenge) -> dict[str, object]:
    return {
        "approval_id": value.approval_id,
        "request_fingerprint": value.request_fingerprint,
        "run_id": value.run_id,
        "session_id": value.session_id,
        "tool_call_id": value.tool_call_id,
        "tool_name": value.tool_name,
        "registration_id": value.registration_id,
        "actions": list(value.actions),
        "resources": [_resource_to_dict(item) for item in value.resources],
        "effects": sorted(value.effects),
        "risk": value.risk,
        "reason": value.reason,
        "safe_preview": _plain_json(value.safe_preview),
        "allowed_scopes": sorted(value.allowed_scopes),
        "expires_at_ms": value.expires_at_ms,
    }


def _challenge_from_dict(raw: Mapping[str, object]) -> ApprovalChallenge:
    expected = {
        "approval_id", "request_fingerprint", "run_id", "session_id", "tool_call_id",
        "tool_name", "registration_id", "actions", "resources", "effects", "risk",
        "reason", "safe_preview", "allowed_scopes", "expires_at_ms",
    }
    _require_exact_keys(raw, expected, "approval challenge")
    return ApprovalChallenge(
        approval_id=_required_text(raw.get("approval_id"), "approval_id"),
        request_fingerprint=_required_text(raw.get("request_fingerprint"), "request_fingerprint"),
        run_id=_required_text(raw.get("run_id"), "run_id"),
        session_id=_required_text(raw.get("session_id"), "session_id"),
        tool_call_id=_required_text(raw.get("tool_call_id"), "tool_call_id"),
        tool_name=_required_text(raw.get("tool_name"), "tool_name"),
        registration_id=_required_text(raw.get("registration_id"), "registration_id"),
        actions=_string_tuple(raw.get("actions"), "actions"),
        resources=_resources(raw.get("resources")),
        effects=frozenset(_string_tuple(raw.get("effects"), "effects")),
        risk=cast(object, _required_text(raw.get("risk"), "risk")),
        reason=_required_text(raw.get("reason"), "reason"),
        safe_preview=_mapping(raw.get("safe_preview"), "safe_preview"),
        allowed_scopes=frozenset(_string_tuple(raw.get("allowed_scopes"), "allowed_scopes")),
        expires_at_ms=_optional_int(raw.get("expires_at_ms"), "expires_at_ms"),
    )


def _grant_to_dict(value: ApprovalGrant) -> dict[str, object]:
    return {
        "grant_id": value.grant_id,
        "approval_id": value.approval_id,
        "request_fingerprint": value.request_fingerprint,
        "scope": value.scope,
        "actions": list(value.actions),
        "resources": [_resource_to_dict(item) for item in value.resources],
        "effects": sorted(value.effects),
        "risk": value.risk,
        "issued_at_ms": value.issued_at_ms,
        "expires_at_ms": value.expires_at_ms,
    }


def _grant_from_dict(raw: Mapping[str, object]) -> ApprovalGrant:
    expected = {
        "grant_id", "approval_id", "request_fingerprint", "scope", "actions",
        "resources", "effects", "risk", "issued_at_ms", "expires_at_ms",
    }
    _require_exact_keys(raw, expected, "approval grant")
    return ApprovalGrant(
        grant_id=_required_text(raw.get("grant_id"), "grant_id"),
        approval_id=_required_text(raw.get("approval_id"), "approval_id"),
        request_fingerprint=_required_text(raw.get("request_fingerprint"), "request_fingerprint"),
        scope=cast(object, _required_text(raw.get("scope"), "scope")),
        actions=_string_tuple(raw.get("actions"), "actions"),
        resources=_resources(raw.get("resources")),
        effects=frozenset(_string_tuple(raw.get("effects"), "effects")),
        risk=cast(object, _required_text(raw.get("risk"), "risk")),
        issued_at_ms=_required_int(raw.get("issued_at_ms"), "issued_at_ms"),
        expires_at_ms=_optional_int(raw.get("expires_at_ms"), "expires_at_ms"),
    )


def _interaction_to_dict(value: InteractionRequest) -> dict[str, object]:
    return value.to_dict()


def _interaction_from_dict(raw: Mapping[str, object]) -> InteractionRequest:
    expected = {
        "interaction_id", "request_fingerprint", "session_id", "tool_call_id",
        "tool_name", "registration_id", "prompt", "options", "allow_free_text",
        "created_at_ms",
    }
    _require_exact_keys(raw, expected, "interaction request")
    return InteractionRequest(
        interaction_id=_required_text(raw.get("interaction_id"), "interaction_id"),
        request_fingerprint=_required_text(raw.get("request_fingerprint"), "request_fingerprint"),
        session_id=_required_text(raw.get("session_id"), "session_id"),
        tool_call_id=_required_text(raw.get("tool_call_id"), "tool_call_id"),
        tool_name=_required_text(raw.get("tool_name"), "tool_name"),
        registration_id=_required_text(raw.get("registration_id"), "registration_id"),
        prompt=_required_text(raw.get("prompt"), "prompt"),
        options=_string_tuple(raw.get("options"), "options"),
        allow_free_text=_required_bool(raw.get("allow_free_text"), "allow_free_text"),
        created_at_ms=_required_int(raw.get("created_at_ms"), "created_at_ms"),
    )


def _result_to_dict(value: ToolResult) -> dict[str, object]:
    return {
        "tool_call_id": value.tool_call_id,
        "tool_name": value.tool_name,
        "status": value.status,
        "content": [_content_to_dict(item) for item in value.content],
        "data": _plain_json(value.data),
        "error": _error_to_dict(value.error) if value.error else None,
        "effects": [_effect_to_dict(item) for item in value.effects],
        "artifacts": [_artifact_to_dict(item) for item in value.artifacts],
        "approval": _challenge_to_dict(value.approval) if value.approval else None,
        "interaction": _plain_json(value.interaction) if value.interaction else None,
        "timing": {
            "queued_at_ms": value.timing.queued_at_ms,
            "started_at_ms": value.timing.started_at_ms,
            "finished_at_ms": value.timing.finished_at_ms,
            "duration_ms": value.timing.duration_ms,
        },
        "registration_id": value.registration_id,
        "output_validation": value.output_validation,
        "content_trust": value.content_trust,
    }


def _result_from_dict(raw: Mapping[str, object]) -> ToolResult:
    expected = {
        "tool_call_id", "tool_name", "status", "content", "data", "error", "effects",
        "artifacts", "approval", "interaction", "timing", "registration_id",
        "output_validation", "content_trust",
    }
    _require_exact_keys(raw, expected, "tool result")
    content = raw.get("content")
    effects = raw.get("effects")
    artifacts = raw.get("artifacts")
    if not isinstance(content, list) or not isinstance(effects, list) or not isinstance(artifacts, list):
        raise ValueError("Tool result content, effects and artifacts must be lists")
    error_raw = raw.get("error")
    approval_raw = raw.get("approval")
    interaction_raw = raw.get("interaction")
    timing = _mapping(raw.get("timing"), "timing")
    _require_exact_keys(
        timing,
        {"queued_at_ms", "started_at_ms", "finished_at_ms", "duration_ms"},
        "tool timing",
    )
    return ToolResult(
        tool_call_id=_required_text(raw.get("tool_call_id"), "tool_call_id"),
        tool_name=_required_text(raw.get("tool_name"), "tool_name"),
        status=cast(object, _required_text(raw.get("status"), "status")),
        content=tuple(_content_from_dict(_mapping(item, "content")) for item in content),
        data=_mapping(raw.get("data"), "data"),
        error=_error_from_dict(_mapping(error_raw, "error")) if error_raw is not None else None,
        effects=tuple(_effect_from_dict(_mapping(item, "effect")) for item in effects),
        artifacts=tuple(_artifact_from_dict(_mapping(item, "artifact")) for item in artifacts),
        approval=(
            _challenge_from_dict(_mapping(approval_raw, "approval"))
            if approval_raw is not None else None
        ),
        interaction=(
            _mapping(interaction_raw, "interaction")
            if interaction_raw is not None else None
        ),
        timing=ToolTiming(
            queued_at_ms=_optional_int(timing.get("queued_at_ms"), "queued_at_ms"),
            started_at_ms=_optional_int(timing.get("started_at_ms"), "started_at_ms"),
            finished_at_ms=_optional_int(timing.get("finished_at_ms"), "finished_at_ms"),
            duration_ms=_optional_int(timing.get("duration_ms"), "duration_ms"),
        ),
        registration_id=_required_text(raw.get("registration_id"), "registration_id"),
        output_validation=cast(object, _required_text(raw.get("output_validation"), "output_validation")),
        content_trust=cast(object, _required_text(raw.get("content_trust"), "content_trust")),
    )


def _content_to_dict(value) -> dict[str, object]:
    if isinstance(value, TextContent):
        return {"type": "text", "text": value.text}
    if isinstance(value, ImageContent):
        return {"type": "image", "data": value.data, "mime_type": value.mime_type, "name": value.name}
    return {"type": "artifact", "artifact": _artifact_to_dict(value.artifact)}


def _content_from_dict(raw: Mapping[str, object]):
    kind = raw.get("type")
    if kind == "text":
        _require_exact_keys(raw, {"type", "text"}, "text content")
        return TextContent(_required_string(raw.get("text"), "text"))
    if kind == "image":
        _require_exact_keys(raw, {"type", "data", "mime_type", "name"}, "image content")
        return ImageContent(
            data=_required_text(raw.get("data"), "image data"),
            mime_type=_required_text(raw.get("mime_type"), "mime_type"),
            name=_optional_text(raw.get("name")),
        )
    if kind == "artifact":
        _require_exact_keys(raw, {"type", "artifact"}, "artifact content")
        return ArtifactContent(_artifact_from_dict(_mapping(raw.get("artifact"), "artifact")))
    raise ValueError("Unknown tool content type")


def _artifact_to_dict(value: ArtifactRef) -> dict[str, object]:
    return {
        "artifact_id": value.artifact_id,
        "media_type": value.media_type,
        "name": value.name,
        "size_bytes": value.size_bytes,
    }


def _artifact_from_dict(raw: Mapping[str, object]) -> ArtifactRef:
    _require_exact_keys(raw, {"artifact_id", "media_type", "name", "size_bytes"}, "artifact")
    return ArtifactRef(
        artifact_id=_required_text(raw.get("artifact_id"), "artifact_id"),
        media_type=_required_text(raw.get("media_type"), "media_type"),
        name=_optional_text(raw.get("name")),
        size_bytes=_optional_int(raw.get("size_bytes"), "size_bytes"),
    )


def _effect_to_dict(value: ToolEffect) -> dict[str, object]:
    return {
        "kind": value.kind,
        "resource": _resource_to_dict(value.resource),
        "operation": value.operation,
        "status": value.status,
        "certainty": value.certainty,
    }


def _effect_from_dict(raw: Mapping[str, object]) -> ToolEffect:
    _require_exact_keys(raw, {"kind", "resource", "operation", "status", "certainty"}, "effect")
    return ToolEffect(
        kind=cast(object, _required_text(raw.get("kind"), "effect kind")),
        resource=_resource_from_dict(_mapping(raw.get("resource"), "resource")),
        operation=_required_text(raw.get("operation"), "operation"),
        status=cast(object, _required_text(raw.get("status"), "effect status")),
        certainty=cast(object, _required_text(raw.get("certainty"), "effect certainty")),
    )


def _error_to_dict(value: ToolError) -> dict[str, object]:
    return {
        "code": value.code,
        "kind": value.kind,
        "message": value.message,
        "retryable": value.retryable,
        "recovery_hint": value.recovery_hint,
        "details": _plain_json(value.details),
    }


def _error_from_dict(raw: Mapping[str, object]) -> ToolError:
    _require_exact_keys(raw, {"code", "kind", "message", "retryable", "recovery_hint", "details"}, "tool error")
    return ToolError(
        code=_required_text(raw.get("code"), "error code"),
        kind=cast(object, _required_text(raw.get("kind"), "error kind")),
        message=_required_text(raw.get("message"), "error message"),
        retryable=_required_bool(raw.get("retryable"), "retryable"),
        recovery_hint=_optional_text(raw.get("recovery_hint")) or "",
        details=_mapping(raw.get("details"), "error details"),
    )


def _resources(value: object) -> tuple[ToolResource, ...]:
    if not isinstance(value, list):
        raise ValueError("resources must be a list")
    return tuple(_resource_from_dict(_mapping(item, "resource")) for item in value)


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return tuple(value)


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return cast(Mapping[str, object], value)


def _require_exact_keys(raw: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(raw) != expected:
        raise ValueError(f"{name} has unknown or missing fields")


def _required_text(value: object, field_name: str) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    return _required_text(value, "optional text")


def _required_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be bool")
    return value


def _required_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be int")
    return value


def _optional_int(value: object, field_name: str) -> int | None:
    return None if value is None else _required_int(value, field_name)


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


def _grant_file_lock(path: Path) -> RLock:
    with _GRANT_FILE_LOCKS_GUARD:
        return _GRANT_FILE_LOCKS.setdefault(path, RLock())


def _read_json_object(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return cast(dict[str, object], value)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    data = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


__all__ = [
    "CheckpointToolStateStore",
    "FileToolGrantStore",
    "TOOL_CHECKPOINT_SCHEMA_VERSION",
    "TOOL_GRANT_SCHEMA_VERSION",
]
