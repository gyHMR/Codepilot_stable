from __future__ import annotations

"""Tool attempt state port used by ToolRuntime approval and recovery flows."""

from dataclasses import dataclass, replace
import hashlib
import json
from threading import RLock
import time
from types import MappingProxyType
from typing import Literal, Mapping, Protocol
from uuid import uuid4

from .contracts import ToolExecutionRequest
from .results import ToolResult
from .security import ApprovalChallenge, ApprovalGrant, ToolAccessRequest


ToolAttemptState = Literal[
    "received",
    "validating",
    "resolving_access",
    "awaiting_approval",
    "awaiting_input",
    "queued",
    "running",
    "succeeded",
    "failed",
    "denied",
    "timed_out",
    "cancelled",
    "interrupted",
]


@dataclass(frozen=True)
class InteractionRequest:
    interaction_id: str
    request_fingerprint: str
    session_id: str
    tool_call_id: str
    tool_name: str
    registration_id: str
    prompt: str
    options: tuple[str, ...] = ()
    allow_free_text: bool = True
    created_at_ms: int = 0

    def __post_init__(self) -> None:
        for name in (
            "interaction_id",
            "request_fingerprint",
            "session_id",
            "tool_call_id",
            "tool_name",
            "registration_id",
            "prompt",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        options = tuple(str(value).strip() for value in self.options)
        if any(not value for value in options):
            raise ValueError("interaction options cannot contain empty values")
        if len(options) != len(set(options)):
            raise ValueError("interaction options must be unique")
        if not isinstance(self.allow_free_text, bool):
            raise TypeError("allow_free_text must be bool")
        if isinstance(self.created_at_ms, bool) or not isinstance(self.created_at_ms, int):
            raise TypeError("created_at_ms must be int")
        object.__setattr__(self, "options", options)

    def to_dict(self) -> dict[str, object]:
        return {
            "interaction_id": self.interaction_id,
            "request_fingerprint": self.request_fingerprint,
            "session_id": self.session_id,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "registration_id": self.registration_id,
            "prompt": self.prompt,
            "options": list(self.options),
            "allow_free_text": self.allow_free_text,
            "created_at_ms": self.created_at_ms,
        }


@dataclass(frozen=True)
class InteractionResponse:
    interaction_id: str
    request_fingerprint: str
    session_id: str
    tool_call_id: str
    tool_name: str
    registration_id: str
    answers: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in (
            "interaction_id",
            "request_fingerprint",
            "session_id",
            "tool_call_id",
            "tool_name",
            "registration_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        if not isinstance(self.answers, Mapping) or not self.answers:
            raise ValueError("interaction answers must be a non-empty mapping")
        copied = json.loads(json.dumps(dict(self.answers), ensure_ascii=False))
        object.__setattr__(self, "answers", MappingProxyType(copied))


def build_interaction_request(
    request: ToolExecutionRequest,
    *,
    prompt: str,
    options: tuple[str, ...] = (),
    allow_free_text: bool = True,
) -> InteractionRequest:
    payload = {
        "session_id": request.session_id,
        "tool_call_id": request.tool_call_id,
        "tool_name": request.tool_name,
        "registration_id": request.registration_id,
        "prompt": str(prompt).strip(),
        "options": list(options),
        "allow_free_text": allow_free_text,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return InteractionRequest(
        interaction_id=f"interaction_{uuid4().hex[:20]}",
        request_fingerprint=fingerprint,
        session_id=request.session_id,
        tool_call_id=request.tool_call_id,
        tool_name=request.tool_name,
        registration_id=request.registration_id,
        prompt=payload["prompt"],
        options=options,
        allow_free_text=allow_free_text,
        created_at_ms=int(time.time() * 1000),
    )


class ToolStateConflictError(RuntimeError):
    """A compare-and-set transition observed a different attempt state."""


@dataclass(frozen=True)
class ToolAttemptRecord:
    attempt_id: str
    request: ToolExecutionRequest
    state: ToolAttemptState = "received"
    challenge: ApprovalChallenge | None = None
    interaction: InteractionRequest | None = None
    interaction_consumed: bool = False
    grant: ApprovalGrant | None = None
    grant_consumed: bool = False
    result: ToolResult | None = None
    cleanup_errors: tuple[str, ...] = ()


class ToolStateStore(Protocol):
    def create(self, record: ToolAttemptRecord) -> None:
        ...

    def get(self, attempt_id: str) -> ToolAttemptRecord | None:
        ...

    def find_by_approval_id(self, approval_id: str) -> ToolAttemptRecord | None:
        ...

    def find_by_interaction_id(self, interaction_id: str) -> ToolAttemptRecord | None:
        ...

    def pending_challenges(self) -> tuple[ApprovalChallenge, ...]:
        ...

    def find_reusable_grant(
        self,
        request: ToolExecutionRequest,
        access: ToolAccessRequest,
    ) -> ApprovalGrant | None:
        ...

    def compare_and_set(
        self,
        attempt_id: str,
        expected_state: ToolAttemptState,
        record: ToolAttemptRecord,
    ) -> None:
        ...

class InMemoryToolStateStore:
    """Small default store; sessions may inject a persistent implementation."""

    def __init__(self) -> None:
        self._attempts: dict[str, ToolAttemptRecord] = {}
        self._approval_index: dict[str, str] = {}
        self._interaction_index: dict[str, str] = {}
        self._lock = RLock()

    def create(self, record: ToolAttemptRecord) -> None:
        with self._lock:
            if record.attempt_id in self._attempts:
                raise ToolStateConflictError(f"Tool attempt already exists: {record.attempt_id}")
            self._attempts[record.attempt_id] = record
            self._index(record)

    def get(self, attempt_id: str) -> ToolAttemptRecord | None:
        with self._lock:
            return self._attempts.get(attempt_id)

    def find_by_approval_id(self, approval_id: str) -> ToolAttemptRecord | None:
        with self._lock:
            attempt_id = self._approval_index.get(approval_id)
            return self._attempts.get(attempt_id) if attempt_id is not None else None

    def find_by_interaction_id(self, interaction_id: str) -> ToolAttemptRecord | None:
        with self._lock:
            attempt_id = self._interaction_index.get(interaction_id)
            return self._attempts.get(attempt_id) if attempt_id is not None else None

    def pending_challenges(self) -> tuple[ApprovalChallenge, ...]:
        with self._lock:
            return tuple(
                record.challenge
                for record in self._attempts.values()
                if record.state == "awaiting_approval" and record.challenge is not None
            )

    def find_reusable_grant(
        self,
        request: ToolExecutionRequest,
        access: ToolAccessRequest,
    ) -> ApprovalGrant | None:
        wanted_resources = tuple(resource.uri for resource in access.resources)
        with self._lock:
            for record in reversed(tuple(self._attempts.values())):
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
        with self._lock:
            current = self._attempts.get(attempt_id)
            if current is None or current.state != expected_state:
                actual = current.state if current is not None else "missing"
                raise ToolStateConflictError(
                    f"Tool attempt state conflict: expected {expected_state}, got {actual}"
                )
            if record.attempt_id != attempt_id:
                raise ValueError("replacement attempt_id must match")
            self._attempts[attempt_id] = record
            self._index(record)

    def _index(self, record: ToolAttemptRecord) -> None:
        if record.challenge is not None:
            self._approval_index[record.challenge.approval_id] = record.attempt_id
        if record.interaction is not None:
            self._interaction_index[record.interaction.interaction_id] = record.attempt_id


def transition(record: ToolAttemptRecord, state: ToolAttemptState, **changes) -> ToolAttemptRecord:
    return replace(record, state=state, **changes)


def attempt_id_for(request: ToolExecutionRequest) -> str:
    return f"{request.session_id}:{request.run_id}:{request.tool_call_id}"


__all__ = [
    "InMemoryToolStateStore",
    "InteractionRequest",
    "InteractionResponse",
    "ToolAttemptRecord",
    "ToolAttemptState",
    "ToolStateConflictError",
    "ToolStateStore",
    "attempt_id_for",
    "build_interaction_request",
    "transition",
]
