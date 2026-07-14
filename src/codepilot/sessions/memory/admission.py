from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .contracts import (
    MemoryProposal,
    MemoryProposalOrigin,
    MemoryRecord,
    MemorySource,
    MemoryStatus,
)


AdmissionDisposition = Literal["accept", "duplicate", "conflict", "reject"]
_CURRENT_STATUSES = frozenset({"candidate", "active", "disabled"})
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\b(?:sk|rk)-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bgh[opusr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(
        r"\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password)\s*[:=]\s*"
        r"['\"]?[A-Za-z0-9_./+=-]{12,}",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class AdmissionResult:
    disposition: AdmissionDisposition
    proposal: MemoryProposal
    reason: str
    source: MemorySource | None = None
    status: MemoryStatus | None = None
    existing_id: str | None = None


class AdmissionPolicy:
    """Deterministic admission rules; no model calls or persistence."""

    def evaluate(
        self,
        proposal: MemoryProposal,
        *,
        existing: tuple[MemoryRecord, ...],
        origin: MemoryProposalOrigin,
        verification_passed: bool,
    ) -> AdmissionResult:
        normalized = MemoryProposal(
            scope=proposal.scope,
            type=proposal.type,
            key=proposal.key,
            content=normalize_content(proposal.content),
        )
        if contains_sensitive_content(normalized.content):
            return AdmissionResult(
                "reject",
                normalized,
                "sensitive_content",
            )
        if len(normalized.content) > 2000:
            return AdmissionResult(
                "reject",
                normalized,
                "content_too_long",
            )
        if (
            origin == "agent_finalization"
            and normalized.type == "experience"
            and not verification_passed
        ):
            return AdmissionResult(
                "reject",
                normalized,
                "experience_requires_verification",
            )

        current = tuple(
            record
            for record in existing
            if record.scope == normalized.scope
            and record.key == normalized.key
            and record.status in _CURRENT_STATUSES
        )
        duplicate = next(
            (record for record in current if record.content == normalized.content),
            None,
        )
        if duplicate is not None:
            return AdmissionResult(
                "duplicate",
                normalized,
                "exact_duplicate",
                existing_id=duplicate.id,
            )

        candidate = next(
            (record for record in current if record.status == "candidate"),
            None,
        )
        if candidate is not None:
            return AdmissionResult(
                "conflict",
                normalized,
                "candidate_conflict",
                existing_id=candidate.id,
            )
        if origin == "user_explicit" and current:
            return AdmissionResult(
                "conflict",
                normalized,
                "current_version_conflict",
                existing_id=current[0].id,
            )

        source: MemorySource
        if origin == "user_explicit":
            source = "user_feedback" if normalized.type == "feedback" else "user_explicit"
        elif normalized.type == "experience" and verification_passed:
            source = "verified_run"
        else:
            source = "agent_extracted"
        status: MemoryStatus = "active" if origin == "user_explicit" else "candidate"
        return AdmissionResult(
            "accept",
            normalized,
            "accepted",
            source=source,
            status=status,
        )


def normalize_content(content: str) -> str:
    return re.sub(r"\s+", " ", str(content or "")).strip()


def contains_sensitive_content(content: str) -> bool:
    return any(pattern.search(content) is not None for pattern in _SECRET_PATTERNS)


__all__ = [
    "AdmissionPolicy",
    "AdmissionResult",
    "contains_sensitive_content",
    "normalize_content",
]
