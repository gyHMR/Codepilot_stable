from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


_RUNTIME_ENVELOPE_FIELDS = frozenset(
    {
        "event_id",
        "run_id",
        "session_id",
        "sequence",
        "timestamp",
        "timestamp_ms",
        "type",
    }
)


@dataclass(frozen=True)
class CoreDomainEvent:
    """A durable semantic event without Runtime envelope fields."""

    kind: str
    payload: Mapping[str, object] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        kind = str(self.kind).strip()
        if not kind:
            raise ValueError("Core domain event kind is required")
        if not isinstance(self.payload, Mapping):
            raise TypeError("Core domain event payload must be a mapping")
        payload = dict(self.payload)
        envelope_fields = _RUNTIME_ENVELOPE_FIELDS.intersection(payload)
        if envelope_fields:
            names = ", ".join(sorted(envelope_fields))
            raise ValueError(
                f"Core domain event cannot contain Runtime envelope fields: {names}"
            )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "payload", MappingProxyType(payload))
        object.__setattr__(
            self,
            "evidence_refs",
            tuple(dict.fromkeys(_required_text(value) for value in self.evidence_refs)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "payload": dict(self.payload),
            "evidence_refs": list(self.evidence_refs),
        }


def _required_text(value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("evidence reference cannot be empty")
    return text


__all__ = ["CoreDomainEvent"]
