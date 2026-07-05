from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .persistence.store import SessionStore


@dataclass(frozen=True)
class SessionOpenMetadata:
    """Read-only metadata needed when reopening an existing session."""

    provider: str | None = None
    model_id: str | None = None
    system_prompt: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "SessionOpenMetadata":
        return cls(
            provider=_optional_text(value.get("provider")),
            model_id=_optional_text(value.get("model_id")),
            system_prompt=_optional_text(value.get("system_prompt")),
        )


def load_session_open_metadata(
    workspace_dir: str | Path,
    session_id: str | None,
) -> SessionOpenMetadata | None:
    """Load the stable metadata snapshot required to reopen a session."""

    if not session_id:
        return None
    raw = SessionStore(workspace_dir=workspace_dir, session_id=session_id).read_meta()
    if raw is None:
        return None
    return SessionOpenMetadata.from_mapping(raw)


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = ["SessionOpenMetadata", "load_session_open_metadata"]
