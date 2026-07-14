from __future__ import annotations


class CoreContractError(ValueError):
    """Raised when a caller violates a Core input or port contract."""


class CoreInvariantError(RuntimeError):
    """Raised when Core produces an internally impossible state."""


class CoreBoundaryCommitError(RuntimeError):
    """Raised when an authoritative Core boundary cannot be persisted."""

    def __init__(self, boundary_kind: str, cause: Exception) -> None:
        self.boundary_kind = str(boundary_kind)
        self.cause = cause
        super().__init__(f"Core boundary commit failed: {self.boundary_kind}: {cause}")


__all__ = [
    "CoreBoundaryCommitError",
    "CoreContractError",
    "CoreInvariantError",
]
