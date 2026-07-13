from __future__ import annotations


class CoreContractError(ValueError):
    """Raised when a caller violates a Core input or port contract."""


class CoreInvariantError(RuntimeError):
    """Raised when Core produces an internally impossible state."""


__all__ = ["CoreContractError", "CoreInvariantError"]
