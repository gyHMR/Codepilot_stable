from __future__ import annotations

"""JSONL RPC response helpers for the CLI interface.

The runner owns command dispatch and stdin/stdout loops.  This module owns the
wire-level response shape so error mapping and serialization stay testable
without constructing a full interactive CLI run.
"""

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable


RpcEmit = Callable[[dict[str, Any]], None]
RPC_PROTOCOL_VERSION = "1.2"


@dataclass(frozen=True)
class RpcError:
    """Normalized JSONL RPC error payload.

    The CLI RPC mode is a machine-facing protocol.  Empty error codes or
    messages make failures hard for clients to classify, so the payload owns
    that invariant instead of relying on individual call sites.
    """

    code: str
    message: str

    def __post_init__(self) -> None:
        code = _require_rpc_text(self.code, field_name="code")
        message = _require_rpc_text(self.message, field_name="message")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", message)


def rpc_json_default(value: Any) -> Any:
    """Serialize dataclasses and sets while preserving readable fallback text."""

    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, set):
        return list(value)
    return str(value)


def rpc_error_from_exception(exc: Exception) -> RpcError:
    """Map Python exceptions to the stable JSONL RPC error contract."""

    raw_code = getattr(exc, "code", None)
    code = raw_code.strip() if isinstance(raw_code, str) and raw_code.strip() else "execution_error"
    message = str(exc).strip() or type(exc).__name__
    return RpcError(code=code, message=message)


def emit_rpc_error(
    emit: RpcEmit,
    *,
    req_id: Any,
    command: Any,
    code: str,
    message: str,
) -> None:
    """Emit one JSONL RPC error response."""

    error = RpcError(code=code, message=message)
    emit(
        {
            "type": "response",
            "id": req_id,
            "command": command,
            "status": "error",
            "error": {"code": error.code, "message": error.message},
        }
    )


def emit_rpc_ready(
    emit: RpcEmit,
    *,
    session_id: str,
) -> None:
    """Emit the initial JSONL RPC handshake message."""

    normalized_session_id = _require_rpc_text(
        session_id,
        field_name="session_id",
    )
    emit(
        {
            "type": "rpc_ready",
            "session_id": normalized_session_id,
            "protocol_version": RPC_PROTOCOL_VERSION,
        }
    )


def emit_rpc_ok(
    emit: RpcEmit,
    *,
    req_id: Any,
    command: str,
    data: dict[str, Any] | None = None,
) -> None:
    """Emit one JSONL RPC success response."""

    normalized_command = _require_rpc_text(command, field_name="command")
    payload: dict[str, Any] = {
        "type": "response",
        "id": req_id,
        "command": normalized_command,
        "status": "ok",
    }
    if data is not None:
        payload["data"] = data
    emit(payload)


def _require_rpc_text(value: str, *, field_name: str) -> str:
    """Return stripped protocol text or raise when the field has no meaning."""

    if not isinstance(value, str):
        raise TypeError(f"RPC error {field_name} must be str")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"RPC error {field_name} cannot be empty")
    return normalized


__all__ = [
    "RPC_PROTOCOL_VERSION",
    "RpcEmit",
    "RpcError",
    "emit_rpc_error",
    "emit_rpc_ok",
    "emit_rpc_ready",
    "rpc_error_from_exception",
    "rpc_json_default",
]
