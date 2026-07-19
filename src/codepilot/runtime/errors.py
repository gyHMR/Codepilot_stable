"""将运行时异常规范化为结构化错误信息和事件载荷。"""

from __future__ import annotations

"""Single Runtime error normalization used by persistence and interface frames."""

from typing import Any

from codepilot.core.contracts import CoreReason
from codepilot.core.errors import CoreContractError, CoreInvariantError
from codepilot.protocols import ErrorInfo


def runtime_error_info(error: Any) -> ErrorInfo | None:
    if error is None:
        return None
    if isinstance(error, ErrorInfo):
        return error
    if isinstance(error, CoreReason):
        details = dict(error.details)
        source = _core_reason_error_source(error.source)
        if source != error.source:
            details.setdefault("reason_source", error.source)
        if error.evidence_refs:
            details.setdefault("evidence_refs", list(error.evidence_refs))
        return ErrorInfo(
            code=error.code,
            message=error.message or error.code,
            retryable=error.recoverable,
            source=source,
            details=details,
        )
    if isinstance(error, dict):
        code = _text(error.get("code")) or "runtime.dispatch_failed"
        message = _text(error.get("message")) or code
        details = error.get("details")
        return ErrorInfo(
            code=code,
            message=message,
            retryable=bool(error.get("retryable", False)),
            source="runtime",
            details=dict(details) if isinstance(details, dict) else {},
        )

    if isinstance(error, (CoreContractError, CoreInvariantError)):
        code = (
            "runtime.core_contract_error"
            if isinstance(error, CoreContractError)
            else "runtime.core_invariant_error"
        )
        return ErrorInfo(
            code=code,
            message=str(error),
            retryable=False,
            source="runtime",
            details={"error_type": type(error).__name__},
        )

    details: dict[str, Any] = {"error_type": type(error).__name__}
    nested = getattr(error, "error", None)
    message = str(error)
    if nested is not None:
        message = str(getattr(nested, "message", message))
        cause_code = _text(getattr(nested, "code", None))
        if cause_code is not None:
            details["cause_code"] = cause_code
    if hasattr(error, "run_id"):
        details["run_id"] = getattr(error, "run_id")
    if hasattr(error, "status"):
        details["status"] = getattr(error, "status")
    return ErrorInfo(
        code="runtime.dispatch_failed",
        message=message,
        retryable=False,
        source="runtime",
        details=details,
    )


def runtime_error_payload(error: Any) -> dict[str, Any]:
    info = runtime_error_info(error)
    if info is None:
        return {
            "code": "runtime.dispatch_failed",
            "message": "Runtime dispatch failed",
            "details": {},
        }
    return {
        "code": info.code,
        "message": info.message,
        "source": info.source,
        "retryable": info.retryable,
        "details": dict(info.details),
    }


def _text(value: object) -> str | None:
    text = value.strip() if isinstance(value, str) else ""
    return text or None


def _core_reason_error_source(source: str) -> str:
    aliases = {
        "model": "llm",
        "tools": "tool",
        "verification": "core",
    }
    allowed = {"llm", "tool", "core", "runtime", "session", "interface"}
    return aliases.get(source, source if source in allowed else "core")


__all__ = ["runtime_error_info", "runtime_error_payload"]
