from __future__ import annotations

"""JSONL RPC mode for non-human clients."""

from dataclasses import asdict, dataclass, is_dataclass
import json
import sys
from typing import Any, Callable

from codepilot.runtime.actions import (
    ApprovalDecided,
    ApprovalRequiredFrame,
    CommandFinishedFrame,
    CommandSubmitted,
    FailedFrame,
    ProgressFrame,
    PromptSubmitted,
    RunFinishedFrame,
)
from codepilot.runtime.gateway import RuntimeGateway

from .render import OutputFn


RpcEmit = Callable[[dict[str, Any]], None]
RPC_PROTOCOL_VERSION = "1.2"


@dataclass(frozen=True)
class RpcError:
    """Normalized JSONL RPC error payload."""

    code: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", require_rpc_text(self.code, field_name="code"))
        object.__setattr__(self, "message", require_rpc_text(self.message, field_name="message"))


async def run_rpc(
    runtime: RuntimeGateway,
    session_id: str,
    *,
    output: OutputFn = print,
) -> None:
    """Run the strict JSONL protocol over stdin/stdout."""

    def emit(obj: dict[str, Any]) -> None:
        output(json.dumps(obj, ensure_ascii=False, default=rpc_json_default))

    emit_rpc_ready(emit, session_id=session_id)
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:
            emit_rpc_error(
                emit,
                req_id=None,
                command=None,
                code="invalid_json",
                message=f"Invalid JSON: {exc}",
            )
            continue
        if await handle_rpc_request(runtime, session_id, req, emit):
            return


async def handle_rpc_request(
    runtime: RuntimeGateway,
    session_id: str,
    req: Any,
    emit: RpcEmit,
) -> bool:
    """Handle one JSONL object. Return True to stop the RPC loop."""

    if not isinstance(req, dict):
        emit_rpc_error(
            emit,
            req_id=None,
            command=None,
            code="invalid_request",
            message="Request must be object",
        )
        return False

    command = req.get("type")
    req_id = req.get("id")
    try:
        if command == "prompt":
            await handle_rpc_prompt(runtime, session_id, req, emit)
        elif command in {"approve", "deny"}:
            await handle_rpc_approval(runtime, session_id, req, emit, decision=command)
        elif command == "state":
            emit_rpc_ok(
                emit,
                req_id=req_id,
                command="state",
                data=session_state(runtime, session_id),
            )
        elif command == "set_task_mode":
            await handle_rpc_command(
                runtime,
                session_id,
                req,
                emit,
                command_name="set_task_mode",
                command_text=f"/mode {require_request_text(req, 'task_mode')}",
                data_builder=lambda result: {
                    "session_id": session_id,
                    "task_mode": result.data.get("task_mode"),
                },
            )
        elif command == "list_entries":
            state = session_state(runtime, session_id)
            emit_rpc_ok(
                emit,
                req_id=req_id,
                command="list_entries",
                data={
                    "session_id": session_id,
                    "entry_ids": state["entry_ids"],
                    "entries": state.get("entries", []),
                    "leaf_id": state["leaf_id"],
                },
            )
        elif command == "show_tree":
            state = session_state(runtime, session_id)
            emit_rpc_ok(
                emit,
                req_id=req_id,
                command="show_tree",
                data={
                    "session_id": session_id,
                    "tree": state.get("tree", []),
                    "leaf_id": state["leaf_id"],
                },
            )
        elif command == "entry_path":
            entry_id = require_request_text(req, "entry_id")
            await handle_rpc_command(
                runtime,
                session_id,
                req,
                emit,
                command_name="entry_path",
                command_text=f"/path {entry_id}",
                data_builder=lambda result: {
                    "session_id": session_id,
                    "entry_id": entry_id,
                    "path": result.data.get("path", []),
                },
            )
        elif command == "fork_entry":
            entry_id = require_request_text(req, "entry_id")
            await handle_rpc_command(
                runtime,
                session_id,
                req,
                emit,
                command_name="fork_entry",
                command_text=f"/fork {entry_id}",
                data_builder=lambda result: dict(result.data),
            )
        elif command == "switch_entry":
            entry_id = require_request_text(req, "entry_id")
            await handle_rpc_command(
                runtime,
                session_id,
                req,
                emit,
                command_name="switch_entry",
                command_text=f"/switch {entry_id}",
                data_builder=lambda result: dict(result.data),
            )
        elif command == "get_commands":
            emit_rpc_ok(
                emit,
                req_id=req_id,
                command="get_commands",
                data={
                    "session_id": session_id,
                    "commands": [
                        command_view.to_dict()
                        for command_view in runtime.describe(session_id).commands
                    ],
                },
            )
        elif command == "shutdown":
            emit_rpc_ok(emit, req_id=req_id, command="shutdown")
            return True
        else:
            emit_rpc_error(
                emit,
                req_id=req_id,
                command=command,
                code="unknown_command",
                message="Unknown command",
            )
    except Exception as exc:
        error = rpc_error_from_exception(exc)
        emit_rpc_error(
            emit,
            req_id=req_id,
            command=command,
            code=error.code,
            message=error.message,
        )
    return False


async def handle_rpc_prompt(
    runtime: RuntimeGateway,
    session_id: str,
    req: dict[str, Any],
    emit: RpcEmit,
) -> None:
    task_mode = req.get("task_mode")
    if task_mode is not None and not isinstance(task_mode, str):
        raise ValueError("task_mode must be a string")

    result_data = None
    async for frame in runtime.dispatch(
        session_id,
        PromptSubmitted(text=str(req.get("text", "")), mode_hint=task_mode),
    ):
        result_data = handle_rpc_run_frame(frame, emit, current_result=result_data)
    emit_rpc_ok(emit, req_id=req.get("id"), command="prompt", data=result_data)


async def handle_rpc_approval(
    runtime: RuntimeGateway,
    session_id: str,
    req: dict[str, Any],
    emit: RpcEmit,
    *,
    decision: Any,
) -> None:
    approval_id = str(req.get("approval_id", "")).strip()
    if not approval_id:
        raise ValueError(f"{decision} requires approval_id")
    result_data = None
    async for frame in runtime.dispatch(
        session_id,
        ApprovalDecided(
            approval_id=approval_id,
            decision=decision,  # type: ignore[arg-type]
            reason=str(req.get("reason", "")).strip(),
        ),
    ):
        result_data = handle_rpc_run_frame(frame, emit, current_result=result_data)
    emit_rpc_ok(emit, req_id=req.get("id"), command=decision, data=result_data)


async def handle_rpc_command(
    runtime: RuntimeGateway,
    session_id: str,
    req: dict[str, Any],
    emit: RpcEmit,
    *,
    command_name: str,
    command_text: str,
    data_builder: Callable[[Any], dict[str, Any]],
) -> None:
    result = await dispatch_rpc_command(runtime, session_id, command_text)
    emit_rpc_ok(
        emit,
        req_id=req.get("id"),
        command=command_name,
        data=data_builder(result),
    )


async def dispatch_rpc_command(runtime: RuntimeGateway, session_id: str, text: str) -> Any:
    async for frame in runtime.dispatch(session_id, CommandSubmitted(text=text)):
        if isinstance(frame, CommandFinishedFrame):
            return frame.record
        if isinstance(frame, FailedFrame):
            raise runtime_error_from_frame(frame.error)
    raise RuntimeFrameError("Runtime command finished without a result")


def handle_rpc_run_frame(
    frame: Any,
    emit: RpcEmit,
    *,
    current_result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if isinstance(frame, ProgressFrame):
        emit({"type": "event", "event": frame.event})
        return current_result
    if isinstance(frame, ApprovalRequiredFrame):
        approval = approval_rpc_data(frame)
        emit({"type": "approval_required", "approval": approval})
        return {
            "status": "waiting_approval",
            "approval_id": approval["approval_id"],
        }
    if isinstance(frame, RunFinishedFrame):
        return run_record_rpc_data(frame.record)
    if isinstance(frame, FailedFrame):
        raise runtime_error_from_frame(frame.error)
    return current_result


def emit_rpc_ready(emit: RpcEmit, *, session_id: str) -> None:
    normalized_session_id = require_rpc_text(session_id, field_name="session_id")
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
    payload: dict[str, Any] = {
        "type": "response",
        "id": req_id,
        "command": require_rpc_text(command, field_name="command"),
        "status": "ok",
    }
    if data is not None:
        payload["data"] = data
    emit(payload)


def emit_rpc_error(
    emit: RpcEmit,
    *,
    req_id: Any,
    command: Any,
    code: str,
    message: str,
) -> None:
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


def rpc_error_from_exception(exc: Exception) -> RpcError:
    raw_code = getattr(exc, "code", None)
    code = raw_code.strip() if isinstance(raw_code, str) and raw_code.strip() else "execution_error"
    message = str(exc).strip() or type(exc).__name__
    return RpcError(code=code, message=message)


def rpc_json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, set):
        return list(value)
    return str(value)


def require_rpc_text(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"RPC {field_name} must be str")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"RPC {field_name} cannot be empty")
    return normalized


def require_request_text(req: dict[str, Any], field_name: str) -> str:
    value = req.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{req.get('type')} requires {field_name}")
    return value.strip()


def session_state(runtime: RuntimeGateway, session_id: str) -> dict[str, Any]:
    return dict(runtime.describe(session_id).state or {})


def run_record_rpc_data(record: Any) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for name in ("run_id", "session_id", "status", "stop_reason", "final_text"):
        value = getattr(record, name, None)
        if value is not None:
            data[name] = value
    return data


def approval_rpc_data(frame: ApprovalRequiredFrame) -> dict[str, Any]:
    approval = frame.approval
    risk = getattr(approval, "risk", None)
    return {
        "approval_id": str(getattr(approval, "approval_id", "")),
        "run_id": str(getattr(approval, "run_id", "")),
        "tool_call_id": str(getattr(approval, "tool_call_id", "")),
        "tool_name": str(getattr(approval, "tool_name", "")),
        "arguments": dict(getattr(approval, "arguments", {}) or {}),
        "reason": str(getattr(approval, "reason", "")),
        "risk_level": str(getattr(risk, "level", "unknown")),
    }


def frame_error_message(error: Any) -> str:
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or "Runtime failed")
    return str(error)


class RuntimeFrameError(RuntimeError):
    code = "runtime.dispatch_failed"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code:
            self.code = code


def runtime_error_from_frame(error: Any) -> RuntimeFrameError:
    if isinstance(error, dict):
        return RuntimeFrameError(
            frame_error_message(error),
            code=str(error.get("code") or "runtime.dispatch_failed"),
        )
    return RuntimeFrameError(str(error))


__all__ = [
    "RPC_PROTOCOL_VERSION",
    "RpcEmit",
    "RpcError",
    "emit_rpc_error",
    "emit_rpc_ok",
    "emit_rpc_ready",
    "rpc_error_from_exception",
    "rpc_json_default",
    "run_rpc",
]
