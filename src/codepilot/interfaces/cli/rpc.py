from __future__ import annotations

"""面向非人类客户端的 JSONL RPC 模式。

``codepilot rpc`` 会在 stdin/stdout 上使用一行一个 JSON 对象的协议：

- 输入：外部程序发送 ``{"type": "prompt", ...}``、``approve``、``deny``、
  ``state`` 等请求。
- 输出：CLI 返回 ``rpc_ready``、``response``、``event``、``approval_required`` 等对象。

这个模式服务于编辑器插件、脚本、测试进程等自动化客户端。它不渲染 Rich UI，
而是把 runtime frame 转成稳定 JSON payload。
"""

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
"""RPC 输出函数类型。参数是已经组装好的 JSON dict。"""

RPC_PROTOCOL_VERSION = "2.0"
"""当前 JSONL RPC 协议版本。客户端可据此判断字段兼容性。"""


@dataclass(frozen=True)
class RpcError:
    """标准化 JSONL RPC 错误载荷。

    Attributes:
        code: 稳定错误码，适合客户端做分支处理。
        message: 人类可读错误消息。
    """

    code: str
    message: str

    def __post_init__(self) -> None:
        """校验错误码和错误消息均为非空字符串。"""
        object.__setattr__(self, "code", require_rpc_text(self.code, field_name="code"))
        object.__setattr__(self, "message", require_rpc_text(self.message, field_name="message"))


async def run_rpc(
    runtime: RuntimeGateway,
    session_id: str,
    *,
    output: OutputFn = print,
) -> None:
    """在 stdin/stdout 上运行 JSONL RPC 循环。

    Args:
        runtime: runtime 网关。
        session_id: 已打开的会话 ID。
        output: 输出函数，默认 ``print``；测试时可替换。

    循环启动后先发送 ``rpc_ready``，随后逐行读取 JSON 请求。空行会被忽略，
    非法 JSON 会被转换成 ``invalid_json`` 响应而不是让进程崩溃。
    """

    def emit(obj: dict[str, Any]) -> None:
        """把 dict 序列化成一行 JSON 输出。"""
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
    """处理一个 JSONL 请求对象。

    Args:
        runtime: runtime 网关。
        session_id: 当前会话 ID。
        req: 反序列化后的 JSON 值，正常应为 dict。
        emit: RPC 输出函数。

    Returns:
        ``True`` 表示收到 shutdown，需要停止 RPC 循环；其他情况返回 ``False``。

    该函数是 RPC 的路由表：不同 ``type`` 会被转换成 runtime action 或状态查询。
    """

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
        elif command == "set_mode":
            await handle_rpc_command(
                runtime,
                session_id,
                req,
                emit,
                command_name="set_mode",
                command_text=f"/mode {require_request_text(req, 'mode')}",
                data_builder=lambda result: {
                    "session_id": session_id,
                    "mode": result.data.get("current_mode"),
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
    """处理 ``prompt`` 请求。

    Args:
        runtime: runtime 网关。
        session_id: 当前会话 ID。
        req: 请求对象，读取 ``text`` 和可选 ``mode``。
        emit: RPC 输出函数。

    prompt 会产生过程事件和最终 response；如果遇到审批，会先输出
    ``approval_required``，最终 response 的 data 状态为 ``waiting_approval``。
    """
    mode = req.get("mode")
    if mode is not None and not isinstance(mode, str):
        raise ValueError("mode must be a string")

    result_data = None
    async for frame in runtime.dispatch(
        session_id,
        PromptSubmitted(text=str(req.get("text", "")), mode_hint=mode),
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
    """处理 ``approve`` 或 ``deny`` 请求。

    Args:
        runtime: runtime 网关。
        session_id: 当前会话 ID。
        req: 请求对象，必须包含 ``approval_id``，可选 ``reason``。
        emit: RPC 输出函数。
        decision: 审批决策，值为 ``approve`` 或 ``deny``。
    """
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
    """把 RPC 请求转成内部斜杠命令并输出响应。

    Args:
        runtime: runtime 网关。
        session_id: 当前会话 ID。
        req: 原始请求，用于读取请求 id。
        emit: RPC 输出函数。
        command_name: 输出 response 中使用的命令名。
        command_text: 发送给 runtime 命令系统的斜杠命令文本。
        data_builder: 把命令结果 record 转成 RPC data 的函数。
    """
    result = await dispatch_rpc_command(runtime, session_id, command_text)
    emit_rpc_ok(
        emit,
        req_id=req.get("id"),
        command=command_name,
        data=data_builder(result),
    )


async def dispatch_rpc_command(runtime: RuntimeGateway, session_id: str, text: str) -> Any:
    """派发 RPC 内部使用的斜杠命令。

    Args:
        runtime: runtime 网关。
        session_id: 当前会话 ID。
        text: 斜杠命令文本。

    Returns:
        ``CommandFinishedFrame.record``。

    Raises:
        RuntimeFrameError: 命令失败或 frame 流异常结束。
    """
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
    """处理 prompt/approval 运行期间产生的单个 runtime frame。

    Args:
        frame: runtime 产生的 frame。
        emit: RPC 输出函数。
        current_result: 当前累计的结果 data；非终态 frame 会原样返回它。

    Returns:
        更新后的结果 data。完成时返回 run record 摘要；等待审批时返回等待状态。
    """
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
    """发送 RPC 就绪事件。

    Args:
        emit: RPC 输出函数。
        session_id: 当前会话 ID。
    """
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
    """发送成功响应。

    Args:
        emit: RPC 输出函数。
        req_id: 原请求 id，会原样带回给客户端。
        command: 响应对应的命令类型。
        data: 可选响应数据。
    """
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
    """发送错误响应。

    Args:
        emit: RPC 输出函数。
        req_id: 原请求 id；解析 JSON 失败时可能为 ``None``。
        command: 原请求 type；未知或无效时可能为 ``None``。
        code: 稳定错误码。
        message: 人类可读错误消息。
    """
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
    """把任意异常转换成 RPC 错误载荷。"""
    raw_code = getattr(exc, "code", None)
    code = raw_code.strip() if isinstance(raw_code, str) and raw_code.strip() else "execution_error"
    message = str(exc).strip() or type(exc).__name__
    return RpcError(code=code, message=message)


def rpc_json_default(value: Any) -> Any:
    """JSON 序列化兜底函数。

    dataclass 会转成 dict，set 会转成 list，其余无法直接序列化的对象转成字符串。
    """
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, set):
        return list(value)
    return str(value)


def require_rpc_text(value: str, *, field_name: str) -> str:
    """校验 RPC 协议内部必需的非空字符串字段。"""
    if not isinstance(value, str):
        raise TypeError(f"RPC {field_name} must be str")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"RPC {field_name} cannot be empty")
    return normalized


def require_request_text(req: dict[str, Any], field_name: str) -> str:
    """从请求中读取必填字符串字段。

    Args:
        req: 请求对象。
        field_name: 字段名。

    Raises:
        ValueError: 字段缺失、不是字符串或为空时抛出。
    """
    value = req.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{req.get('type')} requires {field_name}")
    return value.strip()


def session_state(runtime: RuntimeGateway, session_id: str) -> dict[str, Any]:
    """读取当前 session 对外暴露的状态快照。"""
    return dict(runtime.describe(session_id).state or {})


def run_record_rpc_data(record: Any) -> dict[str, Any]:
    """把 run record 提取成 RPC 响应 data。

    只暴露客户端常用的稳定字段，避免把 runtime 内部对象完整泄露到协议中。
    """
    data: dict[str, Any] = {}
    for name in ("run_id", "session_id", "status", "stop_reason", "final_text"):
        value = getattr(record, name, None)
        if value is not None:
            data[name] = value
    return data


def approval_rpc_data(frame: ApprovalRequiredFrame) -> dict[str, Any]:
    """把审批 frame 转成 RPC 客户端可理解的数据。"""
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
    """从 runtime error payload 中提取错误消息。"""
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or "Runtime failed")
    return str(error)


class RuntimeFrameError(RuntimeError):
    """RPC 消费 runtime frame 时遇到的失败。"""

    code = "runtime.dispatch_failed"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code:
            self.code = code


def runtime_error_from_frame(error: Any) -> RuntimeFrameError:
    """把 ``FailedFrame.error`` 转成 RPC 内部异常。"""
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
