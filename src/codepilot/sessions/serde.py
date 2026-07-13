"""会话层序列化函数 —— 数据类型与 JSON 字典之间的互转。

本文件提供所有 sessions 层数据类型的序列化/反序列化函数：
- 协议消息（Message）↔ dict
- 会话状态（SessionState）↔ dict
- 运行状态（RunState）↔ dict
- 消息记录（MessageRecord）↔ dict
- 运行检查点（RunCheckpoint）↔ dict

所有双向转换是精确的：序列化后再反序列化应得到完全相同的对象。
"""

from dataclasses import asdict
from typing import Any

from codepilot.protocols import (
    AssistantMessage,
    Cost,
    ImageContent,
    LLMErrorInfo,
    Message,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from codepilot.sessions.contracts import (
    ComponentCheckpoint,
    MessageCursor,
    MessageRecord,
    ModelRef,
    RunCheckpoint,
    RunState,
    SessionState,
    WaitingState,
    WorkspaceCheckpoint,
    WorkspaceEffectsSnapshot,
)


# ── 消息序列化 ────────────────────────────────────────────────────────────────


def message_to_dict(message: Message) -> dict[str, Any]:
    """将协议消息序列化为会话 JSONL 格式的字典。

    支持三种消息角色：
    - user: UserMessage（文本或文本+图片）
    - assistant: AssistantMessage（文本/思考/工具调用 + Usage）
    - toolResult: ToolResultMessage（工具调用结果 + 副作用）

    参数:
        message: 协议层消息对象

    返回:
        JSON 兼容的字典
    """

    if isinstance(message, UserMessage):
        content: str | list[dict[str, Any]]
        if isinstance(message.content, str):
            content = message.content
        else:
            content = [_user_block_to_dict(block) for block in message.content]
        return {
            "role": "user",
            "content": content,
            "timestamp": message.timestamp,
            "metadata": dict(message.metadata),
        }

    if isinstance(message, AssistantMessage):
        return {
            "role": "assistant",
            "content": [_assistant_block_to_dict(block) for block in message.content],
            "api": message.api,
            "provider": message.provider,
            "model": message.model,
            "usage": {
                "input": message.usage.input,
                "output": message.usage.output,
                "cache_read": message.usage.cache_read,
                "cache_write": message.usage.cache_write,
                "total_tokens": message.usage.total_tokens,
                "cost": {
                    "input": message.usage.cost.input,
                    "output": message.usage.cost.output,
                    "cache_read": message.usage.cost.cache_read,
                    "cache_write": message.usage.cost.cache_write,
                    "total": message.usage.cost.total,
                },
            },
            "stop_reason": message.stop_reason,
            "response_id": message.response_id,
            "error_message": message.error_message,
            "error_info": asdict(message.error_info) if message.error_info else None,
            "timestamp": message.timestamp,
            "metadata": dict(message.metadata),
        }

    if isinstance(message, ToolResultMessage):
        return {
            "role": "toolResult",
            "tool_call_id": message.tool_call_id,
            "tool_name": message.tool_name,
            "content": [_tool_result_block_to_dict(block) for block in message.content],
            "status": message.status,
            "is_error": message.is_error,
            "approved": message.approved,
            "approval_id": message.approval_id,
            "error_code": message.error_code,
            "exit_code": message.exit_code,
            "affected_paths": list(message.affected_paths),
            "workspace_changed": message.workspace_changed,
            "diff_summary": message.diff_summary,
            "verification": dict(message.verification) if message.verification else None,
            "details": message.details,
            "timestamp": message.timestamp,
            "metadata": dict(message.metadata),
        }

    raise TypeError(f"Unsupported message type: {type(message)!r}")


def message_from_dict(data: dict[str, Any]) -> Message:
    """从会话 JSONL 格式的字典恢复协议消息。

    与 message_to_dict 互逆。

    参数:
        data: JSON 字典

    返回:
        协议层消息对象
    """

    role = data.get("role")
    if role == "user":
        raw_content = data.get("content", "")
        if isinstance(raw_content, str):
            content: str | list[TextContent | ImageContent] = raw_content
        else:
            content = [
                _user_block_from_dict(item)
                for item in raw_content
                if isinstance(item, dict)
            ]
        return UserMessage(
            content=content,
            timestamp=_int(data.get("timestamp")),
            metadata=_message_metadata(data),
        )

    if role == "assistant":
        usage_data = _dict(data.get("usage"))
        cost_data = _dict(usage_data.get("cost"))
        error_info_data = data.get("error_info")
        return AssistantMessage(
            content=[
                _assistant_block_from_dict(item)
                for item in data.get("content", [])
                if isinstance(item, dict)
            ],
            api=str(data.get("api") or ""),
            provider=str(data.get("provider") or ""),
            model=str(data.get("model") or ""),
            usage=Usage(
                input=_int(usage_data.get("input")),
                output=_int(usage_data.get("output")),
                cache_read=_int(usage_data.get("cache_read")),
                cache_write=_int(usage_data.get("cache_write")),
                total_tokens=_int(usage_data.get("total_tokens")),
                cost=Cost(
                    input=_float(cost_data.get("input")),
                    output=_float(cost_data.get("output")),
                    cache_read=_float(cost_data.get("cache_read")),
                    cache_write=_float(cost_data.get("cache_write")),
                    total=_float(cost_data.get("total")),
                ),
            ),
            stop_reason=data.get("stop_reason", "stop"),
            response_id=data.get("response_id") if isinstance(data.get("response_id"), str) else None,
            error_message=data.get("error_message") if isinstance(data.get("error_message"), str) else None,
            error_info=_error_info(error_info_data),
            timestamp=_int(data.get("timestamp")),
            metadata=_message_metadata(data),
        )

    if role == "toolResult":
        is_error = bool(data.get("is_error", False))
        return ToolResultMessage(
            tool_call_id=str(data.get("tool_call_id") or ""),
            tool_name=str(data.get("tool_name") or ""),
            content=[
                _tool_result_block_from_dict(item)
                for item in data.get("content", [])
                if isinstance(item, dict)
            ],
            status=data.get("status", "error" if is_error else "success"),
            is_error=is_error,
            approved=bool(data.get("approved", True)),
            approval_id=data.get("approval_id") if isinstance(data.get("approval_id"), str) else None,
            error_code=data.get("error_code") if isinstance(data.get("error_code"), str) else None,
            exit_code=data.get("exit_code") if isinstance(data.get("exit_code"), int) else None,
            affected_paths=[
                str(path)
                for path in data.get("affected_paths", [])
                if isinstance(path, str)
            ],
            workspace_changed=data.get("workspace_changed") if isinstance(data.get("workspace_changed"), bool) else None,
            diff_summary=data.get("diff_summary") if isinstance(data.get("diff_summary"), str) else None,
            verification=_dict_or_none(data.get("verification")),
            details=data.get("details"),
            timestamp=_int(data.get("timestamp")),
            metadata=_message_metadata(data),
        )

    raise ValueError(f"Unknown role: {role!r}")


# ── 消息内容块序列化辅助 ────────────────────────────────────────────────────


def _user_block_to_dict(block: TextContent | ImageContent) -> dict[str, Any]:
    if isinstance(block, ImageContent):
        return {
            "type": "image",
            "data": block.data,
            "mime_type": block.mime_type,
            "source": block.source,
        }
    return {"type": "text", "text": block.text, "text_signature": block.text_signature}


def _assistant_block_to_dict(block: TextContent | ThinkingContent | ToolCall) -> dict[str, Any]:
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text, "text_signature": block.text_signature}
    if isinstance(block, ThinkingContent):
        return {
            "type": "thinking",
            "thinking": block.thinking,
            "thinking_signature": block.thinking_signature,
            "redacted": block.redacted,
        }
    return {
        "type": "toolCall",
        "id": block.id,
        "name": block.name,
        "arguments": dict(block.arguments),
        "raw_arguments": block.raw_arguments,
        "index": block.index,
        "provider": block.provider,
        "metadata": dict(block.metadata),
    }


def _tool_result_block_to_dict(block: TextContent | ImageContent) -> dict[str, Any]:
    return _user_block_to_dict(block)


def _user_block_from_dict(data: dict[str, Any]) -> TextContent | ImageContent:
    if data.get("type") == "image":
        return ImageContent(
            data=str(data.get("data") or ""),
            mime_type=str(data.get("mime_type") or "image/png"),
            source=data.get("source") if isinstance(data.get("source"), str) else None,
        )
    return TextContent(
        text=str(data.get("text") or ""),
        text_signature=data.get("text_signature") if isinstance(data.get("text_signature"), str) else None,
    )


def _assistant_block_from_dict(data: dict[str, Any]) -> TextContent | ThinkingContent | ToolCall:
    block_type = data.get("type")
    if block_type == "thinking":
        return ThinkingContent(
            thinking=str(data.get("thinking") or ""),
            thinking_signature=data.get("thinking_signature") if isinstance(data.get("thinking_signature"), str) else None,
            redacted=bool(data.get("redacted", False)),
        )
    if block_type == "toolCall":
        return ToolCall(
            id=str(data.get("id") or ""),
            name=str(data.get("name") or ""),
            arguments=_dict(data.get("arguments")),
            raw_arguments=data.get("raw_arguments") if isinstance(data.get("raw_arguments"), str) else None,
            index=data.get("index") if isinstance(data.get("index"), int) else None,
            provider=data.get("provider") if isinstance(data.get("provider"), str) else None,
            metadata=_dict(data.get("metadata")),
        )
    return TextContent(
        text=str(data.get("text") or ""),
        text_signature=data.get("text_signature") if isinstance(data.get("text_signature"), str) else None,
    )


def _tool_result_block_from_dict(data: dict[str, Any]) -> TextContent | ImageContent:
    return _user_block_from_dict(data)


def _error_info(value: object) -> LLMErrorInfo | None:
    if not isinstance(value, dict):
        return None
    return LLMErrorInfo(
        code=str(value.get("code") or "llm.unknown"),
        message=str(value.get("message") or ""),
        retryable=bool(value.get("retryable", False)),
        kind=value.get("kind", "unknown"),
        provider=str(value.get("provider") or ""),
        model=str(value.get("model") or ""),
        status_code=value.get("status_code"),
        details=_dict(value.get("details")),
    )


def _dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _message_metadata(data: dict[str, Any]) -> dict[str, Any]:
    metadata = _dict(data.get("metadata"))
    message_id = data.get("id")
    if isinstance(message_id, str) and message_id:
        metadata.setdefault("session_message_id", message_id)
    return metadata


def _dict_or_none(value: object) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _float(value: object) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


# ── 会话状态序列化 ────────────────────────────────────────────────────────────


def session_state_to_dict(state: SessionState) -> dict[str, Any]:
    """将 SessionState 序列化为 JSON 字典。"""
    return {
        "schema_version": state.schema_version,
        "session_id": state.session_id,
        "workspace_root": state.workspace_root,
        "current_run_id": state.current_run_id,
        "last_run_id": state.last_run_id,
        "leaf_message_id": state.leaf_message_id,
        "parent_session_id": state.parent_session_id,
        "parent_run_id": state.parent_run_id,
        "session_kind": state.session_kind,
        "current_mode": state.current_mode,
        "model": {"provider": state.model.provider, "model": state.model.model},
        "system_prompt_hash": state.system_prompt_hash,
        "created_at": state.created_at,
        "updated_at": state.updated_at,
        "revision": state.revision,
    }


def session_state_from_dict(data: dict[str, Any]) -> SessionState:
    """从 JSON 字典反序列化 SessionState。"""
    _expect_keys(
        data,
        {
            "schema_version",
            "session_id",
            "workspace_root",
            "current_run_id",
            "last_run_id",
            "leaf_message_id",
            "parent_session_id",
            "parent_run_id",
            "session_kind",
            "current_mode",
            "model",
            "system_prompt_hash",
            "created_at",
            "updated_at",
            "revision",
        },
        "session state",
    )
    model = _required_dict(data["model"], "session model")
    _expect_keys(model, {"provider", "model"}, "session model")
    return SessionState(
        schema_version=data["schema_version"],
        session_id=data["session_id"],
        workspace_root=data["workspace_root"],
        current_run_id=data["current_run_id"],
        last_run_id=data["last_run_id"],
        leaf_message_id=data["leaf_message_id"],
        parent_session_id=data["parent_session_id"],
        parent_run_id=data["parent_run_id"],
        session_kind=data["session_kind"],
        current_mode=data["current_mode"],
        model=ModelRef(provider=model["provider"], model=model["model"]),
        system_prompt_hash=data["system_prompt_hash"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        revision=data["revision"],
    )


# ── 消息记录序列化 ────────────────────────────────────────────────────────────


def message_record_to_dict(record: MessageRecord) -> dict[str, Any]:
    return {
        "schema_version": record.schema_version,
        "message_id": record.message_id,
        "session_id": record.session_id,
        "run_id": record.run_id,
        "parent_id": record.parent_id,
        "created_at": record.created_at,
        "message": message_to_dict(record.message),
    }


def message_record_from_dict(data: dict[str, Any]) -> MessageRecord:
    _expect_keys(
        data,
        {
            "schema_version",
            "message_id",
            "session_id",
            "run_id",
            "parent_id",
            "created_at",
            "message",
        },
        "message record",
    )
    return MessageRecord(
        schema_version=data["schema_version"],
        message_id=data["message_id"],
        session_id=data["session_id"],
        run_id=data["run_id"],
        parent_id=data["parent_id"],
        created_at=data["created_at"],
        message=message_from_dict(_required_dict(data["message"], "message")),
    )


# ── 运行状态序列化 ────────────────────────────────────────────────────────────


def run_state_to_dict(state: RunState) -> dict[str, Any]:
    return {
        "schema_version": state.schema_version,
        "run_id": state.run_id,
        "session_id": state.session_id,
        "request_id": state.request_id,
        "request_digest": state.request_digest,
        "status": state.status,
        "phase": state.phase,
        "stop_reason": state.stop_reason,
        "input_message_id": state.input_message_id,
        "latest_message_id": state.latest_message_id,
        "core_state": dict(state.core_state),
        "checkpoint": _checkpoint_to_dict(state.checkpoint),
        "result_ref": state.result_ref,
        "last_commit_id": state.last_commit_id,
        "last_commit_kind": state.last_commit_kind,
        "last_commit_digest": state.last_commit_digest,
        "last_commit_session_revision": state.last_commit_session_revision,
        "last_commit_message_ids": list(state.last_commit_message_ids),
        "workspace_effects": {
            "changed": state.workspace_effects.changed,
            "affected_paths": list(state.workspace_effects.affected_paths),
            "baseline_ref": state.workspace_effects.baseline_ref,
            "final_fingerprint": state.workspace_effects.final_fingerprint,
        },
        "resume_count": state.resume_count,
        "created_at": state.created_at,
        "updated_at": state.updated_at,
        "started_at": state.started_at,
        "ended_at": state.ended_at,
        "revision": state.revision,
    }


def run_state_from_dict(data: dict[str, Any]) -> RunState:
    _expect_keys(
        data,
        {
            "schema_version",
            "run_id",
            "session_id",
            "request_id",
            "request_digest",
            "status",
            "phase",
            "stop_reason",
            "input_message_id",
            "latest_message_id",
            "core_state",
            "checkpoint",
            "result_ref",
            "last_commit_id",
            "last_commit_kind",
            "last_commit_digest",
            "last_commit_session_revision",
            "last_commit_message_ids",
            "workspace_effects",
            "resume_count",
            "created_at",
            "updated_at",
            "started_at",
            "ended_at",
            "revision",
        },
        "run state",
    )
    effects = _required_dict(data["workspace_effects"], "workspace_effects")
    _expect_keys(
        effects,
        {"changed", "affected_paths", "baseline_ref", "final_fingerprint"},
        "workspace_effects",
    )
    checkpoint = data["checkpoint"]
    return RunState(
        schema_version=data["schema_version"],
        run_id=data["run_id"],
        session_id=data["session_id"],
        request_id=data["request_id"],
        request_digest=data["request_digest"],
        status=data["status"],
        phase=data["phase"],
        stop_reason=data["stop_reason"],
        input_message_id=data["input_message_id"],
        latest_message_id=data["latest_message_id"],
        core_state=_required_dict(data["core_state"], "core_state"),
        checkpoint=(
            _checkpoint_from_dict(_required_dict(checkpoint, "checkpoint"))
            if checkpoint is not None
            else None
        ),
        result_ref=data["result_ref"],
        last_commit_id=data["last_commit_id"],
        last_commit_kind=data["last_commit_kind"],
        last_commit_digest=data["last_commit_digest"],
        last_commit_session_revision=data["last_commit_session_revision"],
        last_commit_message_ids=data["last_commit_message_ids"],
        workspace_effects=WorkspaceEffectsSnapshot(
            changed=effects["changed"],
            affected_paths=effects["affected_paths"],
            baseline_ref=effects["baseline_ref"],
            final_fingerprint=effects["final_fingerprint"],
        ),
        resume_count=data["resume_count"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        started_at=data["started_at"],
        ended_at=data["ended_at"],
        revision=data["revision"],
    )


# ── 检查点序列化辅助 ─────────────────────────────────────────────────────────


def _checkpoint_to_dict(checkpoint: RunCheckpoint | None) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    return {
        "schema_version": checkpoint.schema_version,
        "checkpoint_id": checkpoint.checkpoint_id,
        "resume_point": checkpoint.resume_point,
        "message_cursor": {"leaf_message_id": checkpoint.message_cursor.leaf_message_id},
        "waiting": (
            {
                "kind": checkpoint.waiting.kind,
                "request_id": checkpoint.waiting.request_id,
                "payload": dict(checkpoint.waiting.payload),
            }
            if checkpoint.waiting is not None
            else None
        ),
        "components": [
            {
                "owner": component.owner,
                "schema_version": component.schema_version,
                "state": dict(component.state),
            }
            for component in checkpoint.components
        ],
        "workspace": (
            {
                "root": checkpoint.workspace.root,
                "git_head": checkpoint.workspace.git_head,
                "dirty_paths": list(checkpoint.workspace.dirty_paths),
                "tracked_path_hashes": dict(checkpoint.workspace.tracked_path_hashes),
            }
            if checkpoint.workspace is not None
            else None
        ),
        "created_at": checkpoint.created_at,
    }


def _checkpoint_from_dict(data: dict[str, Any]) -> RunCheckpoint:
    _expect_keys(
        data,
        {
            "schema_version",
            "checkpoint_id",
            "resume_point",
            "message_cursor",
            "waiting",
            "components",
            "workspace",
            "created_at",
        },
        "run checkpoint",
    )
    cursor = _required_dict(data["message_cursor"], "message_cursor")
    _expect_keys(cursor, {"leaf_message_id"}, "message_cursor")
    waiting_raw = data["waiting"]
    waiting: WaitingState | None = None
    if waiting_raw is not None:
        waiting_data = _required_dict(waiting_raw, "waiting")
        _expect_keys(waiting_data, {"kind", "request_id", "payload"}, "waiting")
        waiting = WaitingState(
            kind=waiting_data["kind"],
            request_id=waiting_data["request_id"],
            payload=_required_dict(waiting_data["payload"], "waiting payload"),
        )
    components_raw = data["components"]
    if not isinstance(components_raw, list):
        raise ValueError("run checkpoint components must be a list")
    components: list[ComponentCheckpoint] = []
    for value in components_raw:
        component = _required_dict(value, "checkpoint component")
        _expect_keys(component, {"owner", "schema_version", "state"}, "checkpoint component")
        components.append(
            ComponentCheckpoint(
                owner=component["owner"],
                schema_version=component["schema_version"],
                state=_required_dict(component["state"], "component state"),
            )
        )
    workspace_raw = data["workspace"]
    workspace: WorkspaceCheckpoint | None = None
    if workspace_raw is not None:
        workspace_data = _required_dict(workspace_raw, "workspace checkpoint")
        _expect_keys(
            workspace_data,
            {"root", "git_head", "dirty_paths", "tracked_path_hashes"},
            "workspace checkpoint",
        )
        workspace = WorkspaceCheckpoint(
            root=workspace_data["root"],
            git_head=workspace_data["git_head"],
            dirty_paths=workspace_data["dirty_paths"],
            tracked_path_hashes=_required_dict(
                workspace_data["tracked_path_hashes"],
                "tracked_path_hashes",
            ),
        )
    return RunCheckpoint(
        schema_version=data["schema_version"],
        checkpoint_id=data["checkpoint_id"],
        resume_point=data["resume_point"],
        message_cursor=MessageCursor(leaf_message_id=cursor["leaf_message_id"]),
        waiting=waiting,
        components=tuple(components),
        workspace=workspace,
        created_at=data["created_at"],
    )


# ── 通用辅助函数 ──────────────────────────────────────────────────────────────


def _expect_keys(data: dict[str, Any], expected: set[str], name: str) -> None:
    """验证字典的键集合与期望的集合完全一致。

    如果有多余或缺失的键，抛出 ValueError。
    """
    actual = set(data)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"Invalid {name} fields: missing={missing}, unknown={unknown}")


def _required_dict(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return dict(value)


__all__ = [
    "message_from_dict",
    "message_record_from_dict",
    "message_record_to_dict",
    "message_to_dict",
    "run_state_from_dict",
    "run_state_to_dict",
    "session_state_from_dict",
    "session_state_to_dict",
]
