from __future__ import annotations

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
from codepilot.protocols.tools import coerce_tool_result_status


def message_to_dict(message: Message) -> dict[str, Any]:
    """Serialize protocol messages into the session JSONL format."""

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
    """Restore a protocol message from the session JSONL format."""

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
            metadata=_dict(data.get("metadata")),
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
            metadata=_dict(data.get("metadata")),
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
            status=coerce_tool_result_status(
                data.get("status"),
                default="error" if is_error else "success",
            ),
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
            metadata=_dict(data.get("metadata")),
        )

    raise ValueError(f"Unknown role: {role!r}")


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


def _dict_or_none(value: object) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _float(value: object) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


__all__ = ["message_from_dict", "message_to_dict"]
