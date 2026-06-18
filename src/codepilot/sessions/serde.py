from __future__ import annotations

"""
消息序列化/反序列化工具。

目标：
1) 把 ai 层 dataclass 消息安全写入 jsonl；
2) 下次启动时恢复为同等结构，继续参与 Agent 推理。
"""

from typing import Any
from dataclasses import asdict

from codepilot.protocols import (
    AssistantMessage,
    Cost,
    ImageContent,
    Message,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def _user_block_to_dict(block: TextContent | ImageContent) -> dict[str, Any]:
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text, "text_signature": block.text_signature}
    return {"type": "image", "data": block.data, "mime_type": block.mime_type}


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
    return {"type": "toolCall", "id": block.id, "name": block.name, "arguments": block.arguments}


def _tool_result_block_to_dict(block: TextContent | ImageContent) -> dict[str, Any]:
    return _user_block_to_dict(block)


def message_to_dict(message: Message) -> dict[str, Any]:
    """
    将 Message 转成可持久化 dict。
    """

    if isinstance(message, UserMessage):
        content: str | list[dict[str, Any]]
        if isinstance(message.content, str):
            content = message.content
        else:
            content = [_user_block_to_dict(b) for b in message.content]
        return {"role": "user", "content": content, "timestamp": message.timestamp}

    if isinstance(message, AssistantMessage):
        return {
            "role": "assistant",
            "content": [_assistant_block_to_dict(b) for b in message.content],
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
            "metadata": message.metadata,
        }

    if isinstance(message, ToolResultMessage):
        return {
            "role": "toolResult",
            "tool_call_id": message.tool_call_id,
            "tool_name": message.tool_name,
            "content": [_tool_result_block_to_dict(b) for b in message.content],
            "status": message.status,
            "is_error": message.is_error,
            "error_code": message.error_code,
            "exit_code": message.exit_code,
            "affected_paths": message.affected_paths,
            "workspace_changed": message.workspace_changed,
            "diff_summary": message.diff_summary,
            "verification": message.verification,
            "details": message.details,
            "timestamp": message.timestamp,
            "metadata": message.metadata,
        }

    raise TypeError(f"Unsupported message type: {type(message)!r}")


def _user_block_from_dict(data: dict[str, Any]) -> TextContent | ImageContent:
    if data.get("type") == "image":
        return ImageContent(data=data.get("data", ""), mime_type=data.get("mime_type", "image/png"))
    return TextContent(text=data.get("text", ""), text_signature=data.get("text_signature"))


def _assistant_block_from_dict(data: dict[str, Any]) -> TextContent | ThinkingContent | ToolCall:
    t = data.get("type")
    if t == "thinking":
        return ThinkingContent(
            thinking=data.get("thinking", ""),
            thinking_signature=data.get("thinking_signature"),
            redacted=bool(data.get("redacted", False)),
        )
    if t == "toolCall":
        return ToolCall(
            id=data.get("id", ""),
            name=data.get("name", ""),
            arguments=data.get("arguments", {}) if isinstance(data.get("arguments"), dict) else {},
        )
    return TextContent(text=data.get("text", ""), text_signature=data.get("text_signature"))


def _tool_result_block_from_dict(data: dict[str, Any]) -> TextContent | ImageContent:
    return _user_block_from_dict(data)


def message_from_dict(data: dict[str, Any]) -> Message:
    """
    将持久化 dict 恢复成 Message。
    """

    role = data.get("role")
    if role == "user":
        raw_content = data.get("content", "")
        if isinstance(raw_content, str):
            content: str | list[TextContent | ImageContent] = raw_content
        else:
            content = [_user_block_from_dict(i) for i in raw_content if isinstance(i, dict)]
        return UserMessage(content=content, timestamp=int(data.get("timestamp", 0)))

    if role == "assistant":
        usage_data = data.get("usage", {})
        cost_data = usage_data.get("cost", {}) if isinstance(usage_data, dict) else {}
        usage = Usage(
            input=int(usage_data.get("input", 0)) if isinstance(usage_data, dict) else 0,
            output=int(usage_data.get("output", 0)) if isinstance(usage_data, dict) else 0,
            cache_read=int(usage_data.get("cache_read", 0)) if isinstance(usage_data, dict) else 0,
            cache_write=int(usage_data.get("cache_write", 0)) if isinstance(usage_data, dict) else 0,
            total_tokens=int(usage_data.get("total_tokens", 0)) if isinstance(usage_data, dict) else 0,
            cost=Cost(
                input=float(cost_data.get("input", 0.0)) if isinstance(cost_data, dict) else 0.0,
                output=float(cost_data.get("output", 0.0)) if isinstance(cost_data, dict) else 0.0,
                cache_read=float(cost_data.get("cache_read", 0.0)) if isinstance(cost_data, dict) else 0.0,
                cache_write=float(cost_data.get("cache_write", 0.0)) if isinstance(cost_data, dict) else 0.0,
                total=float(cost_data.get("total", 0.0)) if isinstance(cost_data, dict) else 0.0,
            ),
        )
        error_info_data = data.get("error_info")
        error_info = None
        if isinstance(error_info_data, dict):
            error_info = LLMErrorInfo(
                code=str(error_info_data.get("code", "llm.unknown")),
                message=str(error_info_data.get("message", "")),
                retryable=bool(error_info_data.get("retryable", False)),
                kind=error_info_data.get("kind", "unknown"),
                provider=str(error_info_data.get("provider", "")),
                model=str(error_info_data.get("model", "")),
                status_code=error_info_data.get("status_code"),
                details=error_info_data.get("details", {})
                if isinstance(error_info_data.get("details"), dict)
                else {},
            )
        return AssistantMessage(
            content=[_assistant_block_from_dict(i) for i in data.get("content", []) if isinstance(i, dict)],
            api=data.get("api", ""),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            usage=usage,
            stop_reason=data.get("stop_reason", "stop"),
            response_id=data.get("response_id"),
            error_message=data.get("error_message"),
            error_info=error_info,
            timestamp=int(data.get("timestamp", 0)),
            metadata=data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {},
        )

    if role == "toolResult":
        return ToolResultMessage(
            tool_call_id=data.get("tool_call_id", ""),
            tool_name=data.get("tool_name", ""),
            content=[_tool_result_block_from_dict(i) for i in data.get("content", []) if isinstance(i, dict)],
            status=data.get("status", "error" if data.get("is_error") else "success"),
            is_error=bool(data.get("is_error", False)),
            error_code=data.get("error_code"),
            exit_code=data.get("exit_code"),
            affected_paths=[
                str(path)
                for path in data.get("affected_paths", [])
                if isinstance(path, str)
            ],
            workspace_changed=(
                data.get("workspace_changed")
                if isinstance(data.get("workspace_changed"), bool)
                else None
            ),
            diff_summary=data.get("diff_summary"),
            verification=(
                data.get("verification")
                if isinstance(data.get("verification"), dict)
                else None
            ),
            details=data.get("details"),
            timestamp=int(data.get("timestamp", 0)),
            metadata=data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {},
        )

    raise ValueError(f"Unknown role: {role!r}")
