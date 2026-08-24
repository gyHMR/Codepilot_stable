"""Provider 共享工具函数 —— 消息转换、工具 Schema 转换和响应解析辅助。

多个 provider 共用的格式逻辑集中在此文件：

1. 消息转换：
   - to_openai_messages() — 统一 Message → OpenAI Chat Completions 格式
   - to_anthropic_messages() — 统一 Message → Anthropic Messages API 格式
   - to_openai_tools() / to_anthropic_tools() — 工具定义格式转换

2. 工具参数解析：
   - parse_partial_json() — 解析流式工具参数（可能是半截 JSON）
   - finalize_tool_arguments() — 完成流式工具参数的最终解析

3. 辅助：
   - empty_assistant_message() — 创建最小可用的 AssistantMessage
   - normalize_usage() — 标准化 token 用量数据
"""

import json
import time
from typing import Any

from codepilot.protocols import (
    AssistantMessage,
    Context,
    ImageContent,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def now_ms() -> int:
    """获取当前时间戳（毫秒）。"""
    return int(time.time() * 1000)


# ── 工具参数解析 ──────────────────────────────────────────────────────────────


def parse_partial_json(raw: str) -> dict[str, Any]:
    """解析流式工具参数（可能是半截 JSON）。

    在流式过程中，tool_call 的 arguments 是一段段拼接的 JSON 片段，
    这个函数尝试解析当前累积的片段，解析失败时返回 {} 让上层保持稳态。

    参数:
        raw: 累加的参数 JSON 字符串（可能不完整）

    返回:
        解析成功的 dict，或空 dict
    """
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def finalize_tool_arguments(tool_call: ToolCall, raw: str) -> None:
    """完成流式工具参数的最终解析 —— 在工具调用块结束时调用。

    对累积的完整 JSON 做严格解析，设置 tool_call.arguments。
    如果解析失败，设置 argument_parse_error 元数据。

    参数:
        tool_call: 工具调用对象（会被修改）
        raw: 累积的完整参数 JSON 字符串
    """
    tool_call.raw_arguments = raw
    try:
        value = json.loads(raw or "{}")
        if not isinstance(value, dict):
            raise ValueError("Tool arguments must be a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        tool_call.arguments = {}
        tool_call.metadata["argument_parse_error"] = f"Invalid tool arguments: {exc}"
        return
    tool_call.arguments = value
    tool_call.metadata.pop("argument_parse_error", None)


# ── 消息构造辅助 ──────────────────────────────────────────────────────────────


def empty_assistant_message(api: str, provider: str, model: str) -> AssistantMessage:
    """创建一个最小可用的 AssistantMessage，用于边流式边填充。

    流式过程中逐步填充 content、usage、stop_reason 等字段，
    最终通过 end() 返回完整的消息。

    参数:
        api: API 协议标识
        provider: 提供商名称
        model: 模型 ID

    返回:
        空的 AssistantMessage
    """
    return AssistantMessage(
        content=[],
        api=api,
        provider=provider,
        model=model,
        usage=Usage(),
        timestamp=now_ms(),
    )


def normalize_usage(usage: Usage) -> Usage:
    """标准化 token 用量数据 —— 确保派生字段填充一致。

    计算 total_tokens 和 cost.total 等聚合字段，
    确保统计口径一致。

    参数:
        usage: 要标准化的 Usage 对象

    返回:
        标准化后的 Usage（同一对象，方便链式调用）
    """
    if usage.total_tokens <= 0:
        usage.total_tokens = usage.input + usage.output + usage.cache_read + usage.cache_write
    if usage.cost.total <= 0:
        usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
    return usage


# ── OpenAI 格式转换 ───────────────────────────────────────────────────────────


def to_openai_messages(context: Context) -> list[dict[str, Any]]:
    """把统一 Message 列表转换成 OpenAI Chat Completions 的 messages 格式。

    转换规则：
    - UserMessage → {"role": "user", "content": text/parts}
    - AssistantMessage → {"role": "assistant", "content": text, "tool_calls": [...]}
    - ToolResultMessage → {"role": "tool", "tool_call_id": "...", "content": "..."}

    参数:
        context: 统一上下文

    返回:
        OpenAI 格式的消息列表
    """
    out: list[dict[str, Any]] = []
    for msg in context.messages:
        if isinstance(msg, UserMessage):
            if isinstance(msg.content, str):
                out.append({"role": "user", "content": msg.content})
            else:
                parts: list[dict[str, Any]] = []
                for part in msg.content:
                    if isinstance(part, TextContent):
                        parts.append({"type": "text", "text": part.text})
                    elif isinstance(part, ImageContent):
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{part.mime_type};base64,{part.data}"},
                            }
                        )
                out.append({"role": "user", "content": parts})
        elif isinstance(msg, AssistantMessage):
            text = "".join(b.text for b in msg.content if isinstance(b, TextContent))
            tool_calls = [b for b in msg.content if isinstance(b, ToolCall)]
            payload: dict[str, Any] = {"role": "assistant", "content": text}
            if tool_calls:
                payload["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                    }
                    for tc in tool_calls
                ]
            out.append(payload)
        elif isinstance(msg, ToolResultMessage):
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "name": msg.tool_name,
                    "content": "\n".join(
                        p.text for p in msg.content if isinstance(p, TextContent) and isinstance(p.text, str)
                    ),
                }
            )
    return out


def to_openai_tools(tools: list[Tool] | None) -> list[dict[str, Any]] | None:
    """把统一 Tool 定义转换成 OpenAI tools 格式。

    参数:
        tools: 统一工具定义列表

    返回:
        OpenAI 格式的工具列表，或 None
    """
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


# ── Anthropic 格式转换 ────────────────────────────────────────────────────────


def to_anthropic_messages(context: Context) -> list[dict[str, Any]]:
    """把统一 Message 列表转换成 Anthropic Messages API 的 messages 格式。

    转换规则：
    - UserMessage → {"role": "user", "content": text/parts}
    - AssistantMessage → {"role": "assistant", "content": [text_blocks, tool_use_blocks]}
    - ToolResultMessage → {"role": "user", "content": [tool_result_blocks]}

    注意：Anthropic 的 ToolResultMessage 使用 role="user"，这与 OpenAI 不同。

    参数:
        context: 统一上下文

    返回:
        Anthropic 格式的消息列表
    """
    out: list[dict[str, Any]] = []
    for msg in context.messages:
        if isinstance(msg, UserMessage):
            if isinstance(msg.content, str):
                out.append({"role": "user", "content": msg.content})
            else:
                parts: list[dict[str, Any]] = []
                for part in msg.content:
                    if isinstance(part, TextContent):
                        parts.append({"type": "text", "text": part.text})
                    elif isinstance(part, ImageContent):
                        parts.append(
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": part.mime_type, "data": part.data},
                            }
                        )
                out.append({"role": "user", "content": parts})
        elif isinstance(msg, AssistantMessage):
            parts: list[dict[str, Any]] = []
            for block in msg.content:
                if isinstance(block, TextContent):
                    parts.append({"type": "text", "text": block.text})
                elif isinstance(block, ToolCall):
                    parts.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.arguments,
                        }
                    )
            out.append({"role": "assistant", "content": parts})
        elif isinstance(msg, ToolResultMessage):
            text = "\n".join(p.text for p in msg.content if isinstance(p, TextContent))
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": [{"type": "text", "text": text}],
                            "is_error": msg.is_error,
                        }
                    ],
                }
            )
    return out


def to_anthropic_tools(tools: list[Tool] | None) -> list[dict[str, Any]] | None:
    """把统一 Tool 定义转换成 Anthropic tools 格式。

    Anthropic 的 tools 格式使用 input_schema 而不是 OpenAI 的 parameters。

    参数:
        tools: 统一工具定义列表

    返回:
        Anthropic 格式的工具列表，或 None
    """
    if not tools:
        return None
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in tools
    ]