from __future__ import annotations

from codepilot.llm.types import AssistantMessage, Message, TextContent, ToolResultMessage, UserMessage

COMPACTION_SYSTEM_PROMPT = """你是一个上下文压缩助手。请根据以下对话历史生成一份简明摘要。
要求：
1. 保留所有关键事实、决策和结论
2. 保留重要的文件路径、代码片段和技术细节
3. 保留用户的偏好和约束条件
4. 移除重复和冗余信息
5. 用简洁的要点形式输出
6. 使用中文"""


def format_messages_for_summary(messages: list[Message], *, limit: int = 40, text_limit: int = 180) -> str:
    lines: list[str] = []
    for msg in messages[-limit:]:
        if isinstance(msg, UserMessage):
            text = extract_text_from_user(msg, limit=text_limit)
            if text:
                lines.append(f"User: {text}")
        elif isinstance(msg, AssistantMessage):
            text = extract_text_from_assistant(msg, limit=text_limit)
            if text:
                lines.append(f"Assistant: {text}")
        elif isinstance(msg, ToolResultMessage):
            text = extract_text_from_tool_result(msg, limit=text_limit)
            if text:
                lines.append(f"ToolResult({msg.tool_name}): {text}")
    return "\n".join(lines)


def fallback_summary(messages: list[Message], *, limit: int = 20, text_limit: int = 180, max_chars: int = 3000) -> str:
    lines: list[str] = []
    for msg in messages[-limit:]:
        if isinstance(msg, UserMessage):
            text = extract_text_from_user(msg, limit=text_limit)
            if text:
                lines.append(f"- User: {text}")
        elif isinstance(msg, AssistantMessage):
            text = extract_text_from_assistant(msg, limit=text_limit)
            if text:
                lines.append(f"- Assistant: {text}")
        elif isinstance(msg, ToolResultMessage):
            text = extract_text_from_tool_result(msg, limit=text_limit)
            if text:
                lines.append(f"- ToolResult({msg.tool_name}): {text}")
    merged = "\n".join(lines).strip()
    if len(merged) > max_chars:
        merged = merged[:max_chars] + "\n...<summary truncated>..."
    return merged


def extract_text_from_user(message: UserMessage, *, limit: int = 180) -> str:
    if isinstance(message.content, str):
        return message.content[:limit]
    text = "".join(block.text for block in message.content if isinstance(block, TextContent))
    return text[:limit]


def extract_text_from_assistant(message: AssistantMessage, *, limit: int = 180) -> str:
    text = "".join(block.text for block in message.content if isinstance(block, TextContent))
    return text[:limit]


def extract_text_from_tool_result(message: ToolResultMessage, *, limit: int = 180) -> str:
    text = "".join(block.text for block in message.content if isinstance(block, TextContent))
    return text[:limit]
