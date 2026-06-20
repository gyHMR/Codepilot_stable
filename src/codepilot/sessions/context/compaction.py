from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from codepilot.llm.overflow import estimate_context_tokens
from codepilot.protocols import (
    AssistantMessage,
    Message,
    TextContent,
    ToolResultMessage,
    UserMessage,
)

COMPACTION_SYSTEM_PROMPT = """你是一个上下文压缩助手。请根据以下对话历史生成一份简明摘要。
要求：
1. 保留所有关键事实、决策和结论
2. 保留重要的文件路径、代码片段和技术细节
3. 保留用户的偏好和约束条件
4. 移除重复和冗余信息
5. 用简洁的要点形式输出
6. 使用中文

请优先保留下列工作上下文字段：
- 当前任务目标：用户当前真正要完成的目标
- 关键文件：已经阅读、修改或需要继续关注的文件路径
- 关键决策：已确认的方案、约束和命名选择
- 失败原因：失败命令、报错信息和已排除的方向
- 验证：已运行的测试、命令、结果和未验证风险
- 下一步：最合理的继续动作"""


@dataclass(frozen=True)
class ContextCompactionResult:
    messages: list[Message]
    report: dict[str, Any]


def build_compacted_context(
    *,
    messages: list[Message],
    summary_text: str,
    retain_recent_messages: int,
    reason: str,
    system_prompt: str = "",
) -> ContextCompactionResult:
    retain = max(2, min(retain_recent_messages, len(messages) - 1))
    older = messages[:-retain]
    recent = messages[-retain:]
    summary_message = UserMessage(
        content=[TextContent(text=f"[Context Summary]\n{summary_text}")],
    )
    compacted = [summary_message, *recent]
    tokens_before = estimate_context_tokens(messages, system_prompt)
    tokens_after = estimate_context_tokens(compacted, system_prompt)
    report = {
        "reason": reason,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "retained_messages": retain,
        "truncated_tool_results": sum(isinstance(msg, ToolResultMessage) for msg in older),
        "summary_created": bool(summary_text.strip()),
    }
    return ContextCompactionResult(messages=compacted, report=report)


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
