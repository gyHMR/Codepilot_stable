from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from codepilot.llm.api_registry import complete_simple
from codepilot.llm.overflow import estimate_context_tokens
from codepilot.protocols import (
    AssistantMessage,
    Context,
    Message,
    Model,
    SimpleStreamOptions,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

logger = logging.getLogger("codepilot.sessions.context.compaction")

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


CompactionCompleteFn = Callable[
    [Model, Context, SimpleStreamOptions | None],
    Awaitable[AssistantMessage],
]
ApiKeyProvider = Callable[[str], Awaitable[str | None] | str | None]


@dataclass(frozen=True)
class ContextCompactionDecision:
    """一次会话级上下文压缩触发判断。"""

    should_compact: bool
    reason: str
    retain_recent_messages: int
    estimated_tokens: int
    over_message_limit: bool = False
    over_token_limit: bool = False


def decide_context_compaction(
    *,
    message_count: int,
    estimated_tokens: int,
    max_context_messages: int | None,
    max_context_tokens: int | None,
    retain_recent_messages: int,
    force: bool = False,
) -> ContextCompactionDecision:
    """判断会话消息是否需要压缩，并给出触发原因与保留窗口。

    这个函数只负责“是否压缩”的策略判断；摘要生成、工具消息配对修复
    和持久化仍由调用方与 ``build_compacted_context`` 负责。
    """

    retain = max(2, min(retain_recent_messages, message_count - 1))
    over_message_limit = bool(
        max_context_messages
        and max_context_messages > 0
        and message_count > max_context_messages
    )
    over_token_limit = bool(
        max_context_tokens
        and max_context_tokens > 0
        and estimated_tokens > max_context_tokens
    )

    reason = (
        "overflow"
        if force
        else ("token_threshold" if over_token_limit else "message_threshold")
    )
    if not force and not over_message_limit and not over_token_limit:
        return ContextCompactionDecision(
            should_compact=False,
            reason="below_threshold",
            retain_recent_messages=retain,
            estimated_tokens=estimated_tokens,
            over_message_limit=over_message_limit,
            over_token_limit=over_token_limit,
        )
    if message_count <= retain:
        return ContextCompactionDecision(
            should_compact=False,
            reason="not_enough_messages",
            retain_recent_messages=retain,
            estimated_tokens=estimated_tokens,
            over_message_limit=over_message_limit,
            over_token_limit=over_token_limit,
        )
    return ContextCompactionDecision(
        should_compact=True,
        reason=reason,
        retain_recent_messages=retain,
        estimated_tokens=estimated_tokens,
        over_message_limit=over_message_limit,
        over_token_limit=over_token_limit,
    )


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
    compacted, repaired_tool_pairs, unresolved_tool_results = _repair_tool_message_pairs(
        [summary_message, *recent],
        messages,
    )
    tokens_before = estimate_context_tokens(messages, system_prompt)
    tokens_after = estimate_context_tokens(compacted, system_prompt)
    report = {
        "reason": reason,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "retained_messages": retain,
        "truncated_tool_results": sum(isinstance(msg, ToolResultMessage) for msg in older),
        "repaired_tool_pairs": repaired_tool_pairs,
        "unresolved_tool_results": unresolved_tool_results,
        "summary_created": bool(summary_text.strip()),
    }
    return ContextCompactionResult(messages=compacted, report=report)


async def build_llm_compaction_summary(
    messages: list[Message],
    *,
    model: Model,
    get_api_key: ApiKeyProvider | None = None,
    complete_fn: CompactionCompleteFn = complete_simple,
) -> str:
    """使用当前模型把旧消息压缩为会话摘要。

    这个 helper 属于上下文压缩子系统：它只负责把待压缩消息转成摘要
    prompt、调用简化 LLM completion，并提取助手文本。是否触发压缩、
    压缩结果如何写回会话，仍由 ``AgentSession`` 编排。
    """

    formatted = format_messages_for_summary(messages)
    if not formatted.strip():
        return ""

    try:
        summary_context = Context(
            messages=[UserMessage(content=f"请压缩以下对话历史为简明摘要：\n\n{formatted}")],
            system_prompt=COMPACTION_SYSTEM_PROMPT,
        )
        api_key = None
        if get_api_key is not None:
            value = get_api_key(model.provider)
            api_key = await value if inspect.isawaitable(value) else value
        result = await complete_fn(
            model,
            summary_context,
            SimpleStreamOptions(max_tokens=2000, api_key=api_key),
        )
        text_parts = [block.text for block in result.content if isinstance(block, TextContent)]
        summary = "\n".join(text_parts).strip()
        if summary:
            logger.info("LLM compaction summary generated chars=%d", len(summary))
        return summary
    except Exception as exc:
        logger.warning("LLM compaction failed, using fallback: %s", exc)
        return ""


def _repair_tool_message_pairs(
    compacted: list[Message],
    original: list[Message],
) -> tuple[list[Message], int, int]:
    """Ensure retained ToolResultMessage items still have their Assistant ToolCall.

    Session-level compaction rewrites the persisted message list directly.  The
    normal ContextCompiler repair only happens right before provider calls, so
    compaction must also preserve the minimal provider-legal message sequence.
    """

    repaired: list[Message] = []
    seen_tool_call_ids: set[str] = set()
    repaired_count = 0
    unresolved_count = 0
    original_tool_calls = _tool_calls_by_id(original)

    for message in compacted:
        if isinstance(message, ToolResultMessage):
            if message.tool_call_id not in seen_tool_call_ids:
                tool_call = original_tool_calls.get(message.tool_call_id)
                if tool_call is not None:
                    repaired.append(
                        AssistantMessage(
                            content=[tool_call],
                            stop_reason="toolUse",
                        )
                    )
                    seen_tool_call_ids.add(tool_call.id)
                    repaired_count += 1
                else:
                    unresolved_count += 1
        elif isinstance(message, AssistantMessage):
            seen_tool_call_ids.update(
                block.id for block in message.content if isinstance(block, ToolCall)
            )
        repaired.append(message)

    return repaired, repaired_count, unresolved_count


def _tool_calls_by_id(messages: list[Message]) -> dict[str, ToolCall]:
    calls: dict[str, ToolCall] = {}
    for message in messages:
        if not isinstance(message, AssistantMessage):
            continue
        for block in message.content:
            if isinstance(block, ToolCall) and block.id:
                calls.setdefault(block.id, block)
    return calls


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
