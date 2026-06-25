from __future__ import annotations

"""
消息转换模块：将 AgentMessage 列表转换为 LLM 可直接消费的 Message 列表。

本模块是 Agent 消息格式与 LLM 消息格式之间的桥梁。
对标 pi-coding-agent 的 convertToLlm 逻辑。

转换流程：
    1. 过滤掉非标准消息类型（只保留 UserMessage、AssistantMessage、ToolResultMessage）
    2. 裁剪过长的 ToolResult 内容，防止上下文膨胀（默认最大 30000 字符）
    3. 处理 thinking 块：根据目标模型能力决定保留 / 转文本 / 移除
    4. 清理空消息
    5. 确保消息序列对 LLM 有效（不以 assistant 开头，不连续出现同 role）

核心函数：
    - convert_to_llm: 主转换函数
    - _convert_single: 转换单条消息
    - _process_assistant: 处理助手消息（thinking 块转换）
    - _process_tool_result: 处理工具结果消息（内容裁剪）
    - _ensure_valid_sequence: 确保消息序列有效性
"""

from codepilot.protocols import (
    AssistantMessage,
    ImageContent,
    Message,
    TextContent,
    ThinkingContent,
    ToolResultMessage,
    UserMessage,
)
from codepilot.core.types import AgentMessage

# ToolResult 文本最大字符数（防止上下文膨胀）
TOOL_RESULT_MAX_CHARS = 30_000

# 内容截断提示（追加到被截断的文本末尾）
TOOL_RESULT_TRUNCATION_NOTICE = "\n...<content truncated>..."


def convert_to_llm(
    messages: list[AgentMessage],
    *,
    strip_thinking: bool = False,
    thinking_to_text: bool = False,
    tool_result_max_chars: int = TOOL_RESULT_MAX_CHARS,
) -> list[Message]:
    """将 AgentMessage 列表转换为 LLM 可消费的 Message 列表。

    这是消息转换的主入口函数，处理流程：
    1. 遍历每条消息，调用 _convert_single 进行转换
    2. 过滤掉无法转换的消息（返回 None）
    3. 调用 _ensure_valid_sequence 确保消息序列有效

    Args:
        messages: Agent 消息列表。
        strip_thinking: 是否完全移除 thinking 块（适用于不支持 thinking 的模型）。
        thinking_to_text: 是否将 thinking 块转为 TextContent（跨 Provider 切换时使用）。
        tool_result_max_chars: ToolResult 文本最大字符数，超出则截断。

    Returns:
        list[Message]: LLM 可消费的消息列表。
    """
    result: list[Message] = []
    for msg in messages:
        converted = _convert_single(
            msg,
            strip_thinking=strip_thinking,
            thinking_to_text=thinking_to_text,
            tool_result_max_chars=tool_result_max_chars,
        )
        if converted is not None:
            result.append(converted)
    return _ensure_valid_sequence(result)


def _convert_single(
    msg: AgentMessage,
    *,
    strip_thinking: bool,
    thinking_to_text: bool,
    tool_result_max_chars: int,
) -> Message | None:
    """转换单条消息：根据消息类型分发到不同的处理函数。

    Args:
        msg: 待转换的消息。
        strip_thinking: 是否移除 thinking 块。
        thinking_to_text: 是否将 thinking 块转为文本。
        tool_result_max_chars: ToolResult 最大字符数。

    Returns:
        Message | None: 转换后的消息，无法转换时返回 None。
    """
    # UserMessage 直接透传
    if isinstance(msg, UserMessage):
        return msg

    # AssistantMessage 需要处理 thinking 块
    if isinstance(msg, AssistantMessage):
        return _process_assistant(msg, strip_thinking=strip_thinking, thinking_to_text=thinking_to_text)

    # ToolResultMessage 需要裁剪过长内容
    if isinstance(msg, ToolResultMessage):
        return _process_tool_result(msg, max_chars=tool_result_max_chars)

    # 未知消息类型，过滤掉
    return None


def _process_assistant(
    msg: AssistantMessage,
    *,
    strip_thinking: bool,
    thinking_to_text: bool,
) -> AssistantMessage:
    """处理助手消息：根据配置处理 thinking 块。

    处理逻辑：
    - 如果不需要处理 thinking 块，直接返回原消息
    - strip_thinking=True: 完全移除 thinking 块
    - thinking_to_text=True: 将 thinking 块转为 [thinking]...[/thinking] 格式的文本

    Args:
        msg: 助手消息。
        strip_thinking: 是否移除 thinking 块。
        thinking_to_text: 是否将 thinking 块转为文本。

    Returns:
        AssistantMessage: 处理后的助手消息。
    """
    # 如果不需要处理 thinking 块，直接返回
    if not strip_thinking and not thinking_to_text:
        return msg

    new_content = []
    for block in msg.content:
        if isinstance(block, ThinkingContent):
            if strip_thinking:
                # 移除 thinking 块
                continue
            if thinking_to_text and block.thinking:
                # 将 thinking 块转为文本格式
                new_content.append(TextContent(text=f"[thinking]\n{block.thinking}\n[/thinking]"))
                continue
        new_content.append(block)

    # 确保至少有一个内容块
    if not new_content:
        new_content = [TextContent(text="(no content)")]

    # 构建新的 AssistantMessage（保留所有元数据）
    return AssistantMessage(
        role=msg.role,
        content=new_content,
        api=msg.api,
        provider=msg.provider,
        model=msg.model,
        usage=msg.usage,
        stop_reason=msg.stop_reason,
        response_id=msg.response_id,
        error_message=msg.error_message,
        error_info=msg.error_info,
        timestamp=msg.timestamp,
        metadata=dict(msg.metadata),
    )


def _process_tool_result(msg: ToolResultMessage, *, max_chars: int) -> ToolResultMessage:
    """处理工具结果消息：裁剪过长的文本内容，防止上下文膨胀。

    当工具输出过长时（如日志、代码等），会占用大量上下文窗口。
    此函数将文本内容截断到指定最大字符数，并在末尾添加截断提示。

    处理逻辑：
    1. 计算所有 TextContent 的总字符数
    2. 如果未超限，直接返回原消息
    3. 如果超限，逐个裁剪 TextContent 直到达到限制
    4. ImageContent 不参与裁剪（保留原样）

    Args:
        msg: 工具结果消息。
        max_chars: 文本内容最大字符数。

    Returns:
        ToolResultMessage: 处理后的工具结果消息。
    """
    # 计算文本总字符数
    total_chars = sum(len(b.text) for b in msg.content if isinstance(b, TextContent))
    # 未超限则直接返回
    if total_chars <= max_chars:
        return msg

    # 裁剪文本内容
    new_content = []
    remaining = max_chars
    for block in msg.content:
        if isinstance(block, TextContent):
            if remaining <= 0:
                # 已达到限制，跳过剩余文本
                continue
            if len(block.text) > remaining:
                # 裁剪到剩余空间，并添加截断提示
                new_content.append(TextContent(text=block.text[:remaining] + TOOL_RESULT_TRUNCATION_NOTICE))
                remaining = 0
            else:
                new_content.append(block)
                remaining -= len(block.text)
        elif isinstance(block, ImageContent):
            # 图片内容不参与裁剪
            new_content.append(block)

    # 构建新的 ToolResultMessage（保留所有元数据）
    return ToolResultMessage(
        role=msg.role,
        tool_call_id=msg.tool_call_id,
        tool_name=msg.tool_name,
        content=new_content,
        status=msg.status,
        is_error=msg.is_error,
        approved=msg.approved,
        approval_id=msg.approval_id,
        error_code=msg.error_code,
        exit_code=msg.exit_code,
        affected_paths=list(msg.affected_paths),
        workspace_changed=msg.workspace_changed,
        diff_summary=msg.diff_summary,
        verification=dict(msg.verification) if msg.verification else None,
        details=msg.details,
        timestamp=msg.timestamp,
        metadata=dict(msg.metadata),
    )


def _ensure_valid_sequence(messages: list[Message]) -> list[Message]:
    """确保消息序列对 LLM 有效：不以 assistant 开头，不连续出现同 role。

    大多数 LLM API 要求消息序列满足以下规则：
    1. 不能以 assistant 消息开头（需要用户先发起对话）
    2. 不能连续出现两条同 role 的消息（需要交替进行）

    此函数通过过滤连续的 assistant 消息来确保序列有效。

    Args:
        messages: 消息列表。

    Returns:
        list[Message]: 有效的消息列表。
    """
    if not messages:
        return messages
    result: list[Message] = []
    for msg in messages:
        if not result:
            # 第一条消息直接添加
            result.append(msg)
            continue
        prev = result[-1]
        if isinstance(prev, AssistantMessage) and isinstance(msg, AssistantMessage):
            # 连续的 assistant 消息，跳过后面的
            continue
        result.append(msg)
    return result
