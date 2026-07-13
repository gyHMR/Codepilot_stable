"""LLM 端口定义 —— 核心层与 LLM 提供商之间的协议接口。

本文件定义了 ModelPort 协议和流式事件的类型体系：

事件流：
    LLMRequest → stream() → AsyncIterator[LLMEvent]
         ↓                            ↓
    ModelDescriptor + messages    LLMStarted / LLMTextDelta / LLMReasoningDelta
    + options + tools             LLMToolCallDelta / LLMCompleted / LLMFailed

ModelPort 是 core-facing 的端口契约，ProviderModelPort（在 adapter.py 中）
是现有的 provider registry 适配到此端口的具体实现。
"""

from dataclasses import dataclass, field
from typing import AsyncIterator, Literal, Protocol

from codepilot.protocols import (
    AssistantMessage,
    LLMErrorInfo,
    Message,
    Tool,
    ToolCall,
    Usage,
)


# ── 请求类型 ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelDescriptor:
    """模型描述符 —— 标识要使用的 LLM 模型。

    参数:
        provider: LLM 提供商名称（如 "anthropic"、"openai"）
        model_id: 模型 ID（如 "claude-sonnet-4-20250514"）
        capabilities: 模型能力字典（可选，扩展用）
    """
    provider: str
    model_id: str
    capabilities: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMOptions:
    """LLM 调用选项 —— 控制模型行为的参数。

    参数:
        temperature: 温度参数（控制随机性，None=使用模型默认值）
        max_tokens: 最大输出 token 数（None=使用模型默认值）
        reasoning: 推理级别（如 "low" / "medium" / "high"）
        timeout_seconds: HTTP 请求超时（秒，默认 120）
    """
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning: str | None = None
    timeout_seconds: float = 120.0


@dataclass(frozen=True)
class LLMCorrelation:
    """LLM 调用关联信息 —— 用于追踪和日志关联。

    参数:
        run_id: 运行 ID
        session_id: 会话 ID
    """
    run_id: str = ""
    session_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", str(self.run_id).strip())
        object.__setattr__(self, "session_id", str(self.session_id).strip())


@dataclass(frozen=True)
class LLMRequest:
    """LLM 请求 —— 一次模型调用的完整请求数据。

    参数:
        model: 模型描述符（标识目标模型）
        messages: 消息列表（构成对话上下文）
        system_prompt: 系统提示词
        tools: 可用工具列表
        options: 模型调用选项
        correlation: 关联信息（追踪用）
    """
    model: ModelDescriptor
    messages: tuple[Message, ...]
    system_prompt: str = ""
    tools: tuple[Tool, ...] = field(default_factory=tuple)
    options: LLMOptions = field(default_factory=LLMOptions)
    correlation: LLMCorrelation = field(default_factory=LLMCorrelation)


# ── 事件类型 ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LLMStarted:
    """流开始事件 —— 表示模型开始响应。"""
    kind: Literal["started"] = "started"


@dataclass(frozen=True)
class LLMTextDelta:
    """文本增量事件 —— 模型生成的文本片段。

    参数:
        text: 增量的文本内容
    """
    text: str
    kind: Literal["text_delta"] = "text_delta"


@dataclass(frozen=True)
class LLMReasoningDelta:
    """推理增量事件 —— 模型的思考/推理过程片段。

    参数:
        text: 增量的推理内容
    """
    text: str
    kind: Literal["reasoning_delta"] = "reasoning_delta"


@dataclass(frozen=True)
class LLMToolCallDelta:
    """工具调用增量事件 —— 模型生成的工具调用片段。

    参数:
        tool_call: 工具调用对象（在流式过程中可能部分填充）
    """
    tool_call: ToolCall
    kind: Literal["tool_call_delta"] = "tool_call_delta"


@dataclass(frozen=True)
class LLMCompleted:
    """流完成事件 —— 模型响应已完成。

    参数:
        message: 完整的 AssistantMessage
        usage: Token 用量统计
    """
    message: AssistantMessage
    usage: Usage | None = None
    kind: Literal["completed"] = "completed"


@dataclass(frozen=True)
class LLMFailed:
    """流失败事件 —— 模型调用过程中发生错误。

    参数:
        error: 错误信息（LLMErrorInfo / Exception / dict）
    """
    error: LLMErrorInfo | Exception | dict[str, object]
    kind: Literal["failed"] = "failed"


# LLMEvent —— 所有流式事件的联合类型
LLMEvent = (
    LLMStarted
    | LLMTextDelta
    | LLMReasoningDelta
    | LLMToolCallDelta
    | LLMCompleted
    | LLMFailed
)


# ── 端口协议 ──────────────────────────────────────────────────────────────────


class ModelPort(Protocol):
    """模型端口 —— LLM 调用的核心协议接口。

    提供统一的流式调用方法，消费方通过 async for 逐个消费事件，
    或消费 LLMCompleted 事件获取最终结果。
    """

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        """发起一次流式 LLM 调用。

        返回的事件流包含以下事件，按顺序出现：
        1. LLMStarted（可选，标记开始）
        2. 零或多个 LLMTextDelta / LLMReasoningDelta / LLMToolCallDelta
        3. LLMCompleted（成功结束）或 LLMFailed（失败结束）

        参数:
            request: LLM 请求（模型、消息、工具、选项等）

        返回:
            LLMEvent 的异步迭代器
        """
        ...


__all__ = [
    "LLMCompleted",
    "LLMCorrelation",
    "LLMEvent",
    "LLMFailed",
    "LLMOptions",
    "LLMRequest",
    "LLMReasoningDelta",
    "LLMStarted",
    "LLMTextDelta",
    "LLMToolCallDelta",
    "ModelDescriptor",
    "ModelPort",
]