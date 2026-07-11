from __future__ import annotations

# 新手导读：llm.py 定义跨层共享的模型配置、能力、用量和费用协议。
# 关注点：provider stream 事件和调用选项属于 llm 内部，不放在 protocols。

"""
LLM 相关类型定义。

定义了与大语言模型交互所需的核心类型：
- 模型配置：Model、ModelCapabilities
- 用量与费用：Usage、Cost
- 枚举类型：StopReason、ThinkingLevel
"""

from dataclasses import dataclass, field
from typing import Literal, cast


# ── 类型别名 ────────────────────────────────────────────────────

# API 协议标识（如 "anthropic-messages"、"openai-compatible"）
Api = str
# Provider 厂商标识（如 "anthropic"、"openai"、"deepseek"）
Provider = str

# 停止原因：模型停止生成的各类原因
StopReason = Literal["stop", "length", "toolUse", "error", "aborted", "max_iterations"]

# 思考级别：控制模型推理的深度
ThinkingLevel = Literal["minimal", "low", "medium", "high", "xhigh"]

# 模型输入类型：provider 能接收的内容模态。
ModelInput = Literal["text", "image"]
_MODEL_INPUT_TYPES = frozenset({"text", "image"})

@dataclass
class Cost:
    """费用统计信息。

    单位通常为美元（USD），按输入/输出/缓存分别计量。

    Attributes:
        input: 输入 token 费用。
        output: 输出 token 费用。
        cache_read: 缓存读取费用。
        cache_write: 缓存写入费用。
        total: 总费用（由各分项累加得出）。
    """

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0

    def __post_init__(self) -> None:
        self.input = _non_negative_float(self.input, field_name="cost input")
        self.output = _non_negative_float(self.output, field_name="cost output")
        self.cache_read = _non_negative_float(
            self.cache_read,
            field_name="cost cache_read",
        )
        self.cache_write = _non_negative_float(
            self.cache_write,
            field_name="cost cache_write",
        )
        self.total = _non_negative_float(self.total, field_name="cost total")
        if self.total <= 0:
            self.total = self.input + self.output + self.cache_read + self.cache_write


@dataclass
class Usage:
    """Token 用量统计信息。

    记录一次 LLM 调用的 token 消耗，按输入/输出/缓存分别计量。

    Attributes:
        input: 输入 token 数。
        output: 输出 token 数。
        cache_read: 缓存读取 token 数。
        cache_write: 缓存写入 token 数。
        total_tokens: 总 token 数（由各分项累加得出）。
        cost: 费用统计。
    """

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    cost: Cost = field(default_factory=Cost)

    def __post_init__(self) -> None:
        self.input = _non_negative_int(self.input, field_name="input")
        self.output = _non_negative_int(self.output, field_name="output")
        self.cache_read = _non_negative_int(
            self.cache_read,
            field_name="cache_read",
        )
        self.cache_write = _non_negative_int(
            self.cache_write,
            field_name="cache_write",
        )
        self.total_tokens = _non_negative_int(
            self.total_tokens,
            field_name="total_tokens",
        )
        if not isinstance(self.cost, Cost):
            raise TypeError("Usage cost must be Cost")
        if self.total_tokens <= 0:
            self.total_tokens = (
                self.input + self.output + self.cache_read + self.cache_write
            )


@dataclass
class ModelCapabilities:
    """模型能力声明。

    描述一个具体模型支持的功能特性，用于运行时能力判断。

    Attributes:
        tools: 是否支持工具调用。
        vision: 是否支持图片输入（视觉能力）。
        json_schema: 是否支持 JSON Schema 约束输出。
        streaming: 是否支持流式输出。
        reasoning: 是否支持推理/思考模式。
        system_prompt: 是否支持系统提示词。
        tool_choice: 是否支持指定工具调用策略。
        parallel_tool_calls: 是否支持并行工具调用。
    """

    tools: bool = True
    vision: bool = False
    json_schema: bool = False
    streaming: bool = True
    reasoning: bool = False
    system_prompt: bool = True
    tool_choice: bool = False
    parallel_tool_calls: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "tools",
            "vision",
            "json_schema",
            "streaming",
            "reasoning",
            "system_prompt",
            "tool_choice",
            "parallel_tool_calls",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, bool):
                raise TypeError(f"Model capabilities {field_name} must be bool")


@dataclass
class Model:
    """具体模型的配置信息。

    包含模型的标识、连接信息、能力声明等所有运行时所需的配置。

    Attributes:
        id: 模型 ID（如 "claude-sonnet-4-5"）。
        name: 模型的可读名称。
        api: API 协议标识，决定使用哪个 provider 实现。
        provider: 厂商标识，用于分组和默认认证。
        base_url: API 端点基础 URL。
        reasoning: 是否支持推理模式。
        input: 支持的输入类型列表（"text" / "image"）。
        context_window: 上下文窗口大小（token 数）。
        max_tokens: 最大输出 token 数。
        cost: 费用配置。
        headers: 模型级自定义请求头（可选）。
        capabilities: 模型能力声明（可选，未设置时自动从 input/reasoning 推导）。
    """

    id: str
    name: str
    api: Api
    provider: Provider
    base_url: str
    reasoning: bool
    input: list[ModelInput]
    context_window: int
    max_tokens: int
    cost: Cost = field(default_factory=Cost)
    headers: dict[str, str] | None = None
    capabilities: ModelCapabilities | None = None

    def __post_init__(self) -> None:
        """初始化后处理：如果未显式设置 capabilities，则根据 input 和 reasoning 自动推导。"""
        self.id = _require_model_text(self.id, field_name="model id")
        self.name = _require_model_text(self.name, field_name="model name")
        self.api = _require_model_text(self.api, field_name="model api")
        self.provider = _require_model_text(self.provider, field_name="provider")
        self.base_url = _clean_model_text(self.base_url)
        if not isinstance(self.reasoning, bool):
            raise TypeError("Model reasoning must be bool")
        self.input = _clean_model_inputs(self.input)
        self.context_window = _require_positive_int(
            self.context_window,
            field_name="context_window",
        )
        self.max_tokens = _require_positive_int(
            self.max_tokens,
            field_name="max_tokens",
        )
        if not isinstance(self.cost, Cost):
            raise TypeError("Model cost must be Cost")
        if self.headers is not None:
            if not isinstance(self.headers, dict):
                raise TypeError("Model headers must be a dict or None")
            self.headers = {
                _require_model_text(key, field_name="header name"): _require_model_text(
                    value,
                    field_name="header value",
                )
                for key, value in self.headers.items()
            }
        if self.capabilities is None:
            self.capabilities = ModelCapabilities(
                vision="image" in self.input,
                reasoning=self.reasoning,
            )
        elif not isinstance(self.capabilities, ModelCapabilities):
            raise TypeError("Model capabilities must be ModelCapabilities or None")


def _clean_model_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _require_model_text(value: object, *, field_name: str) -> str:
    text = _clean_model_text(value)
    if not text:
        raise ValueError(f"Model {field_name} cannot be empty")
    return text


def _clean_model_inputs(values: list[ModelInput]) -> list[ModelInput]:
    if not isinstance(values, list):
        raise TypeError("Model input must be a list")
    cleaned: list[ModelInput] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_model_text(value)
        if text not in _MODEL_INPUT_TYPES:
            raise ValueError(f"Unknown model input type: {value}")
        if text not in seen:
            cleaned.append(cast(ModelInput, text))
            seen.add(text)
    if not cleaned:
        raise ValueError("Model input cannot be empty")
    return cleaned


def _require_positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Model {field_name} must be int")
    if value <= 0:
        raise ValueError(f"Model {field_name} must be positive")
    return value


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Usage {field_name} must be int")
    if value < 0:
        raise ValueError(f"Usage {field_name} cannot be negative")
    return value


def _non_negative_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Cost {field_name} must be numeric")
    amount = float(value)
    if amount < 0:
        raise ValueError(f"Cost {field_name} cannot be negative")
    return amount


__all__ = [
    "Api",
    "Cost",
    "Model",
    "ModelCapabilities",
    "Provider",
    "StopReason",
    "ThinkingLevel",
    "Usage",
]
