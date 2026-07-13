"""Token/上下文估算函数 —— 供 llm 和 sessions 上下文治理共同使用。

本文件提供纯 token/context 估算功能，不处理 provider 流，也不发起模型调用。

核心功能：
1. 文本 token 估算（estimate_text_tokens）—— 按内容类型估算字符数→token
2. 消息 token 估算（estimate_message）—— 逐消息计算 token 消耗
3. 工具定义 token 估算（estimate_tools_tokens）
4. 完整上下文 token 估算（estimate_context）—— 消息 + 系统提示 + 工具
5. ContextUsageCalibrator —— 基于实际使用数据校准校正因子
6. 上下文溢出检测（is_context_overflow、overflow_ratio）

估算策略：
- 基于字符到 token 的经验比例（不同内容类型比例不同）
- 支持校正因子（correction_factors），通过历史数据自动校准
- 中文文本：~1.8 字符/token，英文文本：~4 字符/token
- 代码文本：~3.2 字符/token，JSON Schema：~2.7 字符/token
"""

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from codepilot.protocols import (
    AssistantMessage,
    Context,
    ImageContent,
    Message,
    Model,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


# ── 内容类型 ──────────────────────────────────────────────────────────────────


ContentType = Literal[
    "chinese_text",      # 中文文本
    "english_text",      # 英文文本
    "code_text",         # 代码文本
    "json_schema",       # JSON Schema
    "tool_output",       # 工具输出
    "markdown_mixed",    # Markdown 混合
    "image",             # 图片
    "tool_call_struct",  # 工具调用结构
    "message_overhead",  # 消息开销（每消息固定 token）
]

# ── 经验常数 ──────────────────────────────────────────────────────────────────

CHARS_PER_TOKEN = 4
IMAGE_TOKEN_ESTIMATE = 1000          # 每张图片估算 1000 token
TOOL_SCHEMA_TOKEN_ESTIMATE = 200     # 每个工具 schema 估算 200 token
MESSAGE_OVERHEAD_TOKENS = 4          # 每消息额外开销
TOOL_CALL_OVERHEAD_TOKENS = 8        # 每个工具调用额外开销
CORRECTION_FACTOR_MIN = 0.6          # 校正因子最小范围
CORRECTION_FACTOR_MAX = 1.8          # 校正因子最大范围
EMA_ALPHA = 0.25                      # 指数移动平均的平滑系数

# 各内容类型对应的字符→token 分母（字符数/分母 = 估算 token 数）
_CONTENT_DENOMINATORS: dict[ContentType, float] = {
    "chinese_text": 1.8,
    "english_text": 4.0,
    "code_text": 3.2,
    "json_schema": 2.7,
    "tool_output": 3.0,
    "markdown_mixed": 2.6,
    "image": 1.0,
    "tool_call_struct": 3.0,
    "message_overhead": 1.0,
}


# ── 结果类型 ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TokenEstimate:
    """Token 估算结果。

    参数:
        total: 总 token 数
        by_type: 按内容类型拆分的 token 数
    """
    total: int
    by_type: dict[str, int] = field(default_factory=dict)


# ── 公开估算函数 ──────────────────────────────────────────────────────────────


def estimate_text_tokens(
    text: str,
    *,
    content_type: ContentType | None = None,
    correction_factors: dict[str, float] | None = None,
) -> TokenEstimate:
    """估算文本的 token 数。

    自动检测内容类型，或传入指定的 content_type。
    应用校正因子（如有）进行修正。

    参数:
        text: 要估算的文本
        content_type: 手动指定的内容类型（None=自动检测）
        correction_factors: 校正因子字典（内容类型→因子）

    返回:
        TokenEstimate 包含总 token 数和按类型拆分
    """
    kind = content_type or classify_content_type(text)
    raw = _estimate_by_kind(text, kind)
    factor = _factor_for(kind, correction_factors)
    total = max(1, math.ceil(raw * factor)) if text else 0
    return TokenEstimate(total=total, by_type={kind: total} if total else {})


def estimate_message_tokens(
    msg: Message,
    *,
    correction_factors: dict[str, float] | None = None,
) -> int:
    """估算单条消息的 token 数（快捷方法）。"""
    return estimate_message(msg, correction_factors=correction_factors).total


def estimate_message(
    msg: Message,
    *,
    correction_factors: dict[str, float] | None = None,
) -> TokenEstimate:
    """估算单条消息的 token 数（含按类型拆分）。

    根据不同消息角色分别处理：
    - UserMessage: 文本或文本+图片
    - AssistantMessage: 文本 + 思考 + 工具调用
    - ToolResultMessage: 工具输出 + 图片

    参数:
        msg: 协议消息
        correction_factors: 校正因子

    返回:
        按内容类型拆分的 TokenEstimate
    """
    total = _apply_factor(
        MESSAGE_OVERHEAD_TOKENS,
        "message_overhead",
        correction_factors,
    )
    by_type: dict[str, int] = {"message_overhead": total}
    if isinstance(msg, UserMessage):
        if isinstance(msg.content, str):
            _merge_estimate(
                by_type,
                estimate_text_tokens(
                    msg.content,
                    correction_factors=correction_factors,
                ),
            )
        else:
            for block in msg.content:
                _merge_block_estimate(by_type, block, correction_factors)
    elif isinstance(msg, AssistantMessage):
        for block in msg.content:
            if isinstance(block, TextContent):
                _merge_estimate(
                    by_type,
                    estimate_text_tokens(
                        block.text,
                        correction_factors=correction_factors,
                    ),
                )
            elif isinstance(block, ThinkingContent):
                _merge_estimate(
                    by_type,
                    estimate_text_tokens(
                        block.thinking,
                        content_type="markdown_mixed",
                        correction_factors=correction_factors,
                    ),
                )
            elif isinstance(block, ToolCall):
                _merge_estimate(
                    by_type,
                    estimate_tool_call_tokens(
                        block.name,
                        block.arguments,
                        correction_factors=correction_factors,
                    ),
                )
    elif isinstance(msg, ToolResultMessage):
        for block in msg.content:
            if isinstance(block, TextContent):
                _merge_estimate(
                    by_type,
                    estimate_text_tokens(
                        block.text,
                        content_type="tool_output",
                        correction_factors=correction_factors,
                    ),
                )
            elif isinstance(block, ImageContent):
                _merge_image_estimate(by_type, block, correction_factors)
    return TokenEstimate(total=sum(by_type.values()), by_type=by_type)


def estimate_tool_call_tokens(
    name: str,
    arguments: dict[str, Any] | None,
    *,
    correction_factors: dict[str, float] | None = None,
) -> TokenEstimate:
    """估算一次工具调用的 token 数。

    参数:
        name: 工具名称
        arguments: 工具参数
        correction_factors: 校正因子

    返回:
        TokenEstimate
    """
    text = f"{name} {json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True)}"
    raw = TOOL_CALL_OVERHEAD_TOKENS + _estimate_by_kind(text, "json_schema")
    total = _apply_factor(raw, "tool_call_struct", correction_factors)
    return TokenEstimate(total=total, by_type={"tool_call_struct": total})


def estimate_tools_tokens(
    tools: list[Tool] | None,
    *,
    correction_factors: dict[str, float] | None = None,
) -> TokenEstimate:
    """估算工具定义列表的 token 数。

    参数:
        tools: 工具定义列表
        correction_factors: 校正因子

    返回:
        TokenEstimate
    """
    if not tools:
        return TokenEstimate(total=0)
    by_type: dict[str, int] = {}
    for tool in tools:
        text = json.dumps(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        raw = max(TOOL_SCHEMA_TOKEN_ESTIMATE, _estimate_by_kind(text, "json_schema"))
        total = _apply_factor(raw, "json_schema", correction_factors)
        by_type["tools"] = by_type.get("tools", 0) + total
        by_type["json_schema"] = by_type.get("json_schema", 0) + total
    return TokenEstimate(total=sum(by_type.values()), by_type=by_type)


def estimate_context_tokens(
    messages: list[Message],
    system_prompt: str = "",
    tools: list[Tool] | None = None,
    *,
    correction_factors: dict[str, float] | None = None,
) -> int:
    """估算完整上下文的 token 数（快捷方法）。"""
    return estimate_context(
        messages,
        system_prompt,
        tools,
        correction_factors=correction_factors,
    ).total


def estimate_context(
    messages: list[Message],
    system_prompt: str = "",
    tools: list[Tool] | None = None,
    *,
    correction_factors: dict[str, float] | None = None,
) -> TokenEstimate:
    """估算完整上下文的 token 数（含按类型拆分）。

    包含：系统提示词 + 所有消息 + 工具定义。

    参数:
        messages: 消息列表
        system_prompt: 系统提示词
        tools: 工具定义列表
        correction_factors: 校正因子

    返回:
        按内容类型拆分的 TokenEstimate
    """
    by_type: dict[str, int] = {}
    _merge_estimate(
        by_type,
        estimate_text_tokens(
            system_prompt or "",
            content_type="markdown_mixed",
            correction_factors=correction_factors,
        ),
    )
    for message in messages:
        _merge_estimate(
            by_type,
            estimate_message(message, correction_factors=correction_factors),
        )
    _merge_estimate(
        by_type,
        estimate_tools_tokens(tools, correction_factors=correction_factors),
    )
    return TokenEstimate(total=sum(by_type.values()), by_type=by_type)


def classify_content_type(text: str) -> ContentType:
    """自动分类文本的内容类型。

    检测逻辑（按优先级）：
    1. 中文文本：CJK 字符数 ≥ max(4, 文本长度 * 0.18)
    2. JSON Schema：结构化字符 ≥ max(8, 长度 * 0.16) 且看起来像 JSON
    3. 代码文本：包含代码特征标记
    4. Markdown：包含 ## / - / ` / | 等标记
    5. 英文文本：默认

    参数:
        text: 待分类的文本

    返回:
        识别出的 ContentType
    """
    if not text:
        return "english_text"
    cjk_count = sum(1 for ch in text if "一" <= ch <= "鿿")
    if cjk_count >= max(4, len(text) * 0.18):
        return "chinese_text"
    structural = sum(1 for ch in text if ch in '{}[]":,')
    if structural >= max(8, len(text) * 0.16) and _looks_like_json(text):
        return "json_schema"
    if _looks_like_code(text):
        return "code_text"
    if any(marker in text for marker in ("## ", "- ", "`", "|")):
        return "markdown_mixed"
    return "english_text"


# ── 校准器 ──────────────────────────────────────────────────────────────────


class ContextUsageCalibrator:
    """上下文用量校准器 —— 持久化 provider/model/content-type 的校正因子。

    基于实际 API 返回的 input_tokens 与估算值的偏差，
    使用指数移动平均（EMA）动态调整校正因子。
    校正因子会持久化到 .codepilot/context_usage.json 文件中。

    参数:
        workspace_dir: 工作区目录
    """

    def __init__(self, workspace_dir: str | Path) -> None:
        self.path = Path(workspace_dir) / ".codepilot" / "context_usage.json"

    def factors_for(self, provider: str | None, model: str | None) -> dict[str, float]:
        """获取指定 provider/model 的校正因子。

        参数:
            provider: 提供商名称
            model: 模型 ID

        返回:
            内容类型 → 校正因子的映射
        """
        data = self._load()
        key_prefix = _calibration_prefix(provider, model)
        factors: dict[str, float] = {}
        for key, payload in data.items():
            if not key.startswith(key_prefix + ":") or not isinstance(payload, dict):
                continue
            content_type = key.rsplit(":", 1)[-1]
            factor = payload.get("correction_factor")
            if isinstance(factor, int | float):
                factors[content_type] = _clamp_factor(float(factor))
        return factors

    def update(
        self,
        *,
        provider: str | None,
        model: str | None,
        raw_estimate: int,
        actual_input_tokens: int,
        breakdown: dict[str, int],
    ) -> None:
        """用一次实际 token 用量更新校正因子。

        使用指数移动平均（EMA）更新：
        new_factor = old_factor * (1 - EMA_ALPHA) + sample_factor * EMA_ALPHA

        参数:
            provider: 提供商
            model: 模型
            raw_estimate: 原始估算值（未校正）
            actual_input_tokens: 实际 API 返回的 input_tokens
            breakdown: 按内容类型的 token 拆分
        """
        if raw_estimate <= 0 or actual_input_tokens <= 0 or not breakdown:
            return
        sample_factor = _clamp_factor(actual_input_tokens / raw_estimate)
        data = self._load()
        prefix = _calibration_prefix(provider, model)
        now = datetime.now(timezone.utc).isoformat()
        for content_type, tokens in breakdown.items():
            if tokens <= 0:
                continue
            key = f"{prefix}:{content_type}"
            old = data.get(key) if isinstance(data.get(key), dict) else {}
            previous = old.get("correction_factor", 1.0)
            previous = float(previous) if isinstance(previous, int | float) else 1.0
            factor = _clamp_factor(previous * (1 - EMA_ALPHA) + sample_factor * EMA_ALPHA)
            count = old.get("sample_count", 0)
            count = count if isinstance(count, int) and count >= 0 else 0
            data[key] = {
                "provider": provider or "unknown",
                "model": model or "unknown",
                "content_type": content_type,
                "correction_factor": factor,
                "sample_count": count + 1,
                "updated_at": now,
            }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    def _load(self) -> dict[str, Any]:
        """从文件加载校正因子数据。"""
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}


# ── 溢出检测 ─────────────────────────────────────────────────────────────────


def is_context_overflow(
    model: Model,
    context: Context,
    *,
    safety_margin: float = 0.95,
) -> bool:
    """检查上下文是否可能溢出模型窗口。

    参数:
        model: 模型配置（含 context_window）
        context: 上下文（消息 + 系统提示 + 工具）
        safety_margin: 安全裕度（默认 0.95，即到达 95% 即认为溢出）

    返回:
        True 表示可能溢出
    """
    limit = int(model.context_window * safety_margin)
    estimated = estimate_context_tokens(
        context.messages,
        context.system_prompt or "",
        context.tools,
    )
    return estimated > limit


def overflow_ratio(model: Model, context: Context) -> float:
    """计算上下文占模型窗口的比例。

    参数:
        model: 模型配置
        context: 上下文

    返回:
        占比（0.0 ~ 1.0+，超过 1.0 表示可能溢出）
    """
    estimated = estimate_context_tokens(
        context.messages,
        context.system_prompt or "",
        context.tools,
    )
    if model.context_window <= 0:
        return 0.0
    return estimated / model.context_window


# ── 内部辅助函数 ──────────────────────────────────────────────────────────────


def _estimate_by_kind(text: str, kind: ContentType) -> int:
    """按内容类型估算 token 数（未校正）。

    参数:
        text: 文本
        kind: 内容类型

    返回:
        原始估算 token 数
    """
    if kind == "image":
        return IMAGE_TOKEN_ESTIMATE
    denominator = _CONTENT_DENOMINATORS[kind]
    return max(1, math.ceil(len(text) / denominator)) if text else 0


def _merge_block_estimate(
    by_type: dict[str, int],
    block: TextContent | ImageContent,
    correction_factors: dict[str, float] | None,
) -> None:
    """合并单个内容块（文本或图片）的估算到已有字典。"""
    if isinstance(block, TextContent):
        _merge_estimate(
            by_type,
            estimate_text_tokens(block.text, correction_factors=correction_factors),
        )
    elif isinstance(block, ImageContent):
        _merge_image_estimate(by_type, block, correction_factors)


def _merge_image_estimate(
    by_type: dict[str, int],
    block: ImageContent,
    correction_factors: dict[str, float] | None,
) -> None:
    """合并图片的 token 估算到已有字典。

    不同图片格式的估算值不同（JPEG 略低）。
    """
    raw = IMAGE_TOKEN_ESTIMATE
    if block.mime_type.endswith("jpeg") or block.mime_type.endswith("jpg"):
        raw = 850
    total = _apply_factor(raw, "image", correction_factors)
    by_type["image"] = by_type.get("image", 0) + total


def _merge_estimate(by_type: dict[str, int], estimate: TokenEstimate) -> None:
    """合并 TokenEstimate 到已有字典。"""
    for key, value in estimate.by_type.items():
        by_type[key] = by_type.get(key, 0) + value


def _apply_factor(
    raw_tokens: int,
    content_type: str,
    correction_factors: dict[str, float] | None,
) -> int:
    """应用校正因子到原始 token 数。

    参数:
        raw_tokens: 原始估算 token 数
        content_type: 内容类型
        correction_factors: 校正因子映射

    返回:
        校正后的 token 数
    """
    factor = _factor_for(content_type, correction_factors)
    return max(0, math.ceil(raw_tokens * factor))


def _factor_for(
    content_type: str,
    correction_factors: dict[str, float] | None,
) -> float:
    """获取指定内容类型的校正因子。

    如果没有校正因子数据，返回 1.0（不校正）。
    """
    if not correction_factors:
        return 1.0
    factor = correction_factors.get(content_type)
    if not isinstance(factor, int | float):
        return 1.0
    return _clamp_factor(float(factor))


def _clamp_factor(value: float) -> float:
    """将校正因子限制在 [CORRECTION_FACTOR_MIN, CORRECTION_FACTOR_MAX] 范围内。"""
    return min(CORRECTION_FACTOR_MAX, max(CORRECTION_FACTOR_MIN, value))


def _calibration_prefix(provider: str | None, model: str | None) -> str:
    """生成校准数据的前缀键。

    格式: "provider:model"
    """
    return f"{provider or 'unknown'}:{model or 'unknown'}"


def _looks_like_json(text: str) -> bool:
    """检测文本是否看起来像 JSON。

    检查：以 { / [ 开头、以 } / ] 结尾，或包含 "key": 模式。
    """
    stripped = text.strip()
    if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
        return True
    return bool(re.search(r'"[^"]+"\s*:', text))


def _looks_like_code(text: str) -> bool:
    """检测文本是否看起来像代码。

    检查是否包含 def / class / import / function / const 等代码标记。
    """
    markers = (
        "def ",
        "class ",
        "import ",
        "from ",
        "function ",
        "const ",
        "let ",
        "var ",
        "return ",
        "=>",
        ");",
        "{",
        "}",
    )
    marker_hits = sum(1 for marker in markers if marker in text)
    line_count = len(text.splitlines())
    return marker_hits >= 2 or (line_count >= 3 and any(ch in text for ch in "{};"))


__all__ = [
    "CHARS_PER_TOKEN",
    "ContentType",
    "ContextUsageCalibrator",
    "CORRECTION_FACTOR_MAX",
    "CORRECTION_FACTOR_MIN",
    "IMAGE_TOKEN_ESTIMATE",
    "TOOL_SCHEMA_TOKEN_ESTIMATE",
    "TokenEstimate",
    "classify_content_type",
    "estimate_context",
    "estimate_context_tokens",
    "estimate_message",
    "estimate_message_tokens",
    "estimate_text_tokens",
    "estimate_tool_call_tokens",
    "estimate_tools_tokens",
    "is_context_overflow",
    "overflow_ratio",
]