from __future__ import annotations

# 新手导读：estimation.py 放纯 token/context 估算函数，供 llm 和 sessions 上下文治理共同使用。
# 关注点：这里不处理 provider 流，也不发起模型调用。

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


ContentType = Literal[
    "chinese_text",
    "english_text",
    "code_text",
    "json_schema",
    "tool_output",
    "markdown_mixed",
    "image",
    "tool_call_struct",
    "message_overhead",
]

CHARS_PER_TOKEN = 4
IMAGE_TOKEN_ESTIMATE = 1000
TOOL_SCHEMA_TOKEN_ESTIMATE = 200
MESSAGE_OVERHEAD_TOKENS = 4
TOOL_CALL_OVERHEAD_TOKENS = 8
CORRECTION_FACTOR_MIN = 0.6
CORRECTION_FACTOR_MAX = 1.8
EMA_ALPHA = 0.25

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


@dataclass(frozen=True)
class TokenEstimate:
    total: int
    by_type: dict[str, int] = field(default_factory=dict)


def estimate_text_tokens(
    text: str,
    *,
    content_type: ContentType | None = None,
    correction_factors: dict[str, float] | None = None,
) -> TokenEstimate:
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
    return estimate_message(msg, correction_factors=correction_factors).total


def estimate_message(
    msg: Message,
    *,
    correction_factors: dict[str, float] | None = None,
) -> TokenEstimate:
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
    text = f"{name} {json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True)}"
    raw = TOOL_CALL_OVERHEAD_TOKENS + _estimate_by_kind(text, "json_schema")
    total = _apply_factor(raw, "tool_call_struct", correction_factors)
    return TokenEstimate(total=total, by_type={"tool_call_struct": total})


def estimate_tools_tokens(
    tools: list[Tool] | None,
    *,
    correction_factors: dict[str, float] | None = None,
) -> TokenEstimate:
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
    if not text:
        return "english_text"
    cjk_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
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


class ContextUsageCalibrator:
    """Persist provider/model/content-type correction factors."""

    def __init__(self, workspace_dir: str | Path) -> None:
        self.path = Path(workspace_dir) / ".codepilot" / "context_usage.json"

    def factors_for(self, provider: str | None, model: str | None) -> dict[str, float]:
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
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}


def is_context_overflow(
    model: Model,
    context: Context,
    *,
    safety_margin: float = 0.95,
) -> bool:
    limit = int(model.context_window * safety_margin)
    estimated = estimate_context_tokens(
        context.messages,
        context.system_prompt or "",
        context.tools,
    )
    return estimated > limit


def overflow_ratio(model: Model, context: Context) -> float:
    estimated = estimate_context_tokens(
        context.messages,
        context.system_prompt or "",
        context.tools,
    )
    if model.context_window <= 0:
        return 0.0
    return estimated / model.context_window


def _estimate_by_kind(text: str, kind: ContentType) -> int:
    if kind == "image":
        return IMAGE_TOKEN_ESTIMATE
    denominator = _CONTENT_DENOMINATORS[kind]
    return max(1, math.ceil(len(text) / denominator)) if text else 0


def _merge_block_estimate(
    by_type: dict[str, int],
    block: TextContent | ImageContent,
    correction_factors: dict[str, float] | None,
) -> None:
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
    raw = IMAGE_TOKEN_ESTIMATE
    if block.mime_type.endswith("jpeg") or block.mime_type.endswith("jpg"):
        raw = 850
    total = _apply_factor(raw, "image", correction_factors)
    by_type["image"] = by_type.get("image", 0) + total


def _merge_estimate(by_type: dict[str, int], estimate: TokenEstimate) -> None:
    for key, value in estimate.by_type.items():
        by_type[key] = by_type.get(key, 0) + value


def _apply_factor(
    raw_tokens: int,
    content_type: str,
    correction_factors: dict[str, float] | None,
) -> int:
    factor = _factor_for(content_type, correction_factors)
    return max(0, math.ceil(raw_tokens * factor))


def _factor_for(
    content_type: str,
    correction_factors: dict[str, float] | None,
) -> float:
    if not correction_factors:
        return 1.0
    factor = correction_factors.get(content_type)
    if not isinstance(factor, int | float):
        return 1.0
    return _clamp_factor(float(factor))


def _clamp_factor(value: float) -> float:
    return min(CORRECTION_FACTOR_MAX, max(CORRECTION_FACTOR_MIN, value))


def _calibration_prefix(provider: str | None, model: str | None) -> str:
    return f"{provider or 'unknown'}:{model or 'unknown'}"


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
        return True
    return bool(re.search(r'"[^"]+"\s*:', text))


def _looks_like_code(text: str) -> bool:
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
