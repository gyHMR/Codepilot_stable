from __future__ import annotations

"""
内容块（Content Block）类型定义。

定义了消息和工具结果中可包含的三种内容块：
- TextContent: 纯文本内容
- ThinkingContent: 模型推理/思考过程
- ImageContent: Base64 编码的图片

ContentBlock 是这三种类型的联合类型。
"""

from dataclasses import dataclass
from typing import Literal, Union


@dataclass
class TextContent:
    """纯文本内容块。

    被消息和工具结果共用，是最基础的内容单元。

    Attributes:
        type: 内容类型标识，固定为 "text"。
        text: 文本内容。
        text_signature: 可选的文本签名（用于验证内容完整性）。
    """

    type: Literal["text"] = "text"
    text: str = ""
    text_signature: str | None = None


@dataclass
class ThinkingContent:
    """推理/思考内容块。

    由支持推理能力的模型（如 Claude 的 extended thinking）生成，
    记录模型在给出最终回答前的思考过程。

    Attributes:
        type: 内容类型标识，固定为 "thinking"。
        thinking: 思考过程的文本内容。
        thinking_signature: 可选的思考签名（用于验证完整性）。
        redacted: 是否被编辑/脱敏（部分模型可能对敏感思考内容进行脱敏处理）。
    """

    type: Literal["thinking"] = "thinking"
    thinking: str = ""
    thinking_signature: str | None = None
    redacted: bool = False


@dataclass
class ImageContent:
    """Base64 编码的图片内容块。

    Attributes:
        type: 内容类型标识，固定为 "image"。
        data: Base64 编码的图片数据。
        mime_type: 图片 MIME 类型，默认为 "image/png"。
        source: 可选的图片来源标识。
    """

    type: Literal["image"] = "image"
    data: str = ""
    mime_type: str = "image/png"
    source: str | None = None


# 内容块联合类型：可以是文本、思考或图片中的任意一种
ContentBlock = Union[TextContent, ThinkingContent, ImageContent]


__all__ = [
    "ContentBlock",
    "ImageContent",
    "TextContent",
    "ThinkingContent",
]
