from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union


@dataclass
class TextContent:
    """Plain text content block shared by messages and tool results."""

    type: Literal["text"] = "text"
    text: str = ""
    text_signature: str | None = None


@dataclass
class ThinkingContent:
    """Reasoning/thinking content block emitted by capable models."""

    type: Literal["thinking"] = "thinking"
    thinking: str = ""
    thinking_signature: str | None = None
    redacted: bool = False


@dataclass
class ImageContent:
    """Base64 image content block."""

    type: Literal["image"] = "image"
    data: str = ""
    mime_type: str = "image/png"
    source: str | None = None


ContentBlock = Union[TextContent, ThinkingContent, ImageContent]


__all__ = [
    "ContentBlock",
    "ImageContent",
    "TextContent",
    "ThinkingContent",
]
