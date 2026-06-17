from __future__ import annotations

"""
Compatibility exports for LLM-facing protocol types.

The concrete cross-layer data structures now live in `codepilot.protocols`.
This module remains the stable import point for existing LLM/core/session code
while later layers migrate gradually.
"""

from codepilot.protocols import (
    Api,
    AssistantBlock,
    AssistantMessage,
    Context,
    Cost,
    ImageContent,
    Message,
    Model,
    ModelCapabilities,
    Provider,
    SimpleStreamOptions,
    StopReason,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ThinkingLevel,
    Tool,
    ToolCall,
    ToolResultBlock,
    ToolResultMessage,
    Usage,
    UserBlock,
    UserMessage,
)


__all__ = [
    "Api",
    "AssistantBlock",
    "AssistantMessage",
    "Context",
    "Cost",
    "ImageContent",
    "Message",
    "Model",
    "ModelCapabilities",
    "Provider",
    "SimpleStreamOptions",
    "StopReason",
    "StreamOptions",
    "TextContent",
    "ThinkingContent",
    "ThinkingLevel",
    "Tool",
    "ToolCall",
    "ToolResultBlock",
    "ToolResultMessage",
    "Usage",
    "UserBlock",
    "UserMessage",
]
