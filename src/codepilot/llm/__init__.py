"""
Codepilot LLM public exports.
"""

from .api_registry import ApiProvider, clear_api_providers, get_api_provider, register_api_provider
from .env_api_keys import get_env_api_key
from .event_stream import AssistantMessageEventStream
from .models import get_model, get_models, get_providers
from .overflow import estimate_context_tokens, estimate_message_tokens, is_context_overflow, overflow_ratio
from .stream import complete, complete_simple, stream, stream_simple
from .types import (
    Api,
    AssistantMessage,
    Context,
    Cost,
    ImageContent,
    Message,
    Model,
    Provider,
    SimpleStreamOptions,
    StopReason,
    StreamOptions,
    TextContent,
    ThinkingLevel,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def register_builtin_api_providers() -> None:
    """
    Lazily import providers to avoid loading optional HTTP dependencies when
    callers only import the LLM package for types.
    """

    from .providers.register_builtins import register_builtin_api_providers as _register

    _register()


def reset_api_providers() -> None:
    """Reset provider registration through the lazily imported provider module."""

    from .providers.register_builtins import reset_api_providers as _reset

    _reset()


__all__ = [
    "ApiProvider",
    "Api",
    "AssistantMessage",
    "AssistantMessageEventStream",
    "Context",
    "Cost",
    "ImageContent",
    "Message",
    "Model",
    "Provider",
    "SimpleStreamOptions",
    "StopReason",
    "StreamOptions",
    "TextContent",
    "ThinkingLevel",
    "ThinkingContent",
    "Tool",
    "ToolCall",
    "ToolResultMessage",
    "Usage",
    "UserMessage",
    "clear_api_providers",
    "complete",
    "complete_simple",
    "estimate_context_tokens",
    "estimate_message_tokens",
    "is_context_overflow",
    "overflow_ratio",
    "get_api_provider",
    "get_env_api_key",
    "get_model",
    "get_models",
    "get_providers",
    "register_api_provider",
    "register_builtin_api_providers",
    "reset_api_providers",
    "stream",
    "stream_simple",
]

# Keep `import codepilot.llm` lightweight. If optional provider dependencies
# are unavailable, type-only imports and small unit tests should still work.
try:
    register_builtin_api_providers()
except ModuleNotFoundError:
    pass
