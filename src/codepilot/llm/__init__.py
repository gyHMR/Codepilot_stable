"""
Codepilot LLM 公共导出模块。

本模块是 LLM 子包的统一入口，集中导出：
- 类型定义（Model、Message、Tool 等协议类型）
- API 注册中心（register/get/clear provider）
- 流式调用函数（stream、complete）
- 上下文溢出检测工具
- 环境变量 API Key 读取
"""

from .api_registry import (
    ApiProvider,
    LLMProvider,
    clear_api_providers,
    complete,
    complete_simple,
    get_api_provider,
    register_api_provider,
    stream,
    stream_simple,
)
from .env_api_keys import get_env_api_key, get_env_api_key_name
from .errors import LLMErrorInfo, LLMErrorKind
from .event_stream import AssistantMessageEventStream, LLMStreamEvent, LLMStreamEventType
from .models import get_model, get_models, get_providers
from .overflow import estimate_context_tokens, estimate_message_tokens, is_context_overflow, overflow_ratio
from codepilot.protocols import (
    Api,
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
    注册内置的 API provider（延迟导入）。

    延迟导入是为了避免在仅使用类型定义时加载 HTTP 依赖，
    使得纯类型导入和小型单元测试不会因缺少可选依赖而失败。
    """
    from .providers.register_builtins import register_builtin_api_providers as _register

    _register()


def reset_api_providers() -> None:
    """重置 provider 注册（通过延迟导入的 provider 模块执行）。"""
    from .providers.register_builtins import reset_api_providers as _reset

    _reset()


__all__ = [
    # ── 类型定义 ──
    "ApiProvider",
    "Api",
    "AssistantMessage",
    "AssistantMessageEventStream",
    "Context",
    "Cost",
    "ImageContent",
    "LLMErrorInfo",
    "LLMErrorKind",
    "LLMProvider",
    "LLMStreamEvent",
    "LLMStreamEventType",
    "Message",
    "Model",
    "ModelCapabilities",
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
    # ── 函数 ──
    "clear_api_providers",
    "complete",
    "complete_simple",
    "estimate_context_tokens",
    "estimate_message_tokens",
    "is_context_overflow",
    "overflow_ratio",
    "get_api_provider",
    "get_env_api_key",
    "get_env_api_key_name",
    "get_model",
    "get_models",
    "get_providers",
    "register_api_provider",
    "register_builtin_api_providers",
    "reset_api_providers",
    "stream",
    "stream_simple",
]

# 模块加载时自动注册内置 provider。
# 使用 try/except 包裹，确保可选依赖缺失时不会阻断整个包的导入。
try:
    register_builtin_api_providers()
except ModuleNotFoundError:
    pass
