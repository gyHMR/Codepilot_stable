"""
Codepilot IM integration layer.

- webhook parsing
- channel-level session routing
- streaming message updates
- runtime-backed Agent calls
- text/card replies
- MEMORY.md support
"""

from codepilot.sessions.memory import (
    load_channel_memory,
    load_global_memory,
    load_merged_memory,
    save_channel_memory,
    save_global_memory,
)

from .events import IMEventWatcher, IMEventWatcherOptions
from .service import IMService, IMServiceConfig
from .session_router import SessionRouter
from .types import (
    IMAdapter,
    IMChannelInfo,
    IMIncomingMessage,
    IMOutgoingCard,
    IMOutgoingText,
    IMUserInfo,
    IMWebhookResult,
)

__all__ = [
    "IMAdapter",
    "IMChannelInfo",
    "IMIncomingMessage",
    "IMOutgoingCard",
    "IMOutgoingText",
    "IMUserInfo",
    "IMWebhookResult",
    "SessionRouter",
    "IMService",
    "IMServiceConfig",
    "IMEventWatcher",
    "IMEventWatcherOptions",
    "load_global_memory",
    "load_channel_memory",
    "load_merged_memory",
    "save_global_memory",
    "save_channel_memory",
]
