"""DingTalk 界面包：转发消息、命令、审批和审计事件。"""

from __future__ import annotations

# 新手导读：interfaces.dingtalk 是手机远程控制入口，只暴露桥接层契约。
# 关注点：它是独立远程入口，不会影响 interfaces.cli 的任何运行模式。

"""DingTalk interface boundary for Codepilot."""

from .bridge import DingTalkBridge
from .commands import help_text, parse_dingtalk_command
from .schemas import (
    DingTalkBridgeConfig,
    DingTalkCommand,
    DingTalkInboundMessage,
    DingTalkOutboundMessage,
    describe_dingtalk_contract,
)
from .transport import (
    DingTalkStreamTransport,
    DingTalkTransport,
    create_stream_transport,
)

__all__ = [
    "DingTalkBridge",
    "DingTalkBridgeConfig",
    "DingTalkCommand",
    "DingTalkInboundMessage",
    "DingTalkOutboundMessage",
    "DingTalkStreamTransport",
    "DingTalkTransport",
    "create_stream_transport",
    "describe_dingtalk_contract",
    "help_text",
    "parse_dingtalk_command",
]
