"""封装 DingTalk 事件接收、回复和长连接传输。"""

from __future__ import annotations

# 新手导读：transport.py 隔离钉钉 SDK 细节，测试可使用 fake transport 而不安装 SDK。
# 关注点：传输层只负责收发消息，不直接创建 session 或执行工具。

"""DingTalk Stream transport adapter."""

import asyncio
import importlib
import inspect
from typing import Any, Callable, Protocol, TYPE_CHECKING

from .schemas import DingTalkInboundMessage, DingTalkOutboundMessage

if TYPE_CHECKING:
    from .bridge import DingTalkBridge


class DingTalkTransport(Protocol):
    """Transport contract implemented by DingTalk Stream and tests."""

    async def start(self, bridge: "DingTalkBridge") -> None:
        """Start receiving DingTalk messages and forwarding them to the bridge."""


class DingTalkStreamTransport:
    """Adapter around the optional ``dingtalk-stream`` SDK."""

    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        sdk: Any,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self._sdk = sdk

    async def start(self, bridge: "DingTalkBridge") -> None:
        credential = self._sdk.Credential(self.client_id, self.client_secret)
        client = self._sdk.DingTalkStreamClient(credential)
        handler = _build_chatbot_handler(self._sdk, bridge)
        topic = getattr(self._sdk.ChatbotMessage, "TOPIC", "/v1.0/im/bot/messages/get")
        client.register_callback_handler(topic, handler)
        starter = getattr(client, "start", None)
        if starter is not None and inspect.iscoroutinefunction(starter):
            await starter()
            return
        if starter is not None:
            await asyncio.to_thread(starter)
            return
        starter = getattr(client, "start_forever")
        await asyncio.to_thread(starter)


def create_stream_transport(
    *,
    client_id: str,
    client_secret: str,
    importer: Callable[[str], Any] = importlib.import_module,
) -> DingTalkStreamTransport:
    """Create the DingTalk Stream transport or raise an install hint."""

    try:
        sdk = importer("dingtalk_stream")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'DingTalk Stream SDK is not installed. Install it with `pip install "codepilot[dingtalk]"`.'
        ) from exc
    return DingTalkStreamTransport(
        client_id=client_id,
        client_secret=client_secret,
        sdk=sdk,
    )


def _build_chatbot_handler(sdk: Any, bridge: "DingTalkBridge") -> Any:
    base_handler = sdk.ChatbotHandler
    chatbot_message = sdk.ChatbotMessage
    ack_message = getattr(sdk, "AckMessage", None)

    class CodepilotChatbotHandler(base_handler):  # type: ignore[misc, valid-type]
        async def process(self, callback: Any) -> Any:
            try:
                incoming = chatbot_message.from_dict(callback.data)
                _message_to_inbound(incoming)
            except Exception as exc:
                self.logger.error("invalid DingTalk chatbot message: %s", exc)
                if ack_message is None:
                    return None
                return ack_message.STATUS_OK, "OK"
            asyncio.create_task(_reply_later(self, bridge, incoming))
            if ack_message is None:
                return None
            return ack_message.STATUS_OK, "OK"

    return CodepilotChatbotHandler()


async def _reply_later(handler: Any, bridge: "DingTalkBridge", incoming: Any) -> None:
    try:
        inbound = _message_to_inbound(incoming)
        stream_replies = getattr(bridge, "iter_replies", None)
        if callable(stream_replies):
            async for reply in stream_replies(inbound):
                await _send_reply(handler, reply, incoming)
            return
        replies = await bridge.handle_message(inbound)
        for reply in replies:
            await _send_reply(handler, reply, incoming)
    except Exception as exc:  # pragma: no cover - SDK background guard
        _log_handler_error(handler, "DingTalk reply task failed: %s", exc)


async def _send_reply(
    handler: Any,
    reply: DingTalkOutboundMessage,
    incoming: Any,
) -> None:
    if reply.format == "markdown":
        markdown_sender = getattr(handler, "reply_markdown", None)
        if callable(markdown_sender):
            await asyncio.to_thread(
                markdown_sender,
                reply.title or "Codepilot",
                reply.text,
                incoming,
            )
            return
    await asyncio.to_thread(handler.reply_text, reply.text, incoming)


def _log_handler_error(handler: Any, message: str, *args: object) -> None:
    logger = getattr(handler, "logger", None)
    if logger is not None:
        logger.error(message, *args)


def _message_to_inbound(message: Any) -> DingTalkInboundMessage:
    text_obj = _field(message, "text")
    text = _field(text_obj, "content") or _field(message, "content") or ""
    return DingTalkInboundMessage(
        message_id=(
            _field(message, "message_id")
            or _field(message, "msg_id")
            or _field(message, "msgId")
        ),
        sender_id=(
            _field(message, "sender_staff_id")
            or _field(message, "senderStaffId")
            or _field(message, "sender_id")
            or _field(message, "senderId")
        ),
        conversation_id=(
            _field(message, "conversation_id")
            or _field(message, "conversationId")
        ),
        text=text,
    )


def _field(value: Any, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


__all__ = [
    "DingTalkStreamTransport",
    "DingTalkTransport",
    "create_stream_transport",
]
