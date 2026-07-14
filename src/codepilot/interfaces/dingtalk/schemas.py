"""定义 DingTalk 入站事件和出站响应的数据结构。"""

from __future__ import annotations

# 新手导读：schemas.py 定义钉钉远程入口和 RuntimeGateway 之间的稳定数据契约。
# 关注点：钉钉层只描述消息、配置和回复，不直接表达工具执行细节。

"""DingTalk interface data contracts."""

from dataclasses import dataclass
from typing import Literal, cast


DingTalkOutboundFormat = Literal["text", "markdown"]
DingTalkCommandAction = Literal[
    "prompt",
    "approve",
    "deny",
    "status",
    "cancel",
    "help",
    "unknown",
]
_COMMAND_ACTIONS = frozenset(
    {"prompt", "approve", "deny", "status", "cancel", "help", "unknown"}
)
_OUTBOUND_FORMATS = frozenset({"text", "markdown"})


@dataclass(frozen=True)
class DingTalkBridgeConfig:
    """Configuration for one local DingTalk bridge process."""

    workspace_dir: str
    allowed_users: tuple[str, ...]
    session_id: str | None = None
    provider: str | None = None
    model_id: str | None = None
    allow_dirty: bool = False
    verbose_events: bool = False
    load_workspace_resources: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "workspace_dir",
            _required_text(self.workspace_dir, field_name="workspace_dir"),
        )
        allowed = _normalize_texts(self.allowed_users, field_name="allowed_users")
        if not allowed:
            raise ValueError("DingTalk allowed_users cannot be empty")
        object.__setattr__(self, "allowed_users", allowed)
        object.__setattr__(
            self,
            "session_id",
            _optional_text(self.session_id, field_name="session_id"),
        )
        object.__setattr__(
            self,
            "provider",
            _optional_text(self.provider, field_name="provider"),
        )
        object.__setattr__(
            self,
            "model_id",
            _optional_text(self.model_id, field_name="model_id"),
        )
        object.__setattr__(self, "allow_dirty", _required_bool(self.allow_dirty, field_name="allow_dirty"))
        object.__setattr__(
            self,
            "verbose_events",
            _required_bool(self.verbose_events, field_name="verbose_events"),
        )
        object.__setattr__(
            self,
            "load_workspace_resources",
            _required_bool(
                self.load_workspace_resources,
                field_name="load_workspace_resources",
            ),
        )


@dataclass(frozen=True)
class DingTalkInboundMessage:
    """Normalized inbound DingTalk chat message."""

    message_id: str
    sender_id: str
    text: str
    conversation_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "message_id",
            _required_text(self.message_id, field_name="message_id"),
        )
        object.__setattr__(
            self,
            "sender_id",
            _required_text(self.sender_id, field_name="sender_id"),
        )
        object.__setattr__(self, "text", _required_text(self.text, field_name="text"))
        object.__setattr__(
            self,
            "conversation_id",
            _optional_text(self.conversation_id, field_name="conversation_id"),
        )


@dataclass(frozen=True)
class DingTalkOutboundMessage:
    """Normalized outbound DingTalk message."""

    receiver_id: str
    text: str
    conversation_id: str | None = None
    format: DingTalkOutboundFormat = "text"
    title: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "receiver_id",
            _required_text(self.receiver_id, field_name="receiver_id"),
        )
        object.__setattr__(self, "text", _required_text(self.text, field_name="text"))
        object.__setattr__(
            self,
            "conversation_id",
            _optional_text(self.conversation_id, field_name="conversation_id"),
        )
        object.__setattr__(self, "format", _ensure_outbound_format(self.format))
        object.__setattr__(
            self,
            "title",
            _optional_text(self.title, field_name="title"),
        )


@dataclass(frozen=True)
class DingTalkCommand:
    """Parsed remote command from a DingTalk text message."""

    action: DingTalkCommandAction
    raw_text: str
    prompt: str = ""
    approval_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _ensure_command_action(self.action))
        object.__setattr__(
            self,
            "raw_text",
            _required_text(self.raw_text, field_name="raw_text"),
        )
        object.__setattr__(self, "prompt", _text_or_empty(self.prompt))
        object.__setattr__(self, "approval_id", _text_or_empty(self.approval_id))


def describe_dingtalk_contract() -> dict[str, object]:
    """Return the stable public contract for the DingTalk interface."""

    return {
        "transport": ["dingtalk-stream"],
        "entrypoint": "codepilot.interfaces.dingtalk",
        "delegates_to": [
            "codepilot.runtime",
            "codepilot.sessions",
            "codepilot.tools",
        ],
        "commands": ["cp", "approve", "deny", "status", "cancel", "help"],
        "responsibilities": [
            "receive_remote_prompt",
            "display_run_summary",
            "display_tool_approval",
            "submit_tool_approval",
        ],
        "non_responsibilities": [
            "llm_provider_calls",
            "agent_loop",
            "filesystem_mutation",
            "shell_execution",
            "session_persistence",
            "desktop_client_automation",
        ],
    }


def _ensure_command_action(value: object) -> DingTalkCommandAction:
    if value not in _COMMAND_ACTIONS:
        raise ValueError(f"Unknown DingTalk command action: {value}")
    return cast(DingTalkCommandAction, value)


def _ensure_outbound_format(value: object) -> DingTalkOutboundFormat:
    if value not in _OUTBOUND_FORMATS:
        raise ValueError(f"Unknown DingTalk outbound format: {value}")
    return cast(DingTalkOutboundFormat, value)


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"DingTalk {field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"DingTalk {field_name} cannot be empty")
    return text


def _optional_text(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"DingTalk {field_name} must be a string")
    text = value.strip()
    return text or None


def _text_or_empty(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError("DingTalk command text fields must be strings")
    return value.strip()


def _required_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"DingTalk {field_name} must be a boolean")
    return value


def _normalize_texts(value: object, *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"DingTalk {field_name} must be a sequence of strings")
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _required_text(item, field_name=field_name)
        if text not in seen:
            cleaned.append(text)
            seen.add(text)
    return tuple(cleaned)


__all__ = [
    "DingTalkBridgeConfig",
    "DingTalkCommand",
    "DingTalkCommandAction",
    "DingTalkInboundMessage",
    "DingTalkOutboundMessage",
    "DingTalkOutboundFormat",
    "describe_dingtalk_contract",
]
