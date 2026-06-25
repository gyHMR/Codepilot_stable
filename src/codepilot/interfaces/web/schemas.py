from __future__ import annotations

"""Web Console 的数据模式定义（请求、响应、事件信封等）。"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, cast

# Web 传输层的审批动作：只表达用户点击了批准还是拒绝。
# tools.ApprovalDecision 是工具层执行结果对象，语义不同，不要混用。
ApprovalDecision = Literal["approve", "deny"]
# Web 事件信封种类：前端据此选择展示、状态刷新或用户操作流程。
WebEventKind = Literal[
    "agent_event",           # Agent 事件（流式输出、工具执行等）
    "session_created",       # 会话创建
    "session_state",         # 会话状态变更
    "tool_approval_required", # 工具审批请求
    "error",                 # 错误
]
_APPROVAL_DECISIONS = frozenset({"approve", "deny"})
_WEB_EVENT_KINDS = frozenset(
    {
        "agent_event",
        "session_created",
        "session_state",
        "tool_approval_required",
        "error",
    }
)
_WEB_ROUTE_METHODS = frozenset({"GET", "POST", "WS"})


@dataclass(frozen=True)
class WebSessionRef:
    """会话引用：标识一个工作区目录和可选的会话 ID。"""
    workspace_dir: str
    session_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "workspace_dir",
            _ensure_non_empty_text(self.workspace_dir, field_name="workspace"),
        )
        object.__setattr__(
            self,
            "session_id",
            _optional_text(self.session_id, field_name="session_id"),
        )


@dataclass(frozen=True)
class WebCreateSessionRequest:
    """创建会话请求：包含工作区、模型、提示词等配置。"""
    workspace_dir: str
    provider: str | None = None
    model_id: str | None = None
    system_prompt: str | None = None
    session_id: str | None = None
    read_only_mode: bool | None = None
    load_workspace_resources: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "workspace_dir",
            _ensure_non_empty_text(self.workspace_dir, field_name="workspace"),
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
        object.__setattr__(
            self,
            "system_prompt",
            _optional_text(self.system_prompt, field_name="system_prompt"),
        )
        object.__setattr__(
            self,
            "session_id",
            _optional_text(self.session_id, field_name="session_id"),
        )
        object.__setattr__(
            self,
            "read_only_mode",
            _optional_bool(self.read_only_mode, field_name="read_only_mode"),
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
class WebSessionSummary:
    """会话摘要：仅包含会话 ID。"""
    session_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "session_id",
            _ensure_non_empty_text(self.session_id, field_name="session_id"),
        )


@dataclass(frozen=True)
class WebPromptRequest:
    """用户消息请求：包含会话引用、文本和可选图片。"""
    session: WebSessionRef
    text: str
    images: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.session, WebSessionRef):
            raise TypeError("WebPromptRequest.session must be WebSessionRef")
        object.__setattr__(
            self,
            "text",
            _ensure_non_empty_text(self.text, field_name="prompt text"),
        )
        object.__setattr__(
            self,
            "images",
            _normalize_text_sequence(
                self.images,
                field_name="image paths",
                item_name="image path",
            ),
        )


@dataclass(frozen=True)
class WebToolApproval:
    """工具审批请求：包含会话 ID、审批 ID、工具调用 ID、决策和原因。"""
    session_id: str
    tool_call_id: str
    decision: ApprovalDecision
    reason: str = ""
    approval_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "session_id",
            _ensure_non_empty_text(self.session_id, field_name="session_id"),
        )
        object.__setattr__(
            self,
            "tool_call_id",
            _ensure_non_empty_text(self.tool_call_id, field_name="tool_call_id"),
        )
        object.__setattr__(
            self,
            "decision",
            _ensure_approval_decision(self.decision),
        )
        object.__setattr__(
            self,
            "reason",
            _text_or_empty(self.reason, field_name="reason"),
        )
        object.__setattr__(
            self,
            "approval_id",
            _optional_text(self.approval_id, field_name="approval_id"),
        )


@dataclass(frozen=True)
class WebEventEnvelope:
    """Web 事件信封：统一的事件传输格式，包含类型、会话 ID 和载荷。"""
    type: WebEventKind
    session_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "type",
            _ensure_web_event_kind(self.type),
        )
        object.__setattr__(
            self,
            "session_id",
            _optional_text(self.session_id, field_name="session_id"),
        )
        object.__setattr__(
            self,
            "payload",
            _copy_payload(self.payload),
        )


@dataclass(frozen=True)
class WebErrorPayload:
    """Web 错误事件载荷：保证前端总能分类并展示错误。"""

    code: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "code",
            _ensure_non_empty_text(self.code, field_name="error code"),
        )
        object.__setattr__(
            self,
            "message",
            _ensure_non_empty_text(self.message, field_name="error message"),
        )


@dataclass(frozen=True)
class WebRouteSpec:
    """路由规格：描述一个 API 端点的 HTTP 方法、路径和说明。"""
    method: str
    path: str
    description: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "method",
            _ensure_route_method(self.method),
        )
        object.__setattr__(
            self,
            "path",
            _ensure_route_path(self.path),
        )
        object.__setattr__(
            self,
            "description",
            _ensure_non_empty_text(self.description, field_name="description"),
        )

    def to_dict(self) -> dict[str, str]:
        """返回稳定的公开契约字段，不暴露 dataclass 内部表示。"""

        return {
            "method": self.method,
            "path": self.path,
            "description": self.description,
        }


def _ensure_approval_decision(value: object) -> ApprovalDecision:
    if value not in _APPROVAL_DECISIONS:
        raise ValueError(f"Unknown web approval decision: {value}")
    return cast(ApprovalDecision, value)


def _ensure_web_event_kind(value: object) -> WebEventKind:
    if value not in _WEB_EVENT_KINDS:
        raise ValueError(f"Unknown web event type: {value}")
    return cast(WebEventKind, value)


def _ensure_route_method(value: object) -> str:
    text = _ensure_non_empty_text(value, field_name="method").upper()
    if text not in _WEB_ROUTE_METHODS:
        raise ValueError(f"Unknown web route method: {value}")
    return text


def _ensure_route_path(value: object) -> str:
    text = _ensure_non_empty_text(value, field_name="path")
    if not text.startswith("/"):
        raise ValueError("Web path must start with /")
    return text


def _ensure_non_empty_text(value: object, *, field_name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"Web {field_name} cannot be empty")
    return text


def _optional_text(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"Web {field_name} must be a string")
    text = value.strip()
    return text or None


def _optional_bool(value: object, *, field_name: str) -> bool | None:
    if value is None:
        return None
    return _required_bool(value, field_name=field_name)


def _required_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"Web {field_name} must be a boolean")
    return value


def _text_or_empty(value: object, *, field_name: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"Web {field_name} must be a string")
    return value.strip()


def _copy_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("Web payload must be a dict")
    return deepcopy(value)


def _normalize_text_sequence(
    value: object,
    *,
    field_name: str,
    item_name: str,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"Web {field_name} must be a sequence of strings")
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"Web {field_name} must contain strings")
        text = item.strip()
        if not text:
            raise ValueError(f"Web {item_name} cannot be empty")
        cleaned.append(text)
    return tuple(cleaned)
