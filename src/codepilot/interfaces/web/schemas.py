from __future__ import annotations

"""Web Console 的数据模式定义（请求、响应、事件信封等）。"""

from dataclasses import dataclass, field
from typing import Any, Literal

# 审批决策类型：批准 / 拒绝
ApprovalDecision = Literal["approve", "deny"]
# Web 事件种类
WebEventKind = Literal[
    "agent_event",           # Agent 事件（流式输出、工具执行等）
    "session_created",       # 会话创建
    "session_state",         # 会话状态变更
    "tool_approval_required", # 工具审批请求
    "error",                 # 错误
]


@dataclass(frozen=True)
class WebSessionRef:
    """会话引用：标识一个工作区目录和可选的会话 ID。"""
    workspace_dir: str
    session_id: str | None = None


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


@dataclass(frozen=True)
class WebSessionSummary:
    """会话摘要：仅包含会话 ID。"""
    session_id: str


@dataclass(frozen=True)
class WebPromptRequest:
    """用户消息请求：包含会话引用、文本和可选图片。"""
    session: WebSessionRef
    text: str
    images: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WebToolApproval:
    """工具审批请求：包含会话 ID、工具调用 ID、决策和原因。"""
    session_id: str
    tool_call_id: str
    decision: ApprovalDecision
    reason: str = ""


@dataclass(frozen=True)
class WebEventEnvelope:
    """Web 事件信封：统一的事件传输格式，包含类型、会话 ID 和载荷。"""
    type: WebEventKind
    session_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WebRouteSpec:
    """路由规格：描述一个 API 端点的 HTTP 方法、路径和说明。"""
    method: str
    path: str
    description: str
