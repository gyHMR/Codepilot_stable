"""定义 Web API 请求、响应和错误载荷 schema。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ApiErrorDetail(BaseModel):
    """Web API 返回的结构化错误详情。"""
    code: str
    message: str
    details: Any | None = None


class ApiError(BaseModel):
    """统一 Web API 错误响应。"""
    error: ApiErrorDetail


class SessionCreate(BaseModel):
    """创建 Session 的请求载荷。"""
    title: str | None = Field(default=None, min_length=1, max_length=120)


class SessionUpdate(BaseModel):
    """更新 Web Session 展示属性。"""

    title: str = Field(min_length=1, max_length=120)


class MessageCreate(BaseModel):
    """向 Session 提交用户消息的请求载荷。"""
    text: str = Field(min_length=1)
    mode_hint: str | None = None


class CommandCreate(BaseModel):
    """向 Session 提交命令的请求载荷。"""
    text: str = Field(min_length=1)


class ApprovalCreate(BaseModel):
    """提交审批决定的请求载荷。"""
    decision: Literal["approve", "deny"]
    reason: str = ""


class InteractionCreate(BaseModel):
    """Answer one persisted user-input interaction."""

    answer: str = Field(min_length=1)


class AcceptedAction(BaseModel):
    """异步动作已被 Runtime 接受的响应。"""
    session_id: str
    accepted: bool = True
    run_id: str | None = None


__all__ = [
    "AcceptedAction",
    "ApiError",
    "ApiErrorDetail",
    "ApprovalCreate",
    "InteractionCreate",
    "CommandCreate",
    "MessageCreate",
    "SessionCreate",
    "SessionUpdate",
]
