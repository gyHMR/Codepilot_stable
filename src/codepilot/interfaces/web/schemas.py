from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ApiErrorDetail(BaseModel):
    code: str
    message: str
    details: Any | None = None


class ApiError(BaseModel):
    error: ApiErrorDetail


class SessionCreate(BaseModel):
    title: str | None = None


class MessageCreate(BaseModel):
    text: str = Field(min_length=1)
    mode_hint: str | None = None


class CommandCreate(BaseModel):
    text: str = Field(min_length=1)


class ApprovalCreate(BaseModel):
    decision: Literal["approve", "deny"]
    reason: str = ""


class AcceptedAction(BaseModel):
    session_id: str
    accepted: bool = True
    run_id: str | None = None


__all__ = [
    "AcceptedAction",
    "ApiError",
    "ApiErrorDetail",
    "ApprovalCreate",
    "CommandCreate",
    "MessageCreate",
    "SessionCreate",
]
