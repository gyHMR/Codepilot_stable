from __future__ import annotations

"""Canonical tool execution results and conversation-message projection."""

from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping, TypeAlias, cast

from codepilot.protocols import (
    ImageContent as ConversationImageContent,
    TextContent as ConversationTextContent,
    ToolResultMessage,
)
from codepilot.protocols.tools import ToolResultStatus as ConversationToolResultStatus

from .security import ApprovalChallenge, ToolEffect


ToolStatus: TypeAlias = Literal[
    "success",
    "error",
    "denied",
    "approval_required",
    "user_input_required",
    "cancelled",
    "timed_out",
    "interrupted",
]
ToolErrorKind: TypeAlias = Literal[
    "registration",
    "validation",
    "unavailable",
    "permission",
    "approval",
    "interaction",
    "queue_timeout",
    "execution_timeout",
    "cancelled",
    "interrupted",
    "execution",
    "output_validation",
    "policy_violation",
    "resource_cleanup",
    "stale_registration",
    "internal",
]
OutputValidation: TypeAlias = Literal["schema_validated", "structurally_validated"]
ContentTrust: TypeAlias = Literal["trusted", "untrusted"]

_TOOL_STATUSES = frozenset(
    {
        "success",
        "error",
        "denied",
        "approval_required",
        "user_input_required",
        "cancelled",
        "timed_out",
        "interrupted",
    }
)
_TOOL_ERROR_KINDS = frozenset(
    {
        "registration",
        "validation",
        "unavailable",
        "permission",
        "approval",
        "interaction",
        "queue_timeout",
        "execution_timeout",
        "cancelled",
        "interrupted",
        "execution",
        "output_validation",
        "policy_violation",
        "resource_cleanup",
        "stale_registration",
        "internal",
    }
)
_OUTPUT_VALIDATION = frozenset({"schema_validated", "structurally_validated"})
_CONTENT_TRUST = frozenset({"trusted", "untrusted"})


@dataclass(frozen=True)
class TextContent:
    text: str
    type: Literal["text"] = "text"

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text content must be str")


@dataclass(frozen=True)
class ImageContent:
    data: str
    mime_type: str
    name: str | None = None
    type: Literal["image"] = "image"

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", _require_text(self.data, "image data"))
        object.__setattr__(self, "mime_type", _require_text(self.mime_type, "image mime_type"))
        object.__setattr__(self, "name", _optional_text(self.name))


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    media_type: str = "application/octet-stream"
    name: str | None = None
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _require_text(self.artifact_id, "artifact_id"))
        object.__setattr__(self, "media_type", _require_text(self.media_type, "artifact media_type"))
        object.__setattr__(self, "name", _optional_text(self.name))
        if self.size_bytes is not None:
            _require_non_negative_int(self.size_bytes, "artifact size_bytes")


@dataclass(frozen=True)
class ArtifactContent:
    artifact: ArtifactRef
    type: Literal["artifact"] = "artifact"

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, ArtifactRef):
            raise TypeError("artifact content must reference ArtifactRef")


ToolContent: TypeAlias = TextContent | ImageContent | ArtifactContent


@dataclass(frozen=True)
class ToolTiming:
    queued_at_ms: int | None = None
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    duration_ms: int | None = None

    def __post_init__(self) -> None:
        for name in ("queued_at_ms", "started_at_ms", "finished_at_ms", "duration_ms"):
            value = getattr(self, name)
            if value is not None:
                _require_non_negative_int(value, name)


@dataclass(frozen=True)
class ToolError:
    code: str
    kind: ToolErrorKind
    message: str
    retryable: bool = False
    recovery_hint: str = ""
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = _clean_text(self.kind)
        if kind not in _TOOL_ERROR_KINDS:
            raise ValueError(f"Unknown tool error kind: {self.kind}")
        if not isinstance(self.retryable, bool):
            raise TypeError("retryable must be bool")
        object.__setattr__(self, "code", _require_text(self.code, "tool error code"))
        object.__setattr__(self, "kind", cast(ToolErrorKind, kind))
        object.__setattr__(self, "message", _require_text(self.message, "tool error message"))
        object.__setattr__(self, "recovery_hint", _clean_text(self.recovery_hint))
        object.__setattr__(self, "details", _freeze_mapping(self.details, "tool error details"))


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    tool_name: str
    status: ToolStatus
    content: tuple[ToolContent, ...] = field(default_factory=tuple)
    data: Mapping[str, object] = field(default_factory=dict)
    error: ToolError | None = None
    effects: tuple[ToolEffect, ...] = field(default_factory=tuple)
    artifacts: tuple[ArtifactRef, ...] = field(default_factory=tuple)
    approval: ApprovalChallenge | None = None
    interaction: Mapping[str, object] | None = None
    timing: ToolTiming = field(default_factory=ToolTiming)
    registration_id: str = ""
    output_validation: OutputValidation = "schema_validated"
    content_trust: ContentTrust = "trusted"

    def __post_init__(self) -> None:
        status = _clean_text(self.status)
        if status not in _TOOL_STATUSES:
            raise ValueError(f"Unknown tool status: {self.status}")
        validation = _clean_text(self.output_validation)
        if validation not in _OUTPUT_VALIDATION:
            raise ValueError(f"Unknown output validation: {self.output_validation}")
        trust = _clean_text(self.content_trust)
        if trust not in _CONTENT_TRUST:
            raise ValueError(f"Unknown content trust: {self.content_trust}")
        content = tuple(self.content)
        if any(not isinstance(item, (TextContent, ImageContent, ArtifactContent)) for item in content):
            raise TypeError("content must contain canonical ToolContent values")
        effects = tuple(self.effects)
        if any(not isinstance(item, ToolEffect) for item in effects):
            raise TypeError("effects must contain ToolEffect values")
        artifacts = tuple(self.artifacts)
        if any(not isinstance(item, ArtifactRef) for item in artifacts):
            raise TypeError("artifacts must contain ArtifactRef values")
        if not isinstance(self.timing, ToolTiming):
            raise TypeError("timing must be ToolTiming")
        if status == "success" and any(
            value is not None for value in (self.error, self.approval, self.interaction)
        ):
            raise ValueError("success result cannot contain error or suspension data")
        if status in {"error", "denied", "cancelled", "timed_out", "interrupted"} and self.error is None:
            raise ValueError(f"{status} result requires an error")
        if status == "approval_required" and self.approval is None:
            raise ValueError("approval_required result requires approval data")
        if status == "user_input_required" and self.interaction is None:
            raise ValueError("user_input_required result requires interaction data")
        if status in {"approval_required", "user_input_required"} and self.error is not None:
            raise ValueError("suspended result cannot contain an error")
        object.__setattr__(self, "tool_call_id", _require_text(self.tool_call_id, "tool_call_id"))
        object.__setattr__(self, "tool_name", _require_text(self.tool_name, "tool_name"))
        object.__setattr__(self, "registration_id", _require_text(self.registration_id, "registration_id"))
        object.__setattr__(self, "status", cast(ToolStatus, status))
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "data", _freeze_mapping(self.data, "tool result data"))
        object.__setattr__(self, "effects", effects)
        object.__setattr__(self, "artifacts", artifacts)
        if self.approval is not None and not isinstance(self.approval, ApprovalChallenge):
            raise TypeError("approval must be ApprovalChallenge")
        object.__setattr__(self, "interaction", _freeze_optional_mapping(self.interaction, "interaction"))
        object.__setattr__(self, "output_validation", cast(OutputValidation, validation))
        object.__setattr__(self, "content_trust", cast(ContentTrust, trust))


def to_tool_result_message(result: ToolResult) -> ToolResultMessage:
    """Project a canonical result into the model conversation protocol."""

    if result.status == "user_input_required":
        raise ValueError("user input suspension cannot be projected as a final tool result")
    conversation_status: ConversationToolResultStatus = cast(
        ConversationToolResultStatus,
        result.status,
    )
    metadata: dict[str, object] = {
        "registration_id": result.registration_id,
        "output_validation": result.output_validation,
        "content_trust": result.content_trust,
    }
    timing = _timing_dict(result.timing)
    if timing:
        metadata["timing"] = timing
    effects = tuple(result.effects)
    return ToolResultMessage(
        tool_call_id=result.tool_call_id,
        tool_name=result.tool_name,
        content=[_to_conversation_content(item) for item in result.content],
        status=conversation_status,
        is_error=conversation_status != "success",
        approved=result.status not in {"approval_required", "denied"},
        approval_id=result.approval.approval_id if result.approval is not None else None,
        error_code=result.error.code if result.error is not None else None,
        exit_code=_optional_int(result.data.get("exit_code")),
        affected_paths=_affected_resource_uris(effects),
        workspace_changed=any(
            effect.kind in {"filesystem_write", "filesystem_delete"} for effect in effects
        ),
        details=_plain_json(result.error.details) if result.error is not None else None,
        metadata=metadata,
    )


def _to_conversation_content(
    item: ToolContent,
) -> ConversationTextContent | ConversationImageContent:
    if isinstance(item, TextContent):
        return ConversationTextContent(text=item.text)
    if isinstance(item, ImageContent):
        return ConversationImageContent(data=item.data, mime_type=item.mime_type)
    return ConversationTextContent(text=f"[artifact:{item.artifact.artifact_id}]")


def _affected_resource_uris(effects: tuple[ToolEffect, ...]) -> list[str]:
    paths: list[str] = []
    for effect in effects:
        if effect.kind.startswith("filesystem_") and effect.resource.uri not in paths:
            paths.append(effect.resource.uri)
    return paths


def _timing_dict(timing: ToolTiming) -> dict[str, int]:
    values = {
        "queued_at_ms": timing.queued_at_ms,
        "started_at_ms": timing.started_at_ms,
        "finished_at_ms": timing.finished_at_ms,
        "duration_ms": timing.duration_ms,
    }
    return {name: value for name, value in values.items() if value is not None}


def _freeze_optional_mapping(
    value: Mapping[str, object] | None,
    field_name: str,
) -> Mapping[str, object] | None:
    return None if value is None else _freeze_mapping(value, field_name)


def _freeze_mapping(value: Mapping[str, object], field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return cast(Mapping[str, object], _freeze_value(dict(value)))


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return deepcopy(value)


def _optional_int(value: object) -> int | None:
    return None if isinstance(value, bool) or not isinstance(value, int) else value


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return deepcopy(value)


def _require_non_negative_int(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value < 0:
        raise ValueError(f"{field_name} cannot be negative")


def _clean_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _require_text(value: object, field_name: str) -> str:
    text = _clean_text(value)
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text


def _optional_text(value: object) -> str | None:
    return _clean_text(value) or None


__all__ = [
    "ArtifactContent",
    "ArtifactRef",
    "ContentTrust",
    "ImageContent",
    "OutputValidation",
    "TextContent",
    "ToolContent",
    "ToolError",
    "ToolErrorKind",
    "ToolResult",
    "ToolStatus",
    "ToolTiming",
    "to_tool_result_message",
]
