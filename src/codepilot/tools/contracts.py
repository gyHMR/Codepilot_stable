from __future__ import annotations

"""Canonical definitions shared by tool owners, Registry, Runtime, and Core."""

import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Awaitable, Callable, Generic, Literal, Protocol, TYPE_CHECKING, TypeAlias, TypeVar, cast

from .security import (
    ApprovalChallenge,
    ApprovalResponse,
    ToolAccessResolution,
    ToolMode,
    ToolPolicy,
)

if TYPE_CHECKING:
    from .registry import ToolCatalogSnapshot
    from .results import ToolResult
    from .state import InteractionResponse


ToolCategory: TypeAlias = Literal[
    "filesystem",
    "search",
    "command",
    "delegation",
    "plan",
    "interaction",
    "external",
]
ToolSource: TypeAlias = Literal["builtin", "caller", "skill", "extension", "mcp"]

_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_TOOL_CATEGORIES = frozenset(
    {"filesystem", "search", "command", "delegation", "plan", "interaction", "external"}
)
_TOOL_SOURCES = frozenset({"builtin", "caller", "skill", "extension", "mcp"})
_TOOL_MODES = frozenset({"plan", "execute", "unrestricted"})


@dataclass(frozen=True)
class ToolSpec:
    """Immutable model-visible definition of a tool."""

    name: str
    description: str
    input_schema: Mapping[str, object]
    output_schema: Mapping[str, object] | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        name = _require_text(self.name, "tool name")
        if _TOOL_NAME_PATTERN.fullmatch(name) is None:
            raise ValueError(f"Invalid tool name: {self.name}")
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise TypeError("schema_version must be int")
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", _require_text(self.description, "description"))
        object.__setattr__(self, "input_schema", _freeze_json_mapping(self.input_schema, "input_schema"))
        if self.output_schema is not None:
            object.__setattr__(
                self,
                "output_schema",
                _freeze_json_mapping(self.output_schema, "output_schema"),
            )


@dataclass(frozen=True)
class ToolExecutionRequest:
    """Runtime request created from the exact catalog snapshot observed by the model."""

    run_id: str
    session_id: str
    tool_call_id: str
    tool_name: str
    arguments: Mapping[str, object]
    mode: ToolMode
    registration_id: str
    idempotency_key: str | None = None
    deadline_at_ms: int | None = None

    def __post_init__(self) -> None:
        for name in ("run_id", "session_id", "tool_call_id", "tool_name", "registration_id"):
            object.__setattr__(self, name, _require_text(getattr(self, name), name))
        mode = _clean_text(self.mode)
        if mode not in _TOOL_MODES:
            raise ValueError(f"Unknown tool mode: {self.mode}")
        if self.deadline_at_ms is not None and (
            isinstance(self.deadline_at_ms, bool) or not isinstance(self.deadline_at_ms, int)
        ):
            raise TypeError("deadline_at_ms must be int or None")
        object.__setattr__(self, "mode", cast(ToolMode, mode))
        object.__setattr__(self, "arguments", _freeze_json_mapping(self.arguments, "arguments"))
        object.__setattr__(self, "idempotency_key", _optional_text(self.idempotency_key))


class CancellationToken(Protocol):
    @property
    def cancelled(self) -> bool: ...

    def raise_if_cancelled(self) -> None: ...


class ProgressReporter(Protocol):
    async def report(
        self,
        kind: str,
        *,
        message: str = "",
        data: Mapping[str, object] | None = None,
    ) -> None: ...


class EffectReporter(Protocol):
    def report(self, effect: object) -> None: ...


CleanupCallback: TypeAlias = Callable[[], Awaitable[None] | None]


class CleanupStack(Protocol):
    def push(self, callback: CleanupCallback) -> None: ...


@dataclass(frozen=True)
class ToolExecutionContext:
    cancellation: CancellationToken
    deadline_at_ms: int | None
    progress: ProgressReporter
    effects: EffectReporter
    cleanup: CleanupStack


TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class ToolHandlerError(Exception):
    """Expected domain failure raised by a tool handler."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: Mapping[str, object] | None = None,
    ) -> None:
        self.code = _require_text(code, "tool handler error code")
        self.message = _require_text(message, "tool handler error message")
        self.retryable = bool(retryable)
        if details is not None and not isinstance(details, Mapping):
            raise TypeError("tool handler error details must be a mapping")
        self.details = deepcopy(dict(details or {}))
        super().__init__(self.message)


class ToolCodec(Protocol, Generic[TInput]):
    @property
    def json_schema(self) -> Mapping[str, object] | None: ...

    def decode(self, value: object) -> TInput: ...

    def encode(self, value: TInput) -> object: ...


class ToolHandler(Protocol, Generic[TInput, TOutput]):
    async def __call__(self, input: TInput, context: ToolExecutionContext) -> TOutput: ...


class ToolAccessResolver(Protocol, Generic[TInput]):
    def resolve(
        self,
        input: TInput,
        request: ToolExecutionRequest,
    ) -> ToolAccessResolution[TInput]: ...


class ToolOutputRenderer(Protocol):
    def render(self, data: Mapping[str, object]) -> tuple[object, ...]: ...


@dataclass(frozen=True)
class ToolRegistration:
    """Owner-provided canonical definition before Registry materialization."""

    version: str
    implementation_version: str
    spec: ToolSpec
    category: ToolCategory
    source: ToolSource
    owner: str
    policy: ToolPolicy
    input_codec: ToolCodec[object]
    output_codec: ToolCodec[object]
    handler: ToolHandler[object, object]
    renderer: ToolOutputRenderer
    access_resolver: ToolAccessResolver[object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _require_text(self.version, "tool version"))
        object.__setattr__(
            self,
            "implementation_version",
            _require_text(self.implementation_version, "implementation_version"),
        )
        object.__setattr__(self, "owner", _require_text(self.owner, "tool owner"))
        if not isinstance(self.spec, ToolSpec):
            raise TypeError("spec must be ToolSpec")
        category = _clean_text(self.category)
        if category not in _TOOL_CATEGORIES:
            raise ValueError(f"Unknown tool category: {self.category}")
        source = _clean_text(self.source)
        if source not in _TOOL_SOURCES:
            raise ValueError(f"Unknown tool source: {self.source}")
        if not isinstance(self.policy, ToolPolicy):
            raise TypeError("policy must be ToolPolicy")
        if not callable(self.handler):
            raise TypeError("handler must be callable")
        if not callable(getattr(self.input_codec, "decode", None)):
            raise TypeError("input_codec must implement decode")
        if not callable(getattr(self.output_codec, "encode", None)):
            raise TypeError("output_codec must implement encode")
        if not callable(getattr(self.renderer, "render", None)):
            raise TypeError("renderer must implement render")
        if not callable(getattr(self.access_resolver, "resolve", None)):
            raise TypeError("access_resolver must implement resolve")
        object.__setattr__(self, "category", cast(ToolCategory, category))
        object.__setattr__(self, "source", cast(ToolSource, source))


class ToolPort(Protocol):
    def catalog_snapshot(self, *, mode: ToolMode | None = None) -> ToolCatalogSnapshot: ...

    def pending_challenges(self) -> tuple[ApprovalChallenge, ...]: ...

    async def execute(self, request: ToolExecutionRequest) -> ToolResult: ...

    async def execute_batch(
        self,
        requests: tuple[ToolExecutionRequest, ...] | list[ToolExecutionRequest],
    ) -> list[ToolResult]: ...

    async def cancel(self, attempt_id: str) -> bool: ...

    def approval_challenge(self, approval_id: str): ...

    async def resume(self, response: ApprovalResponse | InteractionResponse) -> ToolResult: ...


def _freeze_json_mapping(value: Mapping[str, object], field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return cast(Mapping[str, object], _freeze_json_value(dict(value)))


def _freeze_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    return deepcopy(value)


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
    "CancellationToken",
    "CleanupStack",
    "EffectReporter",
    "ProgressReporter",
    "ToolAccessResolver",
    "ToolCategory",
    "ToolCodec",
    "ToolExecutionContext",
    "ToolExecutionRequest",
    "ToolHandler",
    "ToolHandlerError",
    "ToolOutputRenderer",
    "ToolPort",
    "ToolRegistration",
    "ToolSource",
    "ToolSpec",
]
