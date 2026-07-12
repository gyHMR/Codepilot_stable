from __future__ import annotations

"""Canonical input/output codecs backed by JSON Schema Draft 2020-12."""

import json
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Generic, Mapping, TypeVar, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


class ToolCodecError(ValueError):
    """A tool value cannot be decoded or encoded by its declared codec."""


class ToolSchemaError(ValueError):
    """A tool codec was constructed with an invalid JSON Schema."""


class JsonObjectCodec:
    """Strict JSON object codec using a validated Draft 2020-12 schema."""

    def __init__(self, schema: Mapping[str, object]) -> None:
        self._schema = _mapping_copy(schema, "schema")
        validate_json_schema(self._schema, require_object=True)
        self._validator = Draft202012Validator(self._schema)

    @property
    def json_schema(self) -> Mapping[str, object]:
        return deepcopy(self._schema)

    def decode(self, value: object) -> dict[str, object]:
        return self._validate(value)

    def encode(self, value: object) -> dict[str, object]:
        return self._validate(value)

    def _validate(self, value: object) -> dict[str, object]:
        if not isinstance(value, Mapping):
            raise ToolCodecError("Tool value must be a JSON object")
        copied = _mapping_copy(value, "tool value")
        errors = sorted(self._validator.iter_errors(copied), key=lambda item: list(item.path))
        if errors:
            first = errors[0]
            path = ".".join(str(item) for item in first.absolute_path)
            prefix = f"{path}: " if path else ""
            raise ToolCodecError(prefix + first.message)
        return copied


TDataclass = TypeVar("TDataclass")


class DataclassCodec(Generic[TDataclass]):
    """Decode a validated object into one dataclass type and encode it back."""

    def __init__(self, dataclass_type: type[TDataclass], schema: Mapping[str, object]) -> None:
        if not is_dataclass(dataclass_type):
            raise TypeError("dataclass_type must be a dataclass")
        self._type = dataclass_type
        self._object_codec = JsonObjectCodec(schema)

    @property
    def json_schema(self) -> Mapping[str, object]:
        return self._object_codec.json_schema

    def decode(self, value: object) -> TDataclass:
        decoded = self._object_codec.decode(value)
        try:
            return self._type(**decoded)
        except TypeError as exc:
            raise ToolCodecError(f"Cannot construct {self._type.__name__}: {exc}") from exc

    def encode(self, value: TDataclass) -> dict[str, object]:
        if not isinstance(value, self._type):
            raise ToolCodecError(f"Expected {self._type.__name__} output")
        return self._object_codec.encode(cast(Mapping[str, object], asdict(value)))


class UnverifiedJsonCodec:
    """JSON-safe codec for external outputs that do not declare a schema."""

    def __init__(self, *, max_bytes: int = 1_000_000, max_depth: int = 32) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth <= 0:
            raise ValueError("max_depth must be a positive integer")
        self._max_bytes = max_bytes
        self._max_depth = max_depth

    @property
    def json_schema(self) -> None:
        return None

    def decode(self, value: object) -> object:
        return self._validate(value)

    def encode(self, value: object) -> object:
        return self._validate(value)

    def _validate(self, value: object) -> object:
        if _json_depth(value) > self._max_depth:
            raise ToolCodecError(f"JSON value exceeds maximum depth {self._max_depth}")
        try:
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ToolCodecError("Tool value must be JSON-safe") from exc
        if len(encoded.encode("utf-8")) > self._max_bytes:
            raise ToolCodecError(f"JSON value exceeds maximum size {self._max_bytes} bytes")
        return json.loads(encoded)


def _mapping_copy(value: Mapping[str, object], field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return {str(key): _json_copy(item) for key, item in value.items()}


def validate_json_schema(
    schema: Mapping[str, object],
    *,
    require_object: bool = False,
) -> dict[str, object]:
    """Validate and return a defensive Draft 2020-12 schema copy."""

    copied = _mapping_copy(schema, "schema")
    if require_object and copied.get("type") != "object":
        raise ToolSchemaError("JSON object codec schema must declare type=object")
    try:
        Draft202012Validator.check_schema(copied)
    except SchemaError as exc:
        raise ToolSchemaError(f"Invalid JSON Schema: {exc.message}") from exc
    return copied


def _json_copy(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_copy(item) for item in value]
    return deepcopy(value)


def _json_depth(value: object) -> int:
    if isinstance(value, Mapping):
        return 1 + max((_json_depth(item) for item in value.values()), default=0)
    if isinstance(value, (list, tuple)):
        return 1 + max((_json_depth(item) for item in value), default=0)
    return 0


__all__ = [
    "DataclassCodec",
    "JsonObjectCodec",
    "ToolCodecError",
    "ToolSchemaError",
    "UnverifiedJsonCodec",
    "validate_json_schema",
]
