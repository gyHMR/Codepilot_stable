from __future__ import annotations

"""Lightweight JSON Schema validation for tool arguments."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SchemaValidationResult:
    """Result of validating one tool call against its argument schema."""

    valid: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class SchemaValidator:
    """Validator for the JSON Schema subset used by tool definitions."""

    def validate(
        self,
        schema: dict[str, Any] | None,
        params: dict[str, Any],
    ) -> SchemaValidationResult:
        if not schema:
            return SchemaValidationResult(valid=True)
        if not isinstance(schema, dict):
            return SchemaValidationResult(valid=True)
        if not isinstance(params, dict):
            return SchemaValidationResult(
                valid=False,
                errors=("tool arguments must be an object",),
            )

        errors = tuple(_validate_value(params, schema, path="$"))
        return SchemaValidationResult(valid=not errors, errors=errors)


DEFAULT_SCHEMA_VALIDATOR = SchemaValidator()


def validate_tool_arguments(
    schema: dict[str, Any] | None,
    params: dict[str, Any],
) -> SchemaValidationResult:
    """Validate tool arguments against the JSON Schema subset used by Codepilot."""
    return DEFAULT_SCHEMA_VALIDATOR.validate(schema, params)


def _validate_value(value: Any, schema: dict[str, Any], *, path: str) -> list[str]:
    errors: list[str] = []

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        errors.append(f"{_display_path(path)} must be one of {enum_values!r}")
        return errors

    expected_types = _schema_types(schema)
    if expected_types and not any(_matches_json_type(value, expected) for expected in expected_types):
        errors.append(f"{_display_path(path)} must be {_format_types(expected_types)}")
        return errors

    should_validate_object = (
        _matches_json_type(value, "object")
        and (
            "object" in expected_types
            or not expected_types
            or any(key in schema for key in ("properties", "required", "additionalProperties"))
        )
    )
    if should_validate_object:
        errors.extend(_validate_object(value, schema, path=path))

    if _matches_json_type(value, "array") and "items" in schema:
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errors.extend(_validate_value(item, item_schema, path=f"{path}[{index}]"))

    return errors


def _validate_object(
    value: dict[str, Any],
    schema: dict[str, Any],
    *,
    path: str,
) -> list[str]:
    errors: list[str] = []
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}

    required = schema.get("required")
    if isinstance(required, list):
        for name in required:
            if isinstance(name, str) and name not in value:
                errors.append(f"missing required argument: {_display_child(path, name)}")

    for name, property_schema in properties.items():
        if not isinstance(name, str) or name not in value:
            continue
        if isinstance(property_schema, dict):
            errors.extend(
                _validate_value(
                    value[name],
                    property_schema,
                    path=_child_path(path, name),
                )
            )

    additional = schema.get("additionalProperties", True)
    known_names = {name for name in properties if isinstance(name, str)}
    unknown_names = [name for name in value if name not in known_names]
    if additional is False:
        for name in unknown_names:
            errors.append(f"unexpected argument: {_display_child(path, name)}")
    elif isinstance(additional, dict):
        for name in unknown_names:
            errors.extend(
                _validate_value(
                    value[name],
                    additional,
                    path=_child_path(path, name),
                )
            )

    return errors


def _schema_types(schema: dict[str, Any]) -> tuple[str, ...]:
    raw_type = schema.get("type")
    if isinstance(raw_type, str):
        return (raw_type,)
    if isinstance(raw_type, list):
        return tuple(item for item in raw_type if isinstance(item, str))
    if any(key in schema for key in ("properties", "required", "additionalProperties")):
        return ("object",)
    return ()


def _matches_json_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _format_types(types: tuple[str, ...]) -> str:
    if len(types) == 1:
        return types[0]
    return " or ".join(types)


def _child_path(path: str, name: object) -> str:
    if path == "$":
        return str(name)
    return f"{path}.{name}"


def _display_child(path: str, name: object) -> str:
    return _display_path(_child_path(path, name))


def _display_path(path: str) -> str:
    return "arguments" if path == "$" else path.removeprefix("$.")


__all__ = [
    "DEFAULT_SCHEMA_VALIDATOR",
    "SchemaValidationResult",
    "SchemaValidator",
    "validate_tool_arguments",
]
