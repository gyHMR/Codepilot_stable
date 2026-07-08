from __future__ import annotations

"""Tool registry, catalog exposure, and call preparation."""

import json
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any, Iterable

from codepilot.protocols import TextContent

from .contracts import (
    PreparedToolCall,
    PreparedToolCallResult,
    ToolCallRequest,
    ToolCatalogItem,
    ToolCatalogView,
    ToolDefinition,
    ToolMetadata,
)

READ_ONLY_TOOL_NAMES = {"ls", "read", "grep", "find", "workspace_status", "update_plan"}
MUTATING_TOOL_NAMES = {"write", "edit", "apply_patch", "bash"}


@dataclass
class ToolRegistry:
    """In-memory registry for one opened runtime session."""

    _tools: dict[str, ToolDefinition] = field(default_factory=dict)

    def register(self, tool: ToolDefinition, *, replace: bool = True) -> None:
        if not isinstance(tool, ToolDefinition):
            raise TypeError("ToolRegistry.register expects ToolDefinition")
        if not replace and tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def extend(self, tools: Iterable[ToolDefinition], *, replace: bool = True) -> None:
        for tool in tools:
            self.register(tool, replace=replace)

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def metadata_for(self, name: str) -> ToolMetadata | None:
        tool = self.get(name)
        return tool.metadata if tool is not None else None

    def list(self, *, current_mode: str | None = None) -> list[ToolDefinition]:
        tools = sorted(self._tools.values(), key=lambda item: (item.metadata.category, item.name))
        if current_mode is None:
            return tools
        return [tool for tool in tools if tool.metadata.visible_in(current_mode)]

    def catalog(self, *, current_mode: str = "build") -> ToolCatalogView:
        return ToolCatalogView(
            tuple(
                ToolCatalogItem(spec=tool.to_spec(), metadata=tool.metadata)
                for tool in self.list(current_mode=current_mode)
            )
        )

    def prepare_call(
        self,
        *,
        run_id: str,
        tool_call_id: str,
        name: str,
        arguments: dict[str, Any] | str | None,
        current_mode: str,
        source: str = "agent",
        policy_context: Any = None,
        assistant_message: Any = None,
        context: Any = None,
    ) -> PreparedToolCallResult:
        tool = self.get(name)
        if tool is None:
            hint = _tool_name_hint(name, self._tools)
            return PreparedToolCallResult(
                error_code="tool_not_found",
                message=f"Tool '{name}' not found.{hint}",
                recovery_hint=hint.strip(),
            )
        if not tool.metadata.visible_in(current_mode):
            return PreparedToolCallResult(
                error_code="tool_not_available_in_mode",
                message=f"Tool '{name}' is not available in {current_mode} mode.",
                recovery_hint="Use an available tool for the current mode or switch mode.",
            )
        parsed = _parse_arguments(arguments)
        if parsed.error is not None:
            return PreparedToolCallResult(
                error_code="invalid_tool_arguments",
                message=parsed.error,
                recovery_hint="Pass tool arguments as a JSON object matching the schema.",
            )
        unwrapped = _unwrap_arguments(parsed.value, tool.parameters)
        coerced = _coerce_arguments(unwrapped, tool.parameters)
        if coerced.errors:
            return PreparedToolCallResult(
                error_code="invalid_tool_arguments",
                message="Tool arguments failed schema validation: " + "; ".join(coerced.errors),
                recovery_hint="Correct the arguments and retry the same exact tool name.",
            )
        request = ToolCallRequest(
            run_id=run_id,
            tool_call_id=tool_call_id,
            name=name,
            arguments=coerced.value,
            metadata=tool.metadata,
            current_mode=current_mode,
            source=source,  # type: ignore[arg-type]
            policy_context=policy_context or _default_policy_context(),
            assistant_message=assistant_message,
            context=context,
        )
        return PreparedToolCallResult(call=PreparedToolCall(definition=tool, request=request))


def prepare_error_content(result: PreparedToolCallResult) -> tuple[TextContent, ...]:
    text = result.message or result.error_code or "Tool call could not be prepared."
    if result.recovery_hint:
        text += f"\nRecovery hint: {result.recovery_hint}"
    return (TextContent(text=text),)


def _default_policy_context() -> Any:
    from .contracts import ToolPolicyContext

    return ToolPolicyContext()


@dataclass(frozen=True)
class _ParsedArguments:
    value: dict[str, Any]
    error: str | None = None


@dataclass(frozen=True)
class _CoercedArguments:
    value: dict[str, Any]
    errors: tuple[str, ...] = ()


def _parse_arguments(value: dict[str, Any] | str | None) -> _ParsedArguments:
    if value is None:
        return _ParsedArguments({})
    if isinstance(value, dict):
        return _ParsedArguments(dict(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return _ParsedArguments({})
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return _ParsedArguments({}, f"arguments must be valid JSON: {exc.msg}")
        if not isinstance(parsed, dict):
            return _ParsedArguments({}, "arguments must decode to a JSON object")
        return _ParsedArguments(parsed)
    return _ParsedArguments({}, "arguments must be an object or JSON object string")


def _unwrap_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    if set(arguments) != {"arguments"} or not isinstance(arguments.get("arguments"), dict):
        return arguments
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return arguments
    nested = arguments["arguments"]
    if any(key in properties for key in nested):
        return dict(nested)
    return arguments


def _coerce_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> _CoercedArguments:
    if not isinstance(schema, dict) or not schema:
        return _CoercedArguments(dict(arguments))
    value = dict(arguments)
    errors: list[str] = []
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    required = schema.get("required")
    if isinstance(required, list):
        for name in required:
            if isinstance(name, str) and name not in value:
                errors.append(f"missing required argument: {name}")
    for name, raw_schema in properties.items():
        if not isinstance(name, str) or name not in value or not isinstance(raw_schema, dict):
            continue
        converted, error = _coerce_value(value[name], raw_schema, name)
        if error is not None:
            errors.append(error)
        else:
            value[name] = converted
    additional = schema.get("additionalProperties", True)
    if additional is False:
        unknown = sorted(name for name in value if name not in properties)
        errors.extend(f"unexpected argument: {name}" for name in unknown)
    return _CoercedArguments(value=value, errors=tuple(errors))


def _coerce_value(value: Any, schema: dict[str, Any], path: str) -> tuple[Any, str | None]:
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        return None, f"{path} must be one of {enum_values!r}"
    expected = _schema_type(schema)
    if expected == "integer":
        if isinstance(value, bool):
            return None, f"{path} must be integer"
        if isinstance(value, int):
            return value, None
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value), None
        return None, f"{path} must be integer"
    if expected == "number":
        if isinstance(value, bool):
            return None, f"{path} must be number"
        if isinstance(value, (int, float)):
            return value, None
        if isinstance(value, str):
            try:
                return float(value), None
            except ValueError:
                return None, f"{path} must be number"
    if expected == "boolean":
        if isinstance(value, bool):
            return value, None
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true", None
        return None, f"{path} must be boolean"
    if expected == "string":
        return value if isinstance(value, str) else str(value), None
    if expected == "array":
        if not isinstance(value, list):
            return None, f"{path} must be array"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            converted_items: list[Any] = []
            errors: list[str] = []
            for index, item in enumerate(value):
                converted, error = _coerce_value(item, item_schema, f"{path}[{index}]")
                if error:
                    errors.append(error)
                else:
                    converted_items.append(converted)
            if errors:
                return None, "; ".join(errors)
            value = converted_items
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if isinstance(min_items, int) and len(value) < min_items:
            return None, f"{path} must contain at least {min_items} item(s)"
        if isinstance(max_items, int) and len(value) > max_items:
            return None, f"{path} must contain at most {max_items} item(s)"
        return value, None
    if expected == "object":
        if not isinstance(value, dict):
            return None, f"{path} must be object"
        nested = _coerce_arguments(value, schema)
        if nested.errors:
            return None, "; ".join(nested.errors)
        return nested.value, None
    return value, None


def _schema_type(schema: dict[str, Any]) -> str | None:
    raw = schema.get("type")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return next((item for item in raw if isinstance(item, str) and item != "null"), None)
    if any(key in schema for key in ("properties", "required", "additionalProperties")):
        return "object"
    return None


def _tool_name_hint(name: str, tools: dict[str, ToolDefinition]) -> str:
    matches = get_close_matches(name, tools.keys(), n=1)
    if matches:
        return f" Did you mean '{matches[0]}'? Tool names must match exactly."
    return " Tool names must match exactly."


def builtin_metadata(
    name: str,
    *,
    category: str,
    read_only: bool,
    risk_level: str,
    scopes: tuple[str, ...],
    requires_approval: bool = False,
    network_access: bool = False,
    credential_required: bool = False,
    extra: dict[str, Any] | None = None,
) -> ToolMetadata:
    return ToolMetadata(
        name=name,
        category=category,
        read_only=read_only,
        concurrency_safe=read_only,
        exclusive=not read_only,
        requires_approval=requires_approval,
        risk_level=risk_level,  # type: ignore[arg-type]
        scopes=scopes,
        network_access=network_access,
        credential_required=credential_required,
        extra=extra or {},
    )


def get_builtin_tool_metadata(name: str) -> ToolMetadata | None:
    return _BUILTIN_METADATA.get(name)


_BUILTIN_METADATA: dict[str, ToolMetadata] = {
    "ls": builtin_metadata(
        "ls",
        category="filesystem",
        read_only=True,
        risk_level="low",
        scopes=("read", "plan", "build"),
    ),
    "read": builtin_metadata(
        "read",
        category="filesystem",
        read_only=True,
        risk_level="low",
        scopes=("read", "plan", "build"),
    ),
    "write": builtin_metadata(
        "write",
        category="filesystem",
        read_only=False,
        risk_level="medium",
        scopes=("build",),
    ),
    "edit": builtin_metadata(
        "edit",
        category="filesystem",
        read_only=False,
        risk_level="medium",
        scopes=("build",),
    ),
    "apply_patch": builtin_metadata(
        "apply_patch",
        category="filesystem",
        read_only=False,
        risk_level="medium",
        scopes=("build",),
    ),
    "grep": builtin_metadata(
        "grep",
        category="search",
        read_only=True,
        risk_level="low",
        scopes=("read", "plan", "build"),
    ),
    "find": builtin_metadata(
        "find",
        category="search",
        read_only=True,
        risk_level="low",
        scopes=("read", "plan", "build"),
    ),
    "bash": builtin_metadata(
        "bash",
        category="shell",
        read_only=False,
        risk_level="medium",
        scopes=("build",),
    ),
    "workspace_status": builtin_metadata(
        "workspace_status",
        category="workspace",
        read_only=True,
        risk_level="low",
        scopes=("read", "plan", "build"),
    ),
    "update_plan": builtin_metadata(
        "update_plan",
        category="plan",
        read_only=True,
        risk_level="low",
        scopes=("read", "plan", "build"),
    ),
}


__all__ = [
    "MUTATING_TOOL_NAMES",
    "READ_ONLY_TOOL_NAMES",
    "ToolRegistry",
    "builtin_metadata",
    "get_builtin_tool_metadata",
    "prepare_error_content",
]
