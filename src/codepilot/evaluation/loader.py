from __future__ import annotations

"""Strict JSON loader for Codepilot evaluation definitions."""

import json
from pathlib import Path
from typing import Any, cast

from .types import (
    AssertionSpec,
    AssertionType,
    EvalBudgets,
    EvalCase,
    EvalDefinition,
    EvalDimension,
    EvalDomain,
    EvalRuntimeProfile,
    EvalScenario,
    ScenarioStep,
)


class EvalCaseValidationError(ValueError):
    """Benchmark JSON does not match the supported Eval schema."""


_DOMAINS = {
    "runtime",
    "coding",
    "context",
    "memory",
    "security",
    "planning",
    "recovery",
}
_ASSERTION_DIMENSIONS: dict[str, EvalDimension] = {
    "command": "coding_outcome",
    "file": "coding_outcome",
    "diff": "coding_outcome",
    "run": "runtime_contract",
    "trace": "runtime_contract",
    "context": "context_governance",
    "memory": "memory",
    "security": "tool_security",
    "task": "task_planning",
}


def load_eval_definition(path: str | Path) -> EvalDefinition:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalCaseValidationError(f"Cannot load {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvalCaseValidationError(f"{source}: root must be a JSON object")
    return parse_eval_definition(payload, source=str(source))


def load_eval_suite(path: str | Path) -> list[EvalDefinition]:
    root = Path(path)
    if root.is_file():
        return [load_eval_definition(root)]
    if not root.is_dir():
        raise EvalCaseValidationError(f"Suite path does not exist: {root}")
    definitions = [
        load_eval_definition(source)
        for source in sorted(root.rglob("*.json"))
    ]
    if not definitions:
        raise EvalCaseValidationError(f"No JSON cases found under: {root}")
    ids = [item.id for item in definitions]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise EvalCaseValidationError(
            f"Duplicate case ids: {', '.join(duplicates)}"
        )
    return definitions


def parse_eval_definition(
    payload: dict[str, Any],
    *,
    source: str = "<memory>",
) -> EvalDefinition:
    case_id = _required_text(payload, "id", source)
    fixture = _required_text(payload, "fixture", source)
    domain = payload.get("domain")
    if domain not in _DOMAINS:
        raise EvalCaseValidationError(
            f"{source}.domain must be one of: {', '.join(sorted(_DOMAINS))}"
        )
    assertions = _parse_assertions(payload.get("assertions", []), source)
    if not assertions:
        raise EvalCaseValidationError(
            f"{source}.assertions must contain at least one assertion"
        )
    budgets = _parse_budgets(payload.get("budgets", {}), payload, source)
    runtime = _parse_runtime(payload.get("runtime", {}), source)
    tags = _string_list(payload.get("tags", []), f"{source}.tags")

    if "steps" in payload:
        raw_steps = payload["steps"]
        if not isinstance(raw_steps, list) or not raw_steps:
            raise EvalCaseValidationError(
                f"{source}.steps must be a non-empty array"
            )
        return EvalScenario(
            id=case_id,
            domain=cast(EvalDomain, domain),
            fixture=fixture,
            steps=[
                _parse_step(item, f"{source}.steps[{index}]")
                for index, item in enumerate(raw_steps)
            ],
            assertions=assertions,
            budgets=budgets,
            runtime=runtime,
            tags=tags,
        )

    return EvalCase(
        id=case_id,
        domain=cast(EvalDomain, domain),
        fixture=fixture,
        prompt=_required_text(payload, "prompt", source),
        assertions=assertions,
        budgets=budgets,
        runtime=runtime,
        tags=tags,
    )


def _parse_assertions(value: Any, source: str) -> list[AssertionSpec]:
    if not isinstance(value, list):
        raise EvalCaseValidationError(f"{source}.assertions must be an array")
    return [
        _parse_assertion(item, f"{source}.assertions[{index}]")
        for index, item in enumerate(value)
    ]


def _parse_assertion(value: Any, source: str) -> AssertionSpec:
    if not isinstance(value, dict):
        raise EvalCaseValidationError(f"{source} must be an object")
    assertion_type = value.get("type")
    if assertion_type not in _ASSERTION_DIMENSIONS:
        raise EvalCaseValidationError(
            f"{source}.type is not a supported assertion"
        )
    dimension = value.get(
        "dimension",
        _ASSERTION_DIMENSIONS[str(assertion_type)],
    )
    if dimension not in {
        "coding_outcome",
        "runtime_contract",
        "context_governance",
        "memory",
        "tool_security",
        "task_planning",
        "recovery",
        "efficiency",
    }:
        raise EvalCaseValidationError(f"{source}.dimension is not supported")
    options = {
        key: item
        for key, item in value.items()
        if key not in {"type", "dimension", "required"}
    }
    _validate_assertion_options(str(assertion_type), options, source)
    return AssertionSpec(
        type=cast(AssertionType, assertion_type),
        dimension=cast(EvalDimension, dimension),
        options=options,
        required=bool(value.get("required", True)),
    )


def _validate_assertion_options(
    assertion_type: str,
    options: dict[str, Any],
    source: str,
) -> None:
    if assertion_type == "command":
        command = options.get("command")
        if not isinstance(command, str) or not command.strip():
            raise EvalCaseValidationError(
                f"{source}.command must be a non-empty string"
            )
    elif assertion_type == "file":
        path = options.get("path")
        if not isinstance(path, str) or not path.strip():
            raise EvalCaseValidationError(
                f"{source}.path must be a non-empty string"
            )
    elif assertion_type == "diff":
        for key in ("allowed_paths", "forbidden_paths"):
            _string_list(options.get(key, []), f"{source}.{key}")


def _parse_step(value: Any, source: str) -> ScenarioStep:
    if not isinstance(value, dict):
        raise EvalCaseValidationError(f"{source} must be an object")
    step_type = value.get("type")
    if step_type not in {
        "prompt",
        "cancel",
        "modify_file",
        "restart",
        "continue",
        "verify",
        "inspect",
    }:
        raise EvalCaseValidationError(f"{source}.type is not supported")
    options = {key: item for key, item in value.items() if key != "type"}
    if step_type == "prompt":
        text = options.get("text")
        if not isinstance(text, str) or not text.strip():
            raise EvalCaseValidationError(
                f"{source}.text must be a non-empty string"
            )
    elif step_type == "modify_file":
        path = options.get("path")
        if not isinstance(path, str) or not path.strip():
            raise EvalCaseValidationError(
                f"{source}.path must be a non-empty string"
            )
        if "source" not in options and "content" not in options:
            raise EvalCaseValidationError(
                f"{source} requires source or content"
            )
    elif step_type == "verify":
        options["assertion"] = _parse_assertion(
            options.get("assertion"),
            f"{source}.assertion",
        )
    return ScenarioStep(type=step_type, options=options)


def _parse_budgets(
    value: Any,
    payload: dict[str, Any],
    source: str,
) -> EvalBudgets:
    if not isinstance(value, dict):
        raise EvalCaseValidationError(f"{source}.budgets must be an object")
    timeout = value.get(
        "timeout_seconds",
        payload.get("timeout_seconds", 120),
    )
    return EvalBudgets(
        max_model_attempts=_optional_positive_int(
            value.get("max_model_attempts"),
            f"{source}.budgets.max_model_attempts",
        ),
        max_tool_calls=_optional_positive_int(
            value.get("max_tool_calls"),
            f"{source}.budgets.max_tool_calls",
        ),
        max_replans=_optional_positive_int(
            value.get("max_replans"),
            f"{source}.budgets.max_replans",
        ),
        timeout_seconds=_positive_int(
            timeout,
            f"{source}.budgets.timeout_seconds",
        ),
    )


def _parse_runtime(value: Any, source: str) -> EvalRuntimeProfile:
    if not isinstance(value, dict):
        raise EvalCaseValidationError(f"{source}.runtime must be an object")
    permission_mode = str(value.get("permission_mode", "workspace-write"))
    if permission_mode not in {"read-only", "workspace-write", "ask"}:
        raise EvalCaseValidationError(
            f"{source}.runtime.permission_mode is invalid"
        )
    scripted_stream = value.get("scripted_stream")
    if scripted_stream is not None and not isinstance(scripted_stream, str):
        raise EvalCaseValidationError(
            f"{source}.runtime.scripted_stream must be a string"
        )
    return EvalRuntimeProfile(
        context_governance_enabled=bool(
            value.get("context_governance_enabled", True)
        ),
        memory_enabled=bool(value.get("memory_enabled", True)),
        task_control_enabled=bool(value.get("task_control_enabled", True)),
        permission_mode=permission_mode,
        scripted_stream=scripted_stream,
    )


def _required_text(payload: dict[str, Any], key: str, source: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EvalCaseValidationError(
            f"{source}.{key} must be a non-empty string"
        )
    return value


def _string_list(value: Any, source: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise EvalCaseValidationError(
            f"{source} must be an array of strings"
        )
    return list(value)


def _positive_int(value: Any, source: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise EvalCaseValidationError(f"{source} must be a positive integer")
    return value


def _optional_positive_int(value: Any, source: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, source)
