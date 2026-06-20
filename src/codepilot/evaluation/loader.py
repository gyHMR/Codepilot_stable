from __future__ import annotations

"""Strict JSON loading for evaluation cases and scenarios."""

import json
from pathlib import Path
from typing import Any

from .types import EvalCase, EvalScenario, ScenarioStep, VerifierSpec


class EvalCaseValidationError(ValueError):
    """Benchmark JSON does not match the supported Eval schema."""


def load_eval_definition(path: str | Path) -> EvalCase | EvalScenario:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalCaseValidationError(f"Cannot load {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvalCaseValidationError(f"{source}: root must be a JSON object")
    return parse_eval_definition(payload, source=str(source))


def load_eval_suite(path: str | Path) -> list[EvalCase | EvalScenario]:
    root = Path(path)
    if root.is_file():
        return [load_eval_definition(root)]
    if not root.is_dir():
        raise EvalCaseValidationError(f"Suite path does not exist: {root}")
    definitions: list[EvalCase | EvalScenario] = []
    for source in sorted(root.rglob("*.json")):
        definitions.append(load_eval_definition(source))
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
) -> EvalCase | EvalScenario:
    case_id = _required_text(payload, "id", source)
    fixture = _required_text(payload, "fixture", source)
    timeout_seconds = _positive_int(
        payload.get("timeout_seconds", 120),
        f"{source}.timeout_seconds",
    )
    verifiers = _parse_verifiers(payload.get("verifiers", []), source)

    if "steps" in payload:
        raw_steps = payload["steps"]
        if not isinstance(raw_steps, list) or not raw_steps:
            raise EvalCaseValidationError(
                f"{source}.steps must be a non-empty array"
            )
        steps = [
            _parse_step(item, f"{source}.steps[{index}]")
            for index, item in enumerate(raw_steps)
        ]
        return EvalScenario(
            id=case_id,
            fixture=fixture,
            steps=steps,
            verifiers=verifiers,
            timeout_seconds=timeout_seconds,
        )

    category = payload.get("category")
    if category not in {"harness", "coding"}:
        raise EvalCaseValidationError(
            f"{source}.category must be 'harness' or 'coding'"
        )
    prompt = _required_text(payload, "prompt", source)
    return EvalCase(
        id=case_id,
        category=category,
        fixture=fixture,
        prompt=prompt,
        timeout_seconds=timeout_seconds,
        verifiers=verifiers,
    )


def _parse_verifiers(value: Any, source: str) -> list[VerifierSpec]:
    if not isinstance(value, list):
        raise EvalCaseValidationError(f"{source}.verifiers must be an array")
    return [
        _parse_verifier(item, f"{source}.verifiers[{index}]")
        for index, item in enumerate(value)
    ]


def _parse_verifier(value: Any, source: str) -> VerifierSpec:
    if not isinstance(value, dict):
        raise EvalCaseValidationError(f"{source} must be an object")
    verifier_type = value.get("type")
    if verifier_type not in {"command", "file", "diff", "run", "trace"}:
        raise EvalCaseValidationError(
            f"{source}.type is not a supported verifier"
        )
    options = {key: item for key, item in value.items() if key != "type"}
    _validate_verifier_options(verifier_type, options, source)
    return VerifierSpec(type=verifier_type, options=options)


def _validate_verifier_options(
    verifier_type: str,
    options: dict[str, Any],
    source: str,
) -> None:
    if verifier_type == "command":
        command = options.get("command")
        if not isinstance(command, str) or not command.strip():
            raise EvalCaseValidationError(
                f"{source}.command must be a non-empty string"
            )
    elif verifier_type == "file":
        path = options.get("path")
        if not isinstance(path, str) or not path.strip():
            raise EvalCaseValidationError(
                f"{source}.path must be a non-empty string"
            )
    elif verifier_type == "diff":
        for key in ("allowed_paths", "forbidden_paths"):
            value = options.get(key, [])
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                raise EvalCaseValidationError(
                    f"{source}.{key} must be an array of strings"
                )


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
    }:
        raise EvalCaseValidationError(f"{source}.type is not supported")
    options = {key: item for key, item in value.items() if key != "type"}
    if step_type == "prompt":
        text = options.get("text")
        if not isinstance(text, str) or not text.strip():
            raise EvalCaseValidationError(
                f"{source}.text must be a non-empty string"
            )
    if step_type == "modify_file":
        path = options.get("path")
        if not isinstance(path, str) or not path.strip():
            raise EvalCaseValidationError(
                f"{source}.path must be a non-empty string"
            )
        if "source" not in options and "content" not in options:
            raise EvalCaseValidationError(
                f"{source} requires source or content"
            )
    if step_type == "verify":
        verifier = options.get("verifier")
        _parse_verifier(verifier, f"{source}.verifier")
    return ScenarioStep(type=step_type, options=options)


def _required_text(payload: dict[str, Any], key: str, source: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EvalCaseValidationError(
            f"{source}.{key} must be a non-empty string"
        )
    return value


def _positive_int(value: Any, source: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise EvalCaseValidationError(f"{source} must be a positive integer")
    return value
