from __future__ import annotations

# 新手导读：loader.py 负责读取 benchmark JSON 文件并校验基础结构。
# 关注点：新增 benchmark 格式时先看这里。

"""Load evaluation v2 case definitions."""

import json
from pathlib import Path
from typing import Any

from .schema import EvalCase, EvalCheck, EvalStep


class EvalCaseValidationError(ValueError):
    """Raised when an evaluation case does not match the v2 schema."""


def load_eval_case(path: Path | str) -> EvalCase:
    """Load one JSON evaluation case."""

    case_path = Path(path)
    try:
        payload = json.loads(case_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvalCaseValidationError(f"Invalid JSON in {case_path}: {exc}") from exc
    return parse_eval_case(payload, source=case_path)


def load_eval_suite(path: Path | str) -> list[EvalCase]:
    """Load all v2 cases from a file or directory."""

    suite_path = Path(path)
    if suite_path.is_file():
        return [load_eval_case(suite_path)]
    if not suite_path.exists():
        raise FileNotFoundError(f"Evaluation suite not found: {suite_path}")
    cases: list[EvalCase] = []
    for case_path in sorted(suite_path.rglob("*.json")):
        if case_path.name.startswith("_"):
            continue
        cases.append(load_eval_case(case_path))
    if not cases:
        raise EvalCaseValidationError(f"No evaluation cases found in {suite_path}")
    return cases


def parse_eval_case(payload: dict[str, Any], *, source: Path | None = None) -> EvalCase:
    """Parse a dict into an :class:`EvalCase` without legacy compatibility."""

    if not isinstance(payload, dict):
        raise EvalCaseValidationError("Eval case must be a JSON object")
    case_id = _required_text(payload, "id", source)
    module = _literal(
        payload,
        "module",
        {"planning", "context", "memory", "security", "tool"},
        source,
    )
    case_type = _literal(payload, "type", {"task", "scenario"}, source)
    prompt = str(payload.get("prompt") or "")
    steps = [_parse_step(item, source=source) for item in _list(payload.get("steps"))]
    setup = [_parse_step(item, source=source) for item in _list(payload.get("setup"))]
    checks = [_parse_check(item, source=source) for item in _list(payload.get("checks"))]
    metrics = [str(item) for item in _list(payload.get("metrics")) if str(item)]
    tags = [str(item) for item in _list(payload.get("tags")) if str(item)]
    expected = payload.get("expected") or {}
    if not isinstance(expected, dict):
        raise EvalCaseValidationError(_where(source, "expected must be an object"))
    if case_type == "task" and not prompt.strip():
        raise EvalCaseValidationError(_where(source, "task case requires prompt"))
    if case_type == "scenario" and not steps:
        raise EvalCaseValidationError(_where(source, "scenario case requires steps"))
    return EvalCase(
        id=case_id,
        module=module,  # type: ignore[arg-type]
        fixture=_required_text(payload, "fixture", source),
        type=case_type,  # type: ignore[arg-type]
        prompt=prompt,
        steps=steps,
        setup=setup,
        checks=checks,
        metrics=metrics,
        expected=dict(expected),
        tags=tags,
        timeout_seconds=int(payload.get("timeout_seconds") or 120),
    )


def _parse_step(payload: Any, *, source: Path | None) -> EvalStep:
    if not isinstance(payload, dict):
        raise EvalCaseValidationError(_where(source, "step must be an object"))
    kind = _literal(
        payload,
        "kind",
        {"prompt", "restart", "modify_file", "verify", "inspect", "copy"},
        source,
    )
    # "copy" is accepted in setup as a convenience alias for fixture material.
    if kind == "copy":
        kind = "modify_file"
    return EvalStep(
        kind=kind,  # type: ignore[arg-type]
        text=str(payload.get("text") or ""),
        path=str(payload.get("path") or ""),
        source=str(payload.get("source") or ""),
        content=payload.get("content"),
        check=payload.get("check"),
    )


def _parse_check(payload: Any, *, source: Path | None) -> EvalCheck:
    if not isinstance(payload, dict):
        raise EvalCaseValidationError(_where(source, "check must be an object"))
    kind = _required_text(payload, "kind", source)
    options = {key: value for key, value in payload.items() if key != "kind"}
    return EvalCheck(kind=kind, options=options)


def _required_text(
    payload: dict[str, Any],
    key: str,
    source: Path | None,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EvalCaseValidationError(_where(source, f"{key} is required"))
    return value.strip()


def _literal(
    payload: dict[str, Any],
    key: str,
    allowed: set[str],
    source: Path | None,
) -> str:
    value = _required_text(payload, key, source)
    if value not in allowed:
        raise EvalCaseValidationError(
            _where(source, f"{key} must be one of {sorted(allowed)}")
        )
    return value


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    raise EvalCaseValidationError("field must be a list")


def _where(source: Path | None, message: str) -> str:
    return f"{source}: {message}" if source else message


__all__ = [
    "EvalCaseValidationError",
    "load_eval_case",
    "load_eval_suite",
    "parse_eval_case",
]
