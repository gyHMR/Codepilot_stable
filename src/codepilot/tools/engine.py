from __future__ import annotations

# 新手导读：engine.py 是工具调用的统一安全执行管线。
# 关注点：按顺序阅读 schema 校验、权限/审批、工具执行、结果防护。

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Pattern

from codepilot.protocols import TextContent

from .authoring import (
    AgentToolResult,
    AgentToolUpdateCallback,
    ToolMetadata,
    ToolRegistry,
    ToolResultStatus,
    ToolRuntimeRequest,
    ToolRuntimeResult,
)
from .policy import (
    ApprovalProvider,
    DeferredApprovalProvider,
    PermissionPolicy,
    ToolDecision,
    ToolRequest,
)

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

@dataclass(frozen=True)
class _RedactionRule:
    name: str
    pattern: Pattern[str]
    replacement: str


_SECRET_RULES: tuple[_RedactionRule, ...] = (
    _RedactionRule(
        "private_key",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED_SECRET]",
    ),
    _RedactionRule(
        "secret_assignment",
        re.compile(
            r"(?i)\b(api[_-]?key|token|secret|password|credential|cookie)\s*[:=]\s*([^\s,;]+)"
        ),
        r"\1=[REDACTED_SECRET]",
    ),
    _RedactionRule(
        "openai_key",
        re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
        "[REDACTED_SECRET]",
    ),
    _RedactionRule(
        "github_token",
        re.compile(r"\b(?:ghp_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
        "[REDACTED_SECRET]",
    ),
    _RedactionRule(
        "aws_access_key",
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "[REDACTED_SECRET]",
    ),
)

_PII_RULES: tuple[_RedactionRule, ...] = (
    _RedactionRule(
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
)

_PROMPT_INJECTION_PATTERNS: tuple[tuple[str, Pattern[str]], ...] = (
    (
        "ignore_previous_instructions",
        re.compile(r"\bignore\s+(?:all\s+)?previous\s+instructions\b", re.IGNORECASE),
    ),
    (
        "disregard_previous_instructions",
        re.compile(r"\bdisregard\s+(?:all\s+)?previous\s+instructions\b", re.IGNORECASE),
    ),
    (
        "reveal_system_prompt",
        re.compile(r"\b(?:system prompt|developer message)\b", re.IGNORECASE),
    ),
    (
        "dangerous_command_instruction",
        re.compile(r"\b(?:run\s+delete|execute\s+rm|delete_database)\b", re.IGNORECASE),
    ),
)


@dataclass(frozen=True)
class ToolResultGuard:
    """Redacts sensitive result text and labels output trust."""

    def apply(
        self,
        result: AgentToolResult,
        *,
        metadata: ToolMetadata | None = None,
    ) -> AgentToolResult:
        findings: list[str] = []
        redacted = False
        prompt_injection_suspected = False

        for block in result.content:
            if not isinstance(block, TextContent):
                continue
            guarded_text, block_redacted, block_findings = _redact_text(block.text)
            block_prompt_findings = _prompt_injection_findings(guarded_text)
            if block_redacted:
                redacted = True
            findings.extend(block_findings)
            findings.extend(block_prompt_findings)
            if block_prompt_findings:
                prompt_injection_suspected = True
            block.text = guarded_text

        findings = _unique(findings)
        output_trust = _output_trust(
            metadata,
            prompt_injection_suspected=prompt_injection_suspected,
        )
        result.metadata["result_guard"] = {
            "redacted": redacted,
            "findings": findings,
            "prompt_injection_suspected": prompt_injection_suspected,
            "output_trust": output_trust,
        }
        result.metadata["output_trust"] = output_trust
        return result


DEFAULT_TOOL_RESULT_GUARD = ToolResultGuard()


def apply_result_guard(
    result: AgentToolResult,
    *,
    metadata: ToolMetadata | None = None,
) -> AgentToolResult:
    """Redact sensitive text and mark untrusted tool output before model reuse."""
    return DEFAULT_TOOL_RESULT_GUARD.apply(result, metadata=metadata)


def _redact_text(text: str) -> tuple[str, bool, list[str]]:
    redacted = False
    findings: list[str] = []
    guarded = text
    for rule in (*_SECRET_RULES, *_PII_RULES):
        guarded, count = rule.pattern.subn(rule.replacement, guarded)
        if count:
            redacted = True
            findings.append(rule.name)
    return guarded, redacted, findings


def _prompt_injection_findings(text: str) -> list[str]:
    return [name for name, pattern in _PROMPT_INJECTION_PATTERNS if pattern.search(text)]


def _output_trust(
    metadata: ToolMetadata | None,
    *,
    prompt_injection_suspected: bool,
) -> str:
    if prompt_injection_suspected:
        return "untrusted"

    configured = None
    if metadata is not None:
        configured = metadata.extra.get("output_trust")
    if configured in {"trusted", "untrusted"}:
        return str(configured)

    if metadata is not None and (
        metadata.category in {"mcp", "extension"} or metadata.network_access
    ):
        return "untrusted"
    return "trusted"


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values

def _tool_result(
    message: str,
    *,
    status: ToolResultStatus,
    error_code: str | None = None,
    details: dict[str, Any] | None = None,
    is_error: bool = True,
) -> AgentToolResult:
    merged_details = {"status": status, **(details or {})}
    return AgentToolResult(
        content=[TextContent(text=message)],
        details=merged_details,
        is_error=is_error,
        status=status,
        error_code=error_code,
    )


def _sync_result_status(
    request: ToolRuntimeRequest,
    result: AgentToolResult,
    status: ToolResultStatus,
    *,
    approved: bool,
    approval_id: str | None = None,
) -> AgentToolResult:
    result.tool_call_id = request.tool_call_id
    result.tool_name = request.name
    result.status = status
    result.approved = approved
    result.approval_id = approval_id
    if result.details is None:
        result.details = {}
    if isinstance(result.details, dict):
        result.details["status"] = status
        if approval_id is not None:
            result.details["approval_id"] = approval_id
    return result


async def _maybe_await(value: Any) -> Any:
    if asyncio.isfuture(value) or asyncio.iscoroutine(value):
        return await value
    return value


def _permission_record(
    decision: ToolDecision,
    *,
    granted_approval_id: str | None,
) -> dict[str, Any]:
    if granted_approval_id is None:
        return {
            "decision": decision.kind,
            "reason": decision.reason,
            **decision.details,
        }
    return {
        "decision": "allow",
        "reason": "approved_by_user",
        "approval_id": granted_approval_id,
        "policy_decision": decision.kind,
        "policy_reason": decision.reason,
        **decision.details,
    }


@dataclass
class ToolRuntime:
    """工具运行时：将注册表、权限策略和审批提供者组合为统一的执行引擎。"""
    registry: ToolRegistry                                      # 工具注册表
    permission_policy: PermissionPolicy = field(default_factory=PermissionPolicy)  # 权限策略
    approval_provider: ApprovalProvider = field(default_factory=DeferredApprovalProvider)  # 审批提供者
    schema_validator: SchemaValidator = field(default_factory=SchemaValidator)  # 参数校验器
    result_guard: ToolResultGuard = field(default_factory=ToolResultGuard)  # 结果防护器

    async def execute(
        self,
        request: ToolRuntimeRequest,
        *,
        approval_id: str | None = None,
        signal: Any | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> ToolRuntimeResult:
        """执行工具调用：权限检查 → 参数校验 → 审批 → 执行 → 结果防护。"""
        return await self._execute(
            request,
            signal=signal,
            on_update=on_update,
            granted_approval_id=approval_id,
        )

    async def _execute(
        self,
        request: ToolRuntimeRequest,
        *,
        signal: Any | None,
        on_update: AgentToolUpdateCallback | None,
        granted_approval_id: str | None,
    ) -> ToolRuntimeResult:
        tool = self.registry.get(request.name)
        if tool is None:
            result = _tool_result(
                f"Tool {request.name} not found",
                status="error",
                error_code="tool_not_found",
                details={"tool": request.name, "reason": "tool_not_found"},
            )
            result.tool_call_id = request.tool_call_id
            result.tool_name = request.name
            result.approved = False
            return ToolRuntimeResult(
                result=result,
                status="error",
                is_error=True,
                approved=False,
            )

        metadata = self.registry.metadata_for(request.name)
        decision = self.permission_policy.decide(
            ToolRequest(
                name=request.name,
                params=request.params,
                source=request.source,
                metadata=metadata,
            )
        )
        if decision.denied:
            return self._blocked_result(request, decision)

        permission_record = _permission_record(
            decision,
            granted_approval_id=granted_approval_id,
        )

        validation = self.schema_validator.validate(tool.parameters, request.params)
        if not validation.valid:
            result = _tool_result(
                "Tool arguments failed schema validation: "
                + "; ".join(validation.errors),
                status="error",
                error_code="invalid_tool_arguments",
                details={
                    "tool": request.name,
                    "reason": "schema_validation_failed",
                    "errors": list(validation.errors),
                },
            )
            result.tool_call_id = request.tool_call_id
            result.tool_name = request.name
            result.approved = False
            result.metadata["permission_decision"] = permission_record
            result.metadata["schema_validation"] = {
                "valid": False,
                "errors": list(validation.errors),
            }
            return ToolRuntimeResult(
                result=result,
                status="error",
                is_error=True,
                approved=False,
            )

        approval_id: str | None = None
        if decision.requires_approval:
            if granted_approval_id is not None:
                approval_id = granted_approval_id
            else:
                approval = await self.approval_provider.request_approval(
                    request,
                    metadata,
                    decision,
                )
                approval_id = approval.approval_id
                if not approval.approved:
                    if not getattr(approval, "deferred", False):
                        result = _tool_result(
                            approval.reason or "Tool execution denied by user",
                            status="denied",
                            error_code="approval_denied",
                            details={
                                "tool": request.name,
                                "reason": approval.reason or decision.reason,
                                "approval_id": approval.approval_id,
                                "policy_reason": decision.reason,
                                "decision": "deny",
                                **decision.details,
                            },
                        )
                        result.tool_call_id = request.tool_call_id
                        result.tool_name = request.name
                        result.approved = False
                        result.approval_id = approval.approval_id
                        result.metadata["permission_decision"] = permission_record
                        return ToolRuntimeResult(
                            result=result,
                            status="denied",
                            is_error=True,
                            approved=False,
                            approval_id=approval.approval_id,
                        )
                    result = _tool_result(
                        approval.reason or "Tool execution requires approval",
                        status="approval_required",
                        error_code="approval_required",
                        details={
                            "tool": request.name,
                            "reason": decision.reason,
                            "approval_id": approval.approval_id,
                            "policy_reason": decision.reason,
                            "decision": decision.kind,
                            **decision.details,
                        },
                    )
                    result.tool_call_id = request.tool_call_id
                    result.tool_name = request.name
                    result.approved = False
                    result.approval_id = approval.approval_id
                    result.metadata["permission_decision"] = permission_record
                    return ToolRuntimeResult(
                        result=result,
                        status="approval_required",
                        is_error=True,
                        approved=False,
                        approval_id=approval.approval_id,
                    )

        try:
            started_at = time.monotonic()
            value = tool.execute(request.tool_call_id, request.params, signal, on_update)
            result = await _maybe_await(value)
            if approval_id is not None:
                result.approved = True
                result.approval_id = approval_id
            if not isinstance(result.details, dict):
                result.details = {"tool_details": result.details}
            result.details.setdefault(
                "permission",
                permission_record,
            )
            result.metadata.setdefault(
                "permission_decision",
                permission_record,
            )
            result.metadata.setdefault(
                "duration_ms",
                int((time.monotonic() - started_at) * 1000),
            )
            result = self.result_guard.apply(result, metadata=metadata)
            status: ToolResultStatus = result.status
            if status == "success" and result.is_error:
                status = "error"
            _sync_result_status(
                request,
                result,
                status,
                approved=result.approved,
                approval_id=approval_id,
            )
            return ToolRuntimeResult(
                result=result,
                status=status,
                is_error=bool(result.is_error),
                approved=result.approved,
                approval_id=approval_id,
            )
        except Exception as exc:
            result = _tool_result(
                str(exc),
                status="error",
                error_code="tool_exception",
                details={"tool": request.name, "reason": "tool_exception", "error_kind": type(exc).__name__},
            )
            result.approved = True
            result.approval_id = approval_id
            result.metadata.setdefault(
                "permission_decision",
                permission_record,
            )
            result.metadata.setdefault(
                "duration_ms",
                int((time.monotonic() - started_at) * 1000),
            )
            result = self.result_guard.apply(result, metadata=metadata)
            result.tool_call_id = request.tool_call_id
            result.tool_name = request.name
            return ToolRuntimeResult(
                result=result,
                status="error",
                is_error=True,
                approved=result.approved,
                approval_id=approval_id,
            )

    @staticmethod
    def _blocked_result(request: ToolRuntimeRequest, decision: ToolDecision) -> ToolRuntimeResult:
        result = _tool_result(
            decision.reason or "Tool execution was blocked",
            status="denied",
            error_code=decision.reason or "tool_denied",
            details={
                "tool": request.name,
                "reason": decision.reason,
                "policy_reason": decision.reason,
                **decision.details,
            },
        )
        result.tool_call_id = request.tool_call_id
        result.tool_name = request.name
        result.approved = False
        result.metadata["permission_decision"] = {
            "decision": decision.kind,
            "reason": decision.reason,
            **decision.details,
        }
        return ToolRuntimeResult(
            result=result,
            status="denied",
            is_error=True,
            approved=False,
        )
