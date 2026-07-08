from __future__ import annotations

"""Tool result normalization, redaction, and trust metadata."""

import re
import time
from dataclasses import dataclass
from typing import Pattern

from codepilot.protocols import TextContent
from codepilot.protocols.tools import ToolResultStatus, ensure_tool_result_status

from .contracts import ToolMetadata, ToolResult

@dataclass(frozen=True)
class _RedactionRule:
    name: str
    pattern: Pattern[str]
    replacement: str


# Keep these broad enough for local agent logs, but not so broad that normal
# source text becomes unreadable. Each rule is deterministic and reports its
# finding in result metadata.
_SECRET_RULES = (
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
    _RedactionRule("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "[REDACTED_SECRET]"),
    _RedactionRule(
        "github_token",
        re.compile(r"\b(?:ghp_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
        "[REDACTED_SECRET]",
    ),
    _RedactionRule("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_SECRET]"),
)
_PII_RULES = (
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
_LARGE_OUTPUT_CHARS = 30_000


@dataclass(frozen=True)
class ToolResultPolicy:
    large_output_chars: int = _LARGE_OUTPUT_CHARS

    def normalize(
        self,
        result: ToolResult,
        *,
        tool_call_id: str,
        tool_name: str,
        metadata: ToolMetadata | None,
        approval_id: str | None = None,
        permission_decision: dict[str, object] | None = None,
        started_at: float | None = None,
    ) -> ToolResult:
        if not isinstance(result, ToolResult):
            result = ToolResult(
                content=[TextContent(text=str(result))],
                status="success",
            )
        result.tool_call_id = tool_call_id
        result.tool_name = tool_name
        result.status = _effective_status(result)
        result.is_error = result.status != "success"
        if approval_id is not None:
            result.approved = True
            result.approval_id = approval_id
        if permission_decision is not None:
            result.metadata.setdefault("permission_decision", dict(permission_decision))
        if started_at is not None:
            result.metadata.setdefault("duration_ms", int((time.monotonic() - started_at) * 1000))
        if not result.content:
            result.content = [TextContent(text="(no output)")]
            result.metadata.setdefault("output_quality", {"empty": True})
        self._sanitize(result, metadata=metadata)
        self._mark_large_output(result)
        return result

    def _sanitize(self, result: ToolResult, *, metadata: ToolMetadata | None) -> None:
        redacted = False
        findings: list[str] = []
        prompt_injection_suspected = False
        for block in result.content:
            if not isinstance(block, TextContent):
                continue
            text, changed, block_findings = _redact_text(
                block.text,
                preserve_workspace_source=_preserve_workspace_source(metadata),
            )
            prompt_findings = _prompt_injection_findings(text)
            block.text = text
            redacted = redacted or changed
            prompt_injection_suspected = prompt_injection_suspected or bool(prompt_findings)
            findings.extend(block_findings)
            findings.extend(prompt_findings)
        findings = _unique(findings)
        output_trust = _output_trust(metadata, prompt_injection_suspected=prompt_injection_suspected)
        result.metadata["result_guard"] = {
            "redacted": redacted,
            "findings": findings,
            "prompt_injection_suspected": prompt_injection_suspected,
            "output_trust": output_trust,
        }
        result.metadata["output_trust"] = output_trust

    def _mark_large_output(self, result: ToolResult) -> None:
        total = sum(len(block.text) for block in result.content if isinstance(block, TextContent))
        if total <= self.large_output_chars:
            result.metadata.setdefault("output_quality", {}).setdefault("truncated", False)
            return
        result.metadata["tool_artifact"] = {
            "type": "tool_artifact",
            "tool_call_id": result.tool_call_id,
            "tool_name": result.tool_name,
            "original_chars": total,
            "preview_chars": self.large_output_chars,
            "summary": _summarize_blocks(result),
        }
        result.metadata.setdefault("output_quality", {})["artifact_recommended"] = True


def _effective_status(result: ToolResult) -> ToolResultStatus:
    status = ensure_tool_result_status(result.status)
    if result.is_error and status == "success":
        return "error"
    return status


def _redact_text(
    text: str,
    *,
    preserve_workspace_source: bool,
) -> tuple[str, bool, list[str]]:
    rules = _workspace_source_redaction_rules() if preserve_workspace_source else (
        *_SECRET_RULES,
        *_PII_RULES,
    )
    redacted = False
    findings: list[str] = []
    guarded = text
    for rule in rules:
        guarded, count = rule.pattern.subn(rule.replacement, guarded)
        if count:
            redacted = True
            findings.append(rule.name)
    return guarded, redacted, findings


def _workspace_source_redaction_rules() -> tuple[_RedactionRule, ...]:
    return tuple(rule for rule in _SECRET_RULES if rule.name != "secret_assignment")


def _preserve_workspace_source(metadata: ToolMetadata | None) -> bool:
    return metadata is not None and metadata.name == "read" and metadata.category == "filesystem"


def _prompt_injection_findings(text: str) -> list[str]:
    return [name for name, pattern in _PROMPT_INJECTION_PATTERNS if pattern.search(text)]


def _output_trust(
    metadata: ToolMetadata | None,
    *,
    prompt_injection_suspected: bool,
) -> str:
    if prompt_injection_suspected:
        return "untrusted"
    configured = metadata.extra.get("output_trust") if metadata is not None else None
    if configured in {"trusted", "local", "untrusted", "sanitized"}:
        return str(configured)
    if metadata is not None and (
        metadata.category in {"mcp", "extension"} or metadata.network_access
    ):
        return "untrusted"
    return "local"


def _summarize_blocks(result: ToolResult) -> str:
    text = "\n".join(block.text for block in result.content if isinstance(block, TextContent))
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return first_line[:300] or f"{result.tool_name} produced {len(text)} characters"


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


__all__ = ["ToolResultPolicy"]
