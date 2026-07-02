from __future__ import annotations

"""Post-execution guard for tool result content."""

import re
from dataclasses import dataclass
from typing import Pattern

from codepilot.protocols import TextContent

from .contracts import AgentToolResult, ToolMetadata


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


__all__ = ["DEFAULT_TOOL_RESULT_GUARD", "ToolResultGuard", "apply_result_guard"]
