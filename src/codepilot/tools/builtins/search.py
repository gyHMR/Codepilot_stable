from __future__ import annotations

"""Canonical workspace search tools."""

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..codecs import DataclassCodec
from ..contracts import ToolExecutionContext, ToolHandlerError, ToolRegistration, ToolSpec
from ..results import TextContent
from ..sandbox import WorkspaceSandbox
from ..security import (
    ConcurrencyPolicy,
    OutputLimits,
    OutputTrustPolicy,
    TimeoutPolicy,
    ToolAccessRequest,
    ToolAccessResolution,
    ToolEffect,
    ToolPolicy,
    ToolResource,
)

_DRAFT = "https://json-schema.org/draft/2020-12/schema"


@dataclass(frozen=True)
class GrepInput:
    pattern: str
    path: str = "."
    glob: str = "**/*"
    max_results: int = 200
    case_sensitive: bool = True


@dataclass(frozen=True)
class FindInput:
    pattern: str
    path: str = "."
    max_results: int = 500


@dataclass(frozen=True)
class SearchOutput:
    text: str
    details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


_OUTPUT_SCHEMA = {
    "$schema": _DRAFT,
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "details": {"type": "object"},
        "metadata": {"type": "object"},
    },
    "required": ["text", "details", "metadata"],
    "additionalProperties": False,
}


def create_search_registrations(
    sandbox: WorkspaceSandbox,
    *,
    allow: Callable[[str], bool],
) -> list[ToolRegistration]:
    configs = (
        ("grep", GrepInput, _grep_schema(), "Search UTF-8 workspace files with a regular expression and return file:line matches."),
        ("find", FindInput, _find_schema(), "Find workspace files whose relative paths match one glob pattern."),
    )
    result: list[ToolRegistration] = []
    for name, input_type, schema, description in configs:
        if not allow(name):
            continue
        result.append(
            ToolRegistration(
                version="1.0.0",
                implementation_version="2",
                spec=ToolSpec(name, description, schema, _OUTPUT_SCHEMA),
                category="search",
                source="builtin",
                owner="codepilot.builtin",
                policy=_policy(),
                input_codec=DataclassCodec(input_type, schema),
                output_codec=DataclassCodec(SearchOutput, _OUTPUT_SCHEMA),
                handler=_SearchHandler(sandbox, name),
                renderer=_Renderer(),
                access_resolver=_Resolver(sandbox, name),
            )
        )
    return result


@dataclass(frozen=True)
class _Resolver:
    sandbox: WorkspaceSandbox
    name: str

    def resolve(self, input, request):
        _ = request
        root = self.sandbox.ensure_readable_path(self.sandbox.resolve_path(input.path))
        return ToolAccessResolution(
            input=input,
            access=ToolAccessRequest(
                actions=(self.name,),
                resources=(ToolResource("workspace:///" + self.sandbox.relative_path(root)),),
                effects=frozenset({"filesystem_read"}),
                risk="low",
                reason=f"Search workspace path {input.path}",
                safe_preview={"path": input.path, "pattern": input.pattern},
            ),
        )


@dataclass(frozen=True)
class _SearchHandler:
    sandbox: WorkspaceSandbox
    name: str

    async def __call__(self, input, context: ToolExecutionContext) -> SearchOutput:
        context.cancellation.raise_if_cancelled()
        root = self.sandbox.ensure_readable_path(self.sandbox.resolve_path(input.path))
        if not root.exists():
            raise ToolHandlerError(f"{self.name}.path_missing", f"Search path not found: {input.path}")
        output = self._grep(root, input) if self.name == "grep" else self._find(root, input)
        context.effects.report(
            ToolEffect(
                kind="filesystem_read",
                resource=ToolResource("workspace:///" + self.sandbox.relative_path(root)),
                operation=self.name,
                status="completed",
                certainty="observed",
            )
        )
        return output

    def _grep(self, root: Path, input: GrepInput) -> SearchOutput:
        try:
            regex = re.compile(input.pattern, 0 if input.case_sensitive else re.IGNORECASE)
        except re.error as exc:
            raise ToolHandlerError("grep.invalid_pattern", str(exc)) from exc
        matches: list[str] = []
        scanned = 0
        for path in _files(root):
            relative = self.sandbox.relative_path(path)
            if not _glob_matches(relative, input.glob):
                continue
            try:
                safe = self.sandbox.ensure_readable_path(path)
                lines = safe.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError, ValueError):
                continue
            scanned += 1
            for number, line in enumerate(lines, start=1):
                if regex.search(line):
                    matches.append(f"{relative}:{number}:{line}")
                    if len(matches) >= input.max_results:
                        return SearchOutput(
                            text="\n".join(matches),
                            details={"match_count": len(matches), "scanned_files": scanned},
                            metadata={"truncated": True},
                        )
        return SearchOutput(
            text="\n".join(matches),
            details={"match_count": len(matches), "scanned_files": scanned},
            metadata={"truncated": False},
        )

    def _find(self, root: Path, input: FindInput) -> SearchOutput:
        matches: list[str] = []
        for path in _files(root):
            try:
                self.sandbox.ensure_readable_path(path)
            except ValueError:
                continue
            relative = self.sandbox.relative_path(path)
            scoped = path.relative_to(root).as_posix() if root.is_dir() else path.name
            if _glob_matches(scoped, input.pattern):
                matches.append(relative)
                if len(matches) >= input.max_results:
                    return SearchOutput(
                        text="\n".join(matches),
                        details={"match_count": len(matches)},
                        metadata={"truncated": True},
                    )
        return SearchOutput(
            text="\n".join(matches),
            details={"match_count": len(matches)},
            metadata={"truncated": False},
        )


class _Renderer:
    def render(self, data):
        return (TextContent(text=str(data["text"])),)


def _files(root: Path):
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _glob_matches(path: str, pattern: str) -> bool:
    patterns = [pattern]
    if pattern.startswith("**/"):
        patterns.append(pattern[3:])
    return any(fnmatch.fnmatch(path, item) or Path(path).match(item) for item in patterns)


def _policy() -> ToolPolicy:
    return ToolPolicy(
        allowed_modes=frozenset({"plan", "execute"}),
        declared_effects=frozenset({"filesystem_read"}),
        required_permissions=frozenset({"workspace.read"}),
        base_risk="low",
        approval="never",
        timeout=TimeoutPolicy(15_000, 30_000),
        concurrency=ConcurrencyPolicy(mode="parallel"),
        output_limits=OutputLimits(),
        output_trust=OutputTrustPolicy(),
    )


def _grep_schema() -> dict[str, Any]:
    return {"$schema": _DRAFT, "type": "object", "properties": {"pattern": {"type": "string", "minLength": 1}, "path": {"type": "string", "default": "."}, "glob": {"type": "string", "default": "**/*"}, "max_results": {"type": "integer", "minimum": 1, "maximum": 10_000, "default": 200}, "case_sensitive": {"type": "boolean", "default": True}}, "required": ["pattern"], "additionalProperties": False}


def _find_schema() -> dict[str, Any]:
    return {"$schema": _DRAFT, "type": "object", "properties": {"pattern": {"type": "string", "minLength": 1}, "path": {"type": "string", "default": "."}, "max_results": {"type": "integer", "minimum": 1, "maximum": 10_000, "default": 500}}, "required": ["pattern"], "additionalProperties": False}


__all__ = ["create_search_registrations"]
