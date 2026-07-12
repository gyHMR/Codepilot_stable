from __future__ import annotations

"""Canonical workspace file tools."""

import hashlib
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
_AUTO_APPROVE_MAX_FILES = 20
_AUTO_APPROVE_MAX_BYTES = 500_000


@dataclass(frozen=True)
class LsInput:
    path: str = "."
    max_entries: int = 100


@dataclass(frozen=True)
class ReadInput:
    path: str
    max_chars: int = 20_000
    offset: int = 1
    limit: int = 200


@dataclass(frozen=True)
class WriteInput:
    path: str
    content: str
    overwrite: bool = True


@dataclass(frozen=True)
class EditInput:
    path: str
    old_text: str
    new_text: str
    replace_all: bool = False
    occurrence_index: int | None = None
    expected_occurrences: int | None = None
    expected_file_hash: str | None = None


@dataclass(frozen=True)
class ApplyPatchInput:
    edits: list[dict[str, Any]]


@dataclass(frozen=True)
class FileOutput:
    text: str
    details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    affected_paths: list[str] = field(default_factory=list)
    workspace_changed: bool = False
    diff_summary: str | None = None


_OUTPUT_SCHEMA = {
    "$schema": _DRAFT,
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "details": {"type": "object"},
        "metadata": {"type": "object"},
        "affected_paths": {"type": "array", "items": {"type": "string"}},
        "workspace_changed": {"type": "boolean"},
        "diff_summary": {"type": ["string", "null"]},
    },
    "required": [
        "text",
        "details",
        "metadata",
        "affected_paths",
        "workspace_changed",
        "diff_summary",
    ],
    "additionalProperties": False,
}


def create_file_registrations(
    sandbox: WorkspaceSandbox,
    *,
    allow: Callable[[str], bool],
    edit_require_unique_match: bool = True,
) -> list[ToolRegistration]:
    configs = (
        ("ls", LsInput, _ls_schema(), False, "List one workspace directory with names and file sizes."),
        ("read", ReadInput, _read_schema(), False, "Read a UTF-8 workspace file with line pagination and truncation metadata."),
        ("write", WriteInput, _write_schema(), True, "Create or replace one UTF-8 workspace file and report whether content changed."),
        ("edit", EditInput, _edit_schema(), True, "Replace exact text in one UTF-8 workspace file with occurrence and hash guards."),
        ("apply_patch", ApplyPatchInput, _patch_schema(), True, "Validate and atomically apply one to twenty exact text replacements."),
    )
    registrations: list[ToolRegistration] = []
    for name, input_type, input_schema, mutating, description in configs:
        if not allow(name):
            continue
        input_codec = DataclassCodec(input_type, input_schema)
        output_codec = DataclassCodec(FileOutput, _OUTPUT_SCHEMA)
        registrations.append(
            ToolRegistration(
                version="1.0.0",
                implementation_version="2",
                spec=ToolSpec(name, description, input_schema, _OUTPUT_SCHEMA),
                category="filesystem",
                source="builtin",
                owner="codepilot.builtin",
                policy=_policy(mutating),
                input_codec=input_codec,
                output_codec=output_codec,
                handler=_FileHandler(
                    sandbox=sandbox,
                    name=name,
                    unique_edit=edit_require_unique_match,
                ),
                renderer=_Renderer(),
                access_resolver=_Resolver(sandbox, name, mutating),
            )
        )
    return registrations


@dataclass(frozen=True)
class _Resolver:
    sandbox: WorkspaceSandbox
    name: str
    mutating: bool

    def resolve(self, input, request):
        _ = request
        paths = (
            [str(item.get("path", "")) for item in input.edits]
            if self.name == "apply_patch"
            else [str(input.path)]
        )
        resources: list[ToolResource] = []
        for raw in paths:
            target = self.sandbox.resolve_path(raw)
            target = (
                self.sandbox.ensure_mutable_path(target)
                if self.mutating
                else self.sandbox.ensure_readable_path(target)
            )
            resources.append(_resource(self.sandbox, target))
        effects = (
            frozenset({"filesystem_read", "filesystem_write"})
            if self.mutating
            else frozenset({"filesystem_read"})
        )
        file_count, estimated_bytes = _mutation_size(self.name, input)
        bulk = self.mutating and (
            file_count > _AUTO_APPROVE_MAX_FILES
            or estimated_bytes > _AUTO_APPROVE_MAX_BYTES
        )
        return ToolAccessResolution(
            input=input,
            access=ToolAccessRequest(
                actions=(f"{self.name}.bulk" if bulk else self.name,),
                resources=tuple(resources),
                effects=effects,
                risk="medium" if bulk else "low",
                reason=f"{self.name} workspace path(s)",
                safe_preview={
                    "paths": paths,
                    "file_count": file_count,
                    "estimated_bytes": estimated_bytes,
                    "operation_profile": (
                        "bulk_write"
                        if bulk
                        else "workspace_write" if self.mutating else "workspace_read"
                    ),
                },
                approval_scopes=(
                    frozenset({"once"})
                    if bulk
                    else frozenset({"once", "session", "project"})
                ),
            ),
        )


@dataclass(frozen=True)
class _FileHandler:
    sandbox: WorkspaceSandbox
    name: str
    unique_edit: bool

    async def __call__(self, input, context: ToolExecutionContext) -> FileOutput:
        context.cancellation.raise_if_cancelled()
        if self.name == "ls":
            return self._ls(input, context)
        if self.name == "read":
            return self._read(input, context)
        if self.name == "write":
            return self._write(input, context)
        if self.name == "edit":
            return self._edit(input, context)
        return self._patch(input, context)

    def _ls(self, input: LsInput, context: ToolExecutionContext) -> FileOutput:
        target = self.sandbox.ensure_readable_path(self.sandbox.resolve_path(input.path))
        if not target.is_dir():
            raise ToolHandlerError("ls.not_directory", f"Directory not found: {input.path}")
        entries = sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        shown = entries[: input.max_entries]
        lines = [
            f"{item.name}/" if item.is_dir() else f"{item.name}\t{item.stat().st_size} bytes"
            for item in shown
        ]
        truncated = len(entries) > len(shown)
        if truncated:
            lines.append(f"... {len(entries) - len(shown)} more entries")
        self._effect(context, target, "filesystem_read", "list directory")
        relative = self.sandbox.relative_path(target) or "."
        return FileOutput(
            text="\n".join(lines),
            details={"path": relative, "entry_count": len(entries)},
            metadata={"truncated": truncated},
        )

    def _read(self, input: ReadInput, context: ToolExecutionContext) -> FileOutput:
        target = self.sandbox.ensure_readable_path(self.sandbox.resolve_path(input.path))
        if not target.is_file():
            raise ToolHandlerError("read.not_file", f"File not found: {input.path}")
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolHandlerError("read.not_utf8", f"File is not valid UTF-8: {input.path}") from exc
        lines = text.splitlines(keepends=True)
        start = min(len(lines), input.offset - 1)
        selected = lines[start : start + input.limit]
        rendered = "".join(selected)
        char_truncated = len(rendered) > input.max_chars
        if char_truncated:
            rendered = rendered[: input.max_chars]
            if "\n" in rendered:
                rendered = rendered[: rendered.rfind("\n") + 1]
        line_truncated = start + len(selected) < len(lines)
        truncated = char_truncated or line_truncated
        self._effect(context, target, "filesystem_read", "read file")
        relative = self.sandbox.relative_path(target)
        return FileOutput(
            text=rendered,
            details={
                "path": relative,
                "offset": input.offset,
                "line_count": len(lines),
                "returned_lines": len(rendered.splitlines()),
            },
            metadata={
                "truncated": truncated,
                "output_quality": {
                    "truncated": truncated,
                    "original_chars": len(text),
                    "returned_chars": len(rendered),
                    "reliable_for_reasoning": not truncated,
                },
            },
        )

    def _write(self, input: WriteInput, context: ToolExecutionContext) -> FileOutput:
        target = self.sandbox.ensure_mutable_path(self.sandbox.resolve_path(input.path))
        if target.exists() and not input.overwrite:
            raise ToolHandlerError("write.exists", f"File already exists: {input.path}")
        previous = target.read_text(encoding="utf-8") if target.is_file() else None
        changed = previous != input.content
        if changed:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(input.content, encoding="utf-8", newline="\n")
        return self._mutation_output(target, changed, context, "write file")

    def _edit(self, input: EditInput, context: ToolExecutionContext) -> FileOutput:
        target = self.sandbox.ensure_mutable_path(self.sandbox.resolve_path(input.path))
        if not target.is_file():
            raise ToolHandlerError("edit.not_file", f"File not found: {input.path}")
        text = target.read_text(encoding="utf-8")
        if input.expected_file_hash and _hash_text(text) != input.expected_file_hash:
            raise ToolHandlerError("edit.hash_mismatch", "File content hash no longer matches")
        count = text.count(input.old_text)
        if input.expected_occurrences is not None and count != input.expected_occurrences:
            raise ToolHandlerError("edit.occurrence_mismatch", f"Expected {input.expected_occurrences} matches, found {count}")
        if count == 0:
            raise ToolHandlerError("edit.no_match", "old_text was not found")
        if input.occurrence_index is not None:
            updated = _replace_occurrence(text, input.old_text, input.new_text, input.occurrence_index)
            replacements = 1
        elif input.replace_all:
            updated = text.replace(input.old_text, input.new_text)
            replacements = count
        else:
            if self.unique_edit and count != 1:
                raise ToolHandlerError("edit.match_not_unique", f"old_text matched {count} times")
            updated = text.replace(input.old_text, input.new_text, 1)
            replacements = 1
        changed = updated != text
        if changed:
            target.write_text(updated, encoding="utf-8", newline="\n")
        output = self._mutation_output(target, changed, context, "edit file")
        return FileOutput(**{**output.__dict__, "details": {"replacements": replacements}})

    def _patch(self, input: ApplyPatchInput, context: ToolExecutionContext) -> FileOutput:
        if not 1 <= len(input.edits) <= 20:
            raise ToolHandlerError("apply_patch.invalid_count", "edits must contain between 1 and 20 items")
        staged: dict[Path, str] = {}
        changed_paths: list[Path] = []
        for index, edit in enumerate(input.edits):
            path = str(edit.get("path", ""))
            old_text = edit.get("old_text")
            new_text = edit.get("new_text")
            if not isinstance(old_text, str) or not isinstance(new_text, str):
                raise ToolHandlerError("apply_patch.invalid_edit", f"edits[{index}] requires string old_text/new_text")
            target = self.sandbox.ensure_mutable_path(self.sandbox.resolve_path(path))
            if not target.is_file():
                raise ToolHandlerError("apply_patch.not_file", f"File not found: {path}")
            current = staged.get(target)
            if current is None:
                current = target.read_text(encoding="utf-8")
            count = current.count(old_text)
            if count != 1:
                raise ToolHandlerError("apply_patch.match_not_unique", f"edits[{index}] matched {count} times")
            staged[target] = current.replace(old_text, new_text, 1)
            if target not in changed_paths:
                changed_paths.append(target)
        for target in changed_paths:
            target.write_text(staged[target], encoding="utf-8", newline="\n")
            self._effect(context, target, "filesystem_read", "validate patch")
            self._effect(context, target, "filesystem_write", "apply patch")
        relatives = [self.sandbox.relative_path(path) for path in changed_paths]
        return FileOutput(
            text=f"Applied {len(input.edits)} edit(s) across {len(changed_paths)} file(s).",
            details={"edit_count": len(input.edits)},
            affected_paths=relatives,
            workspace_changed=True,
            diff_summary=f"updated {len(changed_paths)} file(s)",
        )

    def _mutation_output(
        self,
        target: Path,
        changed: bool,
        context: ToolExecutionContext,
        operation: str,
    ) -> FileOutput:
        self._effect(context, target, "filesystem_read", f"{operation} precondition")
        if changed:
            self._effect(context, target, "filesystem_write", operation)
        relative = self.sandbox.relative_path(target)
        return FileOutput(
            text=f"{'Updated' if changed else 'Unchanged'} {relative}",
            affected_paths=[relative] if changed else [],
            workspace_changed=changed,
            diff_summary=f"updated {relative}" if changed else None,
        )

    def _effect(
        self,
        context: ToolExecutionContext,
        target: Path,
        kind: str,
        operation: str,
    ) -> None:
        context.effects.report(
            ToolEffect(
                kind=kind,
                resource=_resource(self.sandbox, target),
                operation=operation,
                status="completed",
                certainty="observed",
            )
        )


class _Renderer:
    def render(self, data):
        return (TextContent(text=str(data["text"])),)


def _policy(mutating: bool) -> ToolPolicy:
    return ToolPolicy(
        allowed_modes=frozenset({"execute"} if mutating else {"plan", "execute"}),
        declared_effects=(
            frozenset({"filesystem_read", "filesystem_write"})
            if mutating
            else frozenset({"filesystem_read"})
        ),
        required_permissions=(
            frozenset({"workspace.read", "workspace.write"})
            if mutating
            else frozenset({"workspace.read"})
        ),
        base_risk="medium" if mutating else "low",
        approval="on_risk" if mutating else "never",
        timeout=TimeoutPolicy(15_000, 60_000),
        concurrency=(
            ConcurrencyPolicy(mode="serial", group="workspace_mutation")
            if mutating
            else ConcurrencyPolicy(mode="parallel")
        ),
        output_limits=OutputLimits(),
        output_trust=OutputTrustPolicy(),
    )


def _resource(sandbox: WorkspaceSandbox, target: Path) -> ToolResource:
    return ToolResource("workspace:///" + sandbox.relative_path(target))


def _mutation_size(name: str, input: object) -> tuple[int, int]:
    if name == "write":
        return 1, len(input.content.encode("utf-8"))
    if name == "edit":
        return 1, len(input.old_text.encode("utf-8")) + len(input.new_text.encode("utf-8"))
    if name == "apply_patch":
        return len(input.edits), sum(
            len(str(item.get("old_text", "")).encode("utf-8"))
            + len(str(item.get("new_text", "")).encode("utf-8"))
            for item in input.edits
        )
    return 1, 0


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _replace_occurrence(text: str, old: str, new: str, index: int) -> str:
    if index < 1:
        raise ToolHandlerError("edit.invalid_occurrence", "occurrence_index must be positive")
    start = -1
    cursor = 0
    for _ in range(index):
        start = text.find(old, cursor)
        if start < 0:
            raise ToolHandlerError("edit.occurrence_missing", f"Occurrence {index} was not found")
        cursor = start + len(old)
    return text[:start] + new + text[start + len(old) :]


def _ls_schema() -> dict[str, Any]:
    return {"$schema": _DRAFT, "type": "object", "properties": {"path": {"type": "string", "default": "."}, "max_entries": {"type": "integer", "minimum": 1, "maximum": 10_000, "default": 100}}, "additionalProperties": False}


def _read_schema() -> dict[str, Any]:
    return {"$schema": _DRAFT, "type": "object", "properties": {"path": {"type": "string"}, "max_chars": {"type": "integer", "minimum": 1, "default": 20_000}, "offset": {"type": "integer", "minimum": 1, "default": 1}, "limit": {"type": "integer", "minimum": 1, "default": 200}}, "required": ["path"], "additionalProperties": False}


def _write_schema() -> dict[str, Any]:
    return {"$schema": _DRAFT, "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string", "maxLength": 1_000_000}, "overwrite": {"type": "boolean", "default": True}}, "required": ["path", "content"], "additionalProperties": False}


def _edit_schema() -> dict[str, Any]:
    return {"$schema": _DRAFT, "type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string", "minLength": 1}, "new_text": {"type": "string"}, "replace_all": {"type": "boolean", "default": False}, "occurrence_index": {"type": ["integer", "null"], "minimum": 1}, "expected_occurrences": {"type": ["integer", "null"], "minimum": 0}, "expected_file_hash": {"type": ["string", "null"]}}, "required": ["path", "old_text", "new_text"], "additionalProperties": False}


def _patch_schema() -> dict[str, Any]:
    return {"$schema": _DRAFT, "type": "object", "properties": {"edits": {"type": "array", "minItems": 1, "maxItems": 20, "items": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string", "minLength": 1}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"], "additionalProperties": False}}}, "required": ["edits"], "additionalProperties": False}


__all__ = ["create_file_registrations"]
