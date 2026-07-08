from __future__ import annotations

"""Built-in filesystem tools: ls, read, write, edit, apply_patch."""

from dataclasses import dataclass
from typing import Any, Callable

from codepilot.protocols import TextContent
from codepilot.tools.contracts import ToolCallRequest, ToolDefinition, ToolResult
from codepilot.tools.registry import get_builtin_tool_metadata
from codepilot.tools.sandbox import WorkspaceSandbox, file_state_for_path


@dataclass(frozen=True)
class _PatchEdit:
    path: str
    old_text: str
    new_text: str


def create_file_tools(
    sandbox: WorkspaceSandbox,
    *,
    allow: Callable[[str], bool],
    edit_require_unique_match: bool = True,
) -> list[ToolDefinition]:
    tools: list[ToolDefinition] = []
    max_write_chars = 1_000_000

    async def ls_tool(request: ToolCallRequest, signal=None, on_update=None) -> ToolResult:
        _ = signal, on_update
        params = request.arguments
        path_text = str(params.get("path", "."))
        max_entries = int(params.get("max_entries", 100))
        target, error = _resolve_read_path(sandbox, path_text)
        if error is not None:
            return error
        if not target.exists():
            return _error_result(f"Path not found: {path_text}", "path_not_found")
        if not target.is_dir():
            return _error_result(f"Not a directory: {path_text}", "not_a_directory")
        items = sorted(target.iterdir(), key=lambda path: path.name)[:max_entries]
        lines = []
        for item in items:
            suffix = "/" if item.is_dir() else ""
            size = "-" if item.is_dir() else str(item.stat().st_size)
            lines.append(f"{item.name}{suffix}\t{size}")
        rel = target.relative_to(sandbox.root).as_posix()
        return ToolResult(
            content=[TextContent(text="\n".join(lines) if lines else "(empty)")],
            metadata={
                "read_paths": [rel],
                "output_quality": {"truncated": len(lines) >= max_entries},
            },
        )

    async def read_tool(request: ToolCallRequest, signal=None, on_update=None) -> ToolResult:
        _ = signal, on_update
        params = request.arguments
        path_text = str(params.get("path", ""))
        if not path_text:
            return _error_result("Missing path", "missing_path")
        max_chars = int(params.get("max_chars", 4000))
        offset = int(params.get("offset", 1))
        limit = params.get("limit")
        limit = int(limit) if limit is not None else None
        target, error = _resolve_read_path(sandbox, path_text)
        if error is not None:
            return error
        if not target.exists():
            return _error_result(f"Path not found: {path_text}", "path_not_found")
        if not target.is_file():
            return _error_result(f"Not a file: {path_text}", "not_a_file")
        try:
            raw = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return _error_result(
                f"File is not valid UTF-8 text: {path_text}",
                "invalid_utf8",
                metadata={"output_quality": _output_quality(decode_status="invalid_utf8", may_be_binary=True)},
            )
        lines = raw.splitlines()
        start_index = min(max(offset, 1) - 1, len(lines))
        end_index = len(lines) if limit is None else min(len(lines), start_index + max(limit, 1))
        selected = lines[start_index:end_index]
        rendered = "\n".join(
            f"{line_no}\t{line}"
            for line_no, line in enumerate(selected, start=start_index + 1)
        )
        char_truncated = len(rendered) > max_chars
        if char_truncated:
            rendered = rendered[:max_chars] + "\n...<truncated>..."
        truncated = char_truncated or end_index < len(lines)
        relative_path = target.relative_to(sandbox.root).as_posix()
        state = file_state_for_path(sandbox.root, relative_path)
        return ToolResult(
            content=[TextContent(text=rendered or "(empty)")],
            details={"file_state": state},
            metadata={
                "file_state": state,
                "read_paths": [relative_path],
                "start_line": start_index + 1 if selected else None,
                "end_line": end_index if selected else None,
                "total_lines": len(lines),
                "truncated": truncated,
                "char_truncated": char_truncated,
                "output_quality": _output_quality(
                    truncated=truncated,
                    original_chars=len(raw),
                    returned_chars=len(rendered),
                ),
            },
        )

    async def write_tool(request: ToolCallRequest, signal=None, on_update=None) -> ToolResult:
        _ = signal, on_update
        params = request.arguments
        path_text = str(params.get("path", ""))
        content = str(params.get("content", ""))
        overwrite = bool(params.get("overwrite", True))
        if not path_text:
            return _error_result("Missing path", "missing_path")
        if len(content) > max_write_chars:
            return _error_result("Content is too large", "content_too_large")
        target, error = _resolve_write_path(sandbox, path_text)
        if error is not None:
            return error
        if target.exists() and not target.is_file():
            return _error_result(f"Target is not a file: {path_text}", "target_not_file")
        if target.exists() and not overwrite:
            return _error_result(f"File exists: {path_text}", "file_exists")
        try:
            original = target.read_text(encoding="utf-8") if target.exists() else None
        except UnicodeDecodeError:
            return _error_result(f"Existing file is not valid UTF-8: {path_text}", "invalid_utf8")
        relative_path = target.relative_to(sandbox.root).as_posix()
        before_hash = _state_hash(sandbox.root, relative_path)
        if original == content:
            state = file_state_for_path(sandbox.root, relative_path)
            return ToolResult(
                content=[TextContent(text=f"File unchanged: {relative_path}")],
                affected_paths=[relative_path],
                workspace_changed=False,
                diff_summary="No content change",
                details={"changed": False, "file_state": state},
                metadata={"file_state": state, "change_evidence": _change_evidence("unchanged", relative_path, before_hash, _state_hash(sandbox.root, relative_path))},
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
        state = file_state_for_path(sandbox.root, relative_path)
        action = "created" if original is None else "updated"
        return ToolResult(
            content=[TextContent(text=f"Wrote file: {relative_path}")],
            affected_paths=[relative_path],
            workspace_changed=True,
            diff_summary=f"{action} {relative_path}: {len(original or '')} -> {len(content)} characters",
            details={"changed": True, "action": action, "file_state": state},
            metadata={
                "file_state": state,
                "change_evidence": _change_evidence(
                    "create" if original is None else "update",
                    relative_path,
                    before_hash,
                    str(state.get("sha256", "<missing>")),
                ),
            },
        )

    async def edit_tool(request: ToolCallRequest, signal=None, on_update=None) -> ToolResult:
        _ = signal, on_update
        params = request.arguments
        edit = _PatchEdit(
            path=str(params.get("path", "")),
            old_text=str(params.get("old_text", "")),
            new_text=str(params.get("new_text", "")),
        )
        replace_all = bool(params.get("replace_all", False))
        occurrence_index = params.get("occurrence_index")
        occurrence_index = int(occurrence_index) if occurrence_index is not None else None
        expected_occurrences = params.get("expected_occurrences")
        expected_occurrences = int(expected_occurrences) if expected_occurrences is not None else None
        expected_hash = params.get("expected_file_hash")
        return _apply_single_edit(
            sandbox,
            edit,
            replace_all=replace_all,
            occurrence_index=occurrence_index,
            expected_occurrences=expected_occurrences,
            expected_file_hash=str(expected_hash) if expected_hash is not None else None,
            require_unique_match=edit_require_unique_match,
            max_chars=max_write_chars,
            label="Edited file",
        )

    async def apply_patch_tool(request: ToolCallRequest, signal=None, on_update=None) -> ToolResult:
        _ = signal, on_update
        params = request.arguments
        raw_edits = params.get("edits")
        if not isinstance(raw_edits, list) or not raw_edits:
            return _error_result("edits must contain at least one edit", "invalid_patch")
        if len(raw_edits) > 20:
            return _error_result("apply_patch supports at most 20 edits", "patch_too_large")
        edits: list[_PatchEdit] = []
        for index, item in enumerate(raw_edits):
            if not isinstance(item, dict):
                return _error_result(f"edits[{index}] must be an object", "invalid_patch")
            edit = _PatchEdit(
                path=str(item.get("path", "")),
                old_text=str(item.get("old_text", "")),
                new_text=str(item.get("new_text", "")),
            )
            if not edit.path or edit.old_text == "":
                return _error_result(f"edits[{index}] requires path and old_text", "invalid_patch")
            edits.append(edit)
        prepared: list[tuple[_PatchEdit, Any, str, str, str]] = []
        for edit in edits:
            target, error = _resolve_write_path(sandbox, edit.path)
            if error is not None:
                return error
            if not target.exists() or not target.is_file():
                return _error_result(f"Path not found or not file: {edit.path}", "path_not_file")
            try:
                original = target.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return _error_result(f"File is not valid UTF-8: {edit.path}", "invalid_utf8")
            count = original.count(edit.old_text)
            if count != 1:
                return _error_result(
                    f"Patch for {edit.path} expected exactly one match, found {count}",
                    "patch_match_count",
                    metadata={"matches": count},
                )
            updated = original.replace(edit.old_text, edit.new_text, 1)
            rel = target.relative_to(sandbox.root).as_posix()
            before_hash = _state_hash(sandbox.root, rel)
            prepared.append((edit, target, rel, updated, before_hash))
        affected: list[str] = []
        evidences = []
        for _edit, target, rel, updated, before_hash in prepared:
            target.write_text(updated, encoding="utf-8", newline="\n")
            affected.append(rel)
            evidences.append(
                _change_evidence("update", rel, before_hash, _state_hash(sandbox.root, rel))
            )
        return ToolResult(
            content=[TextContent(text=f"Applied patch edits: {len(prepared)}")],
            affected_paths=affected,
            workspace_changed=True,
            diff_summary=f"applied {len(prepared)} patch edit(s)",
            details={"edits": len(prepared)},
            metadata={"change_evidence": evidences},
        )

    if allow("ls"):
        tools.append(_tool("ls", "List Directory", "列出目录内容，返回文件名和大小。", _ls_schema(), ls_tool))
    if allow("read"):
        tools.append(_tool("read", "Read File", "读取文本文件内容。", _read_schema(), read_tool))
    if allow("write"):
        tools.append(_tool("write", "Write File", "写入文本文件。", _write_schema(), write_tool))
    if allow("edit"):
        tools.append(_tool("edit", "Edit File", "按 old_text -> new_text 替换文件内容。", _edit_schema(), edit_tool))
    if allow("apply_patch"):
        tools.append(_tool("apply_patch", "Apply Patch", "按结构化 edits 对多个文件执行唯一匹配替换。", _apply_patch_schema(), apply_patch_tool))
    return tools


def _apply_single_edit(
    sandbox: WorkspaceSandbox,
    edit: _PatchEdit,
    *,
    replace_all: bool,
    occurrence_index: int | None,
    expected_occurrences: int | None,
    expected_file_hash: str | None,
    require_unique_match: bool,
    max_chars: int,
    label: str,
) -> ToolResult:
    if not edit.path:
        return _error_result("Missing path", "missing_path")
    if edit.old_text == "":
        return _error_result("old_text cannot be empty", "empty_old_text")
    if len(edit.old_text) + len(edit.new_text) > max_chars:
        return _error_result("Edit payload is too large", "content_too_large")
    target, error = _resolve_write_path(sandbox, edit.path)
    if error is not None:
        return error
    if not target.exists() or not target.is_file():
        return _error_result(f"Path not found or not file: {edit.path}", "path_not_file")
    try:
        original = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return _error_result(f"File is not valid UTF-8: {edit.path}", "invalid_utf8")
    relative_path = target.relative_to(sandbox.root).as_posix()
    state = file_state_for_path(sandbox.root, relative_path)
    before_hash = str(state.get("sha256", "<missing>"))
    if expected_file_hash is not None and before_hash != expected_file_hash:
        return _error_result(
            "File changed since it was read; read it again before editing",
            "stale_file",
            metadata={"file_state": state},
        )
    count = original.count(edit.old_text)
    if expected_occurrences is not None and count != expected_occurrences:
        return _error_result(f"Expected {expected_occurrences} matches, found {count}", "unexpected_match_count")
    if count == 0:
        return _error_result("No match found", "no_match")
    if not replace_all and count > 1 and occurrence_index is None and require_unique_match:
        return _error_result("Multiple matches found; refine old_text or use occurrence_index", "multiple_matches")
    if replace_all:
        updated = original.replace(edit.old_text, edit.new_text)
        replacements = count
    elif occurrence_index is not None:
        if occurrence_index < 1 or occurrence_index > count:
            return _error_result("occurrence_index is out of range", "occurrence_out_of_range")
        updated = _replace_nth(original, edit.old_text, edit.new_text, occurrence_index)
        replacements = 1
    else:
        updated = original.replace(edit.old_text, edit.new_text, 1)
        replacements = 1
    target.write_text(updated, encoding="utf-8", newline="\n")
    new_state = file_state_for_path(sandbox.root, relative_path)
    return ToolResult(
        content=[TextContent(text=f"{label}: {relative_path} (replacements={replacements})")],
        affected_paths=[relative_path],
        workspace_changed=updated != original,
        diff_summary=f"edited {relative_path}: {replacements} replacement(s)",
        details={"replacements": replacements, "file_state": new_state},
        metadata={
            "file_state": new_state,
            "change_evidence": _change_evidence(
                "update" if updated != original else "unchanged",
                relative_path,
                before_hash,
                str(new_state.get("sha256", "<missing>")),
            ),
        },
    )


def _tool(name: str, label: str, description: str, parameters: dict[str, Any], execute) -> ToolDefinition:
    metadata = get_builtin_tool_metadata(name)
    if metadata is None:
        raise ValueError(f"Missing builtin metadata for {name}")
    return ToolDefinition(
        name=name,
        label=label,
        description=description,
        parameters=parameters,
        metadata=metadata,
        execute=execute,
    )


def _resolve_read_path(sandbox: WorkspaceSandbox, path_text: str) -> tuple[Any | None, ToolResult | None]:
    try:
        return sandbox.resolve_path(path_text), None
    except ValueError:
        return None, _error_result(f"Path escapes workspace boundary: {path_text}", "path_escapes_workspace")


def _resolve_write_path(sandbox: WorkspaceSandbox, path_text: str) -> tuple[Any | None, ToolResult | None]:
    try:
        return sandbox.ensure_mutable_path(sandbox.resolve_path(path_text)), None
    except ValueError as exc:
        return None, _error_result(str(exc), "path_escapes_workspace")


def _error_result(
    message: str,
    error_code: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> ToolResult:
    return ToolResult(
        content=[TextContent(text=message)],
        status="error",
        is_error=True,
        error_code=error_code,
        metadata={**_metadata_for_error(error_code), **(metadata or {})},
    )


def _metadata_for_error(error_code: str) -> dict[str, Any]:
    hints = {
        "path_not_found": "List or search the workspace to confirm the path.",
        "path_not_file": "List or search the workspace to confirm the path.",
        "path_escapes_workspace": "Use a path inside the current workspace.",
        "stale_file": "Read the file again before editing.",
        "multiple_matches": "Read a larger target region and provide unique old_text.",
        "no_match": "Read the current file content before retrying.",
        "invalid_patch": "Pass edits as [{path, old_text, new_text}].",
    }
    if error_code not in hints:
        return {}
    return {"recovery_hint": {"message": hints[error_code]}}


def _output_quality(
    *,
    decode_status: str = "ok",
    truncated: bool = False,
    original_chars: int | None = None,
    returned_chars: int | None = None,
    may_be_binary: bool = False,
) -> dict[str, Any]:
    return {
        "encoding": "utf-8" if decode_status != "invalid_utf8" else "unknown",
        "decode_status": decode_status,
        "truncated": truncated,
        "original_chars": original_chars,
        "returned_chars": returned_chars,
        "may_be_binary": may_be_binary,
        "reliable_for_reasoning": decode_status == "ok" and not truncated and not may_be_binary,
    }


def _change_evidence(change_kind: str, path: str, before_hash: str, after_hash: str) -> dict[str, Any]:
    return {
        "change_kind": change_kind,
        "before_hashes": {path: before_hash},
        "after_hashes": {path: after_hash},
        "affected_paths": [path],
        "effect_detection": "direct",
        "effect_detection_confidence": "high",
        "safe_revert_available": False,
    }


def _state_hash(workspace: Any, path: str) -> str:
    return str(file_state_for_path(workspace, path).get("sha256", "<missing>"))


def _replace_nth(text: str, old: str, new: str, nth: int) -> str:
    start = 0
    for index in range(nth):
        found = text.find(old, start)
        if found < 0:
            raise ValueError("nth occurrence not found")
        if index == nth - 1:
            return text[:found] + new + text[found + len(old) :]
        start = found + len(old)
    return text


def _ls_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "max_entries": {"type": "integer"},
        },
        "required": [],
        "additionalProperties": False,
    }


def _read_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "max_chars": {"type": "integer"},
            "offset": {"type": "integer"},
            "limit": {"type": "integer"},
        },
        "required": ["path"],
        "additionalProperties": False,
    }


def _write_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "overwrite": {"type": "boolean"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }


def _edit_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
            "replace_all": {"type": "boolean"},
            "occurrence_index": {"type": "integer"},
            "expected_occurrences": {"type": "integer"},
            "expected_file_hash": {"type": "string"},
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    }


def _apply_patch_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "edits": {
                "type": "array",
                "minItems": 1,
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                    },
                    "required": ["path", "old_text", "new_text"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["edits"],
        "additionalProperties": False,
    }


__all__ = ["create_file_tools"]
