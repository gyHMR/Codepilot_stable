from __future__ import annotations

"""内置文件工具：ls（列目录）、read（读文件）、write（写文件）、edit（精确替换）。"""

from typing import Any, Callable

from codepilot.protocols import TextContent
from codepilot.tools.sandbox import WorkspaceSandbox, file_state_for_path
from codepilot.tools.types import AgentTool, AgentToolResult


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
        "reliable_for_reasoning": (
            decode_status == "ok"
            and not truncated
            and not may_be_binary
        ),
    }


def _recovery_hint(error_code: str) -> dict[str, Any] | None:
    hints: dict[str, tuple[str, str, str, bool]] = {
        "invalid_utf8": (
            "ask_user",
            "The file is not valid UTF-8 text. Do not treat it as reliable source text.",
            "inspect_non_text_file",
            True,
        ),
        "stale_file": (
            "retry_read",
            "The file changed since it was read. Read it again before editing.",
            "read_context",
            False,
        ),
        "multiple_matches": (
            "refine_edit",
            "Read a larger target region and provide unique old_text or occurrence_index.",
            "edit_file",
            False,
        ),
        "no_match": (
            "retry_read",
            "Read the current file content before retrying the edit.",
            "read_context",
            False,
        ),
        "unexpected_match_count": (
            "refine_edit",
            "Re-read the target area and adjust the expected match count or old_text.",
            "edit_file",
            False,
        ),
        "path_not_found": (
            "retry_read",
            "List or search the workspace to confirm the current path.",
            "read_context",
            False,
        ),
        "path_not_file": (
            "retry_read",
            "List or search the workspace to confirm the current path.",
            "read_context",
            False,
        ),
    }
    spec = hints.get(error_code)
    if spec is None:
        return None
    category, message, suggested, requires_user = spec
    return {
        "category": category,
        "message": message,
        "suggested_action_intent": suggested,
        "requires_user_confirmation": requires_user,
    }


def _metadata_for_error(error_code: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    hint = _recovery_hint(error_code)
    if hint is not None:
        metadata["recovery_hint"] = hint
    if error_code == "invalid_utf8":
        metadata["output_quality"] = _output_quality(
            decode_status="invalid_utf8",
            may_be_binary=True,
        )
    return metadata


def _error_result(message: str, error_code: str, **details: Any) -> AgentToolResult:
    return AgentToolResult(
        content=[TextContent(text=message)],
        status="error",
        error_code=error_code,
        details=details,
        metadata=_metadata_for_error(error_code),
    )


def _change_evidence(
    *,
    change_kind: str,
    path: str,
    before_hash: str,
    after_hash: str,
) -> dict[str, Any]:
    return {
        "change_kind": change_kind,
        "before_hashes": {path: before_hash},
        "after_hashes": {path: after_hash},
        "affected_paths": [path],
        "effect_detection": "direct",
        "effect_detection_confidence": "high",
        "safe_revert_available": False,
    }


def _replace_nth(text: str, old: str, new: str, nth: int) -> str:
    if nth <= 0:
        raise ValueError("nth must be >= 1")
    start = 0
    match_count = 0
    while True:
        idx = text.find(old, start)
        if idx < 0:
            raise ValueError("nth occurrence not found")
        match_count += 1
        if match_count == nth:
            return text[:idx] + new + text[idx + len(old) :]
        start = idx + len(old)


def create_file_tools(
    sandbox: WorkspaceSandbox,
    *,
    allow: Callable[[str], bool],
    edit_require_unique_match: bool = True,
) -> list[AgentTool]:
    workspace = sandbox.root
    tools: list[AgentTool] = []
    max_write_chars = 1_000_000

    async def ls_tool(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
        _ = tool_call_id, signal, on_update
        path_text = str(params.get("path", "."))
        max_entries = int(params.get("max_entries", 100))
        target = sandbox.resolve_path(path_text)
        if not target.exists():
            return _error_result(f"Path not found: {path_text}", "path_not_found")
        if not target.is_dir():
            return _error_result(f"Not a directory: {path_text}", "not_a_directory")

        items = sorted(target.iterdir(), key=lambda p: p.name)[:max_entries]
        lines = []
        for item in items:
            suffix = "/" if item.is_dir() else ""
            size = "-" if item.is_dir() else str(item.stat().st_size)
            lines.append(f"{item.name}{suffix}\t{size}")
        return AgentToolResult(content=[TextContent(text="\n".join(lines) if lines else "(empty)")], details={})

    async def read_tool(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
        _ = tool_call_id, signal, on_update
        path_text = str(params.get("path", ""))
        max_chars = int(params.get("max_chars", 4000))
        offset = int(params.get("offset", 1))
        limit_raw = params.get("limit")
        limit = int(limit_raw) if limit_raw is not None else None
        if not path_text:
            return _error_result("Missing path", "missing_path")
        target = sandbox.resolve_path(path_text)
        if not target.exists():
            return _error_result(f"Path not found: {path_text}", "path_not_found")
        if not target.is_file():
            return _error_result(f"Not a file: {path_text}", "not_a_file")

        if offset < 1 or (limit is not None and limit < 1):
            return _error_result(
                "offset and limit must be positive integers",
                "invalid_line_range",
            )
        try:
            raw = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return _error_result(
                f"File is not valid UTF-8 text: {path_text}",
                "invalid_utf8",
            )
        lines = raw.splitlines()
        start_index = min(offset - 1, len(lines))
        end_index = len(lines) if limit is None else min(len(lines), start_index + limit)
        selected = lines[start_index:end_index]
        rendered = "\n".join(
            f"{line_no}\t{line}"
            for line_no, line in enumerate(selected, start=offset)
        )
        truncated = end_index < len(lines)
        char_truncated = len(rendered) > max_chars
        if char_truncated:
            rendered = rendered[:max_chars] + "\n...<truncated>..."
            truncated = True
        state = file_state_for_path(workspace, path_text)
        quality = _output_quality(
            truncated=truncated,
            original_chars=len(raw),
            returned_chars=len(rendered),
        )
        return AgentToolResult(
            content=[TextContent(text=rendered)],
            details={"file_state": state},
            metadata={
                "file_state": state,
                "start_line": offset if selected else None,
                "end_line": end_index if selected else None,
                "total_lines": len(lines),
                "truncated": truncated,
                "char_truncated": char_truncated,
                "output_quality": quality,
            },
        )

    async def write_tool(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
        _ = tool_call_id, signal, on_update
        path_text = str(params.get("path", ""))
        content = str(params.get("content", ""))
        overwrite = bool(params.get("overwrite", True))
        if not path_text:
            return _error_result("Missing path", "missing_path")
        if len(content) > max_write_chars:
            return _error_result(
                f"Content exceeds {max_write_chars} characters",
                "content_too_large",
                content_chars=len(content),
            )

        target = sandbox.resolve_path(path_text)
        if target.exists() and not target.is_file():
            return _error_result(
                f"Target is not a file: {path_text}",
                "target_not_file",
            )
        if target.exists() and not overwrite:
            return _error_result(f"File exists: {path_text}", "file_exists")
        try:
            original = target.read_text(encoding="utf-8") if target.exists() else None
        except UnicodeDecodeError:
            return _error_result(
                f"Existing file is not valid UTF-8: {path_text}",
                "invalid_utf8",
            )
        changed = original != content
        relative_path = target.relative_to(workspace).as_posix()
        before_hash = (
            file_state_for_path(workspace, relative_path).get("sha256", "<missing>")
            if target.exists()
            else "<missing>"
        )
        if not changed:
            state = file_state_for_path(workspace, relative_path)
            return AgentToolResult(
                content=[TextContent(text=f"File unchanged: {relative_path}")],
                affected_paths=[relative_path],
                workspace_changed=False,
                diff_summary="No content change",
                details={"changed": False, "file_state": state},
                metadata={
                    "file_state": state,
                    "change_evidence": _change_evidence(
                        change_kind="unchanged",
                        path=relative_path,
                        before_hash=str(before_hash),
                        after_hash=str(state.get("sha256", before_hash)),
                    ),
                },
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        action = "created" if original is None else "updated"
        state = file_state_for_path(workspace, relative_path)
        return AgentToolResult(
            content=[TextContent(text=f"Wrote file: {relative_path}")],
            affected_paths=[relative_path],
            workspace_changed=True,
            diff_summary=(
                f"{action} {relative_path}: "
                f"{len(original or '')} -> {len(content)} characters"
            ),
            details={"changed": True, "action": action, "file_state": state},
            metadata={
                "file_state": state,
                "change_evidence": _change_evidence(
                    change_kind="create" if original is None else "update",
                    path=relative_path,
                    before_hash=str(before_hash),
                    after_hash=str(state.get("sha256", "<missing>")),
                ),
            },
        )

    async def edit_tool(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
        _ = tool_call_id, signal, on_update
        path_text = str(params.get("path", ""))
        old_text = str(params.get("old_text", ""))
        new_text = str(params.get("new_text", ""))
        replace_all = bool(params.get("replace_all", False))
        occurrence_index_raw = params.get("occurrence_index")
        expected_occurrences_raw = params.get("expected_occurrences")
        expected_file_hash = params.get("expected_file_hash")
        if not path_text:
            return _error_result("Missing path", "missing_path")
        if old_text == "":
            return _error_result("old_text cannot be empty", "empty_old_text")
        occurrence_index = None if occurrence_index_raw is None else int(occurrence_index_raw)
        expected_occurrences = None if expected_occurrences_raw is None else int(expected_occurrences_raw)
        if occurrence_index is not None and occurrence_index <= 0:
            return _error_result(
                "occurrence_index must be >= 1",
                "invalid_occurrence_index",
            )
        if len(old_text) + len(new_text) > max_write_chars:
            return _error_result(
                "Edit payload is too large",
                "content_too_large",
            )
        if expected_occurrences is not None and expected_occurrences < 0:
            return _error_result(
                "expected_occurrences must be >= 0",
                "invalid_expected_occurrences",
            )
        target = sandbox.resolve_path(path_text)
        if not target.exists() or not target.is_file():
            return _error_result(
                f"Path not found or not file: {path_text}",
                "path_not_file",
            )

        try:
            original = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return _error_result(
                f"File is not valid UTF-8: {path_text}",
                "invalid_utf8",
            )
        current_state = file_state_for_path(workspace, path_text)
        before_hash = str(current_state.get("sha256", "<missing>"))
        if (
            expected_file_hash is not None
            and current_state.get("sha256") != str(expected_file_hash)
        ):
            return _error_result(
                "File changed since it was read; read it again before editing",
                "stale_file",
                path=current_state.get("path"),
                expected_file_hash=str(expected_file_hash),
                actual_file_hash=current_state.get("sha256"),
                file_state=current_state,
            )
        count = original.count(old_text)
        if expected_occurrences is not None and count != expected_occurrences:
            return _error_result(
                f"Expected {expected_occurrences} matches, but found {count}",
                "unexpected_match_count",
                matches=count,
                expected_occurrences=expected_occurrences,
            )
        if count == 0:
            return _error_result(
                "No match found",
                "no_match",
                replacements=0,
            )
        if not replace_all and count > 1 and occurrence_index is None and edit_require_unique_match:
            return _error_result(
                "Multiple matches found; set replace_all=true or provide more unique old_text",
                "multiple_matches",
                matches=count,
            )
        if replace_all:
            updated = original.replace(old_text, new_text)
            replaced = count
        else:
            if occurrence_index is not None:
                if occurrence_index > count:
                    return _error_result(
                        f"occurrence_index={occurrence_index} is out of range (matches={count})",
                        "occurrence_out_of_range",
                        matches=count,
                        occurrence_index=occurrence_index,
                    )
                updated = _replace_nth(original, old_text, new_text, occurrence_index)
            else:
                updated = original.replace(old_text, new_text, 1)
            replaced = 1
        target.write_text(updated, encoding="utf-8")
        relative_path = target.relative_to(workspace).as_posix()
        state = file_state_for_path(workspace, relative_path)
        after_hash = str(state.get("sha256", "<missing>"))
        return AgentToolResult(
            content=[TextContent(text=f"Edited file: {relative_path} (replacements={replaced})")],
            affected_paths=[relative_path],
            workspace_changed=updated != original,
            diff_summary=(
                f"edited {relative_path}: {replaced} replacement"
                f"{'s' if replaced != 1 else ''}"
            ),
            details={"replacements": replaced, "file_state": state},
            metadata={
                "file_state": state,
                "change_evidence": _change_evidence(
                    change_kind="update" if updated != original else "unchanged",
                    path=relative_path,
                    before_hash=before_hash,
                    after_hash=after_hash,
                ),
            },
        )

    if allow("ls"):
        tools.append(
            AgentTool(
                name="ls",
                label="List Directory",
                description="列出目录内容，返回文件名和大小。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对 workspace 的目录路径"},
                        "max_entries": {"type": "number", "description": "最多返回条目数，默认 100"},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
                execute=ls_tool,
            )
        )

    if allow("read"):
        tools.append(
            AgentTool(
                name="read",
                label="Read File",
                description="读取文本文件内容。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对 workspace 的文件路径"},
                        "max_chars": {"type": "number", "description": "最大返回字符数，默认 4000"},
                        "offset": {"type": "number", "description": "起始行，1-based，默认 1"},
                        "limit": {"type": "number", "description": "最多读取行数"},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                execute=read_tool,
            )
        )

    if allow("write"):
        tools.append(
            AgentTool(
                name="write",
                label="Write File",
                description="写入文本文件。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对 workspace 的文件路径"},
                        "content": {"type": "string", "description": "写入内容"},
                        "overwrite": {"type": "boolean", "description": "是否覆盖已存在文件，默认 true"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
                execute=write_tool,
            )
        )

    if allow("edit"):
        tools.append(
            AgentTool(
                name="edit",
                label="Edit File",
                description="按 old_text -> new_text 替换文件内容。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对 workspace 的文件路径"},
                        "old_text": {"type": "string", "description": "待替换原文"},
                        "new_text": {"type": "string", "description": "替换后的新文本"},
                        "replace_all": {"type": "boolean", "description": "是否替换全部匹配，默认 false"},
                        "occurrence_index": {"type": "number", "description": "替换第几次匹配（1-based）"},
                        "expected_occurrences": {"type": "number", "description": "期望 old_text 出现次数，不匹配则拒绝修改"},
                        "expected_file_hash": {
                            "type": "string",
                            "description": "可选；最近一次 read 返回的 sha256，不一致时拒绝编辑",
                        },
                    },
                    "required": ["path", "old_text", "new_text"],
                    "additionalProperties": False,
                },
                execute=edit_tool,
            )
        )

    return tools
