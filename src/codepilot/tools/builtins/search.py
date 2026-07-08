from __future__ import annotations

"""Built-in search tools: grep and find."""

import re
from typing import Any, Callable

from codepilot.protocols import TextContent
from codepilot.tools.contracts import ToolCallRequest, ToolDefinition, ToolResult
from codepilot.tools.registry import get_builtin_tool_metadata
from codepilot.tools.sandbox import WorkspaceSandbox

_IGNORED_DIRS = {
    ".git",
    ".codepilot",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
}
_MAX_SCAN_FILES = 5000
_MAX_FILE_BYTES = 2 * 1024 * 1024


def create_search_tools(
    sandbox: WorkspaceSandbox,
    *,
    allow: Callable[[str], bool],
) -> list[ToolDefinition]:
    workspace = sandbox.root
    tools: list[ToolDefinition] = []

    async def grep_tool(request: ToolCallRequest, signal=None, on_update=None) -> ToolResult:
        _ = signal, on_update
        params = request.arguments
        pattern = str(params.get("pattern", ""))
        start_path = str(params.get("path", "."))
        glob_pattern = str(params.get("glob", "**/*"))
        max_matches = int(params.get("max_matches", 200))
        case_sensitive = bool(params.get("case_sensitive", True))
        if not pattern:
            return _error_result("Missing pattern", "missing_pattern")
        root, error = _resolve_path(sandbox, start_path)
        if error is not None:
            return error
        if not root.exists():
            return _error_result(f"Path not found: {start_path}", "path_not_found")
        try:
            regex = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
        except re.error as exc:
            return _error_result(f"Invalid regex: {exc}", "invalid_regex")
        files = sorted(
            (path for path in root.glob(glob_pattern) if path.is_file() and not _is_ignored(path, root)),
            key=lambda path: path.as_posix(),
        )
        matches: list[str] = []
        scanned = 0
        skipped_binary = 0
        for file_path in files[:_MAX_SCAN_FILES]:
            scanned += 1
            try:
                if file_path.stat().st_size > _MAX_FILE_BYTES or _is_binary(file_path):
                    skipped_binary += 1
                    continue
                text = file_path.read_text(encoding="utf-8")
            except Exception:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    rel = file_path.relative_to(workspace).as_posix()
                    matches.append(f"{rel}:{line_no}:{line[:220]}")
                    if len(matches) >= max_matches:
                        break
            if len(matches) >= max_matches:
                break
        truncated = len(matches) >= max_matches or len(files) > _MAX_SCAN_FILES
        return ToolResult(
            content=[TextContent(text="\n".join(matches) if matches else "(no matches)")],
            details={
                "matches": len(matches),
                "scanned_files": scanned,
                "skipped_binary_files": skipped_binary,
            },
            metadata={
                "search_paths": [root.relative_to(workspace).as_posix()],
                "matches": len(matches),
                "truncated": truncated,
                "output_quality": {"truncated": truncated},
            },
        )

    async def find_tool(request: ToolCallRequest, signal=None, on_update=None) -> ToolResult:
        _ = signal, on_update
        params = request.arguments
        start_path = str(params.get("path", "."))
        pattern = str(params.get("pattern", "**/*"))
        max_results = int(params.get("max_results", 200))
        root, error = _resolve_path(sandbox, start_path)
        if error is not None:
            return error
        if not root.exists():
            return _error_result(f"Path not found: {start_path}", "path_not_found")
        candidates = sorted(
            (path for path in root.glob(pattern) if not _is_ignored(path, root)),
            key=lambda path: path.as_posix(),
        )
        results = []
        for path in candidates[:_MAX_SCAN_FILES]:
            rel = path.relative_to(workspace).as_posix()
            results.append(rel + ("/" if path.is_dir() else ""))
            if len(results) >= max_results:
                break
        truncated = len(results) >= max_results or len(candidates) > _MAX_SCAN_FILES
        return ToolResult(
            content=[TextContent(text="\n".join(results) if results else "(no files)")],
            details={"count": len(results)},
            metadata={
                "search_paths": [root.relative_to(workspace).as_posix()],
                "truncated": truncated,
                "output_quality": {"truncated": truncated},
            },
        )

    if allow("grep"):
        tools.append(_tool("grep", "Search Content", "在文件内容里按正则搜索。", _grep_schema(), grep_tool))
    if allow("find"):
        tools.append(_tool("find", "Find Files", "按 glob 查找文件/目录路径。", _find_schema(), find_tool))
    return tools


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


def _resolve_path(sandbox: WorkspaceSandbox, path_text: str) -> tuple[Any | None, ToolResult | None]:
    try:
        return sandbox.resolve_path(path_text), None
    except ValueError:
        return None, _error_result(f"Path escapes workspace boundary: {path_text}", "path_escapes_workspace")


def _error_result(message: str, error_code: str) -> ToolResult:
    return ToolResult(
        content=[TextContent(text=message)],
        status="error",
        is_error=True,
        error_code=error_code,
        metadata={"recovery_hint": {"message": "Adjust the search path or pattern and retry."}},
    )


def _is_ignored(path, root) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    return any(part in _IGNORED_DIRS for part in relative.parts)


def _is_binary(path) -> bool:
    try:
        with path.open("rb") as handle:
            chunk = handle.read(4096)
    except OSError:
        return True
    return b"\x00" in chunk


def _grep_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "glob": {"type": "string"},
            "max_matches": {"type": "integer"},
            "case_sensitive": {"type": "boolean"},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }


def _find_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "pattern": {"type": "string"},
            "max_results": {"type": "integer"},
        },
        "required": [],
        "additionalProperties": False,
    }


__all__ = ["create_search_tools"]
