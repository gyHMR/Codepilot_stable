from __future__ import annotations

"""内置搜索工具：grep（正则内容搜索）、find（glob 文件查找）。"""

import re
from typing import Any, Callable

from codepilot.protocols import TextContent
from codepilot.tools.sandbox import WorkspaceSandbox
from codepilot.tools.types import AgentTool, AgentToolResult

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


def _error_result(message: str, error_code: str) -> AgentToolResult:
    return AgentToolResult(
        content=[TextContent(text=message)],
        status="error",
        error_code=error_code,
        details={},
    )


def _coerce_int(
    value: Any,
    *,
    name: str,
    default: int,
    minimum: int = 1,
) -> tuple[int | None, AgentToolResult | None]:
    if value is None:
        return default, None
    if isinstance(value, bool):
        return None, _error_result(f"{name} must be an integer", "invalid_argument")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None, _error_result(f"{name} must be an integer", "invalid_argument")
    if parsed < minimum:
        return None, _error_result(f"{name} must be >= {minimum}", "invalid_argument")
    return parsed, None


def _resolve_path(
    sandbox: WorkspaceSandbox,
    path_text: str,
) -> tuple[Any | None, AgentToolResult | None]:
    try:
        return sandbox.resolve_path(path_text), None
    except ValueError:
        return None, _error_result(
            f"Path escapes workspace boundary: {path_text}",
            "path_escapes_workspace",
        )


def create_search_tools(sandbox: WorkspaceSandbox, *, allow: Callable[[str], bool]) -> list[AgentTool]:
    workspace = sandbox.root
    tools: list[AgentTool] = []

    async def grep_tool(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
        _ = tool_call_id, signal, on_update
        pattern = str(params.get("pattern", ""))
        start_path = str(params.get("path", "."))
        glob_pattern = str(params.get("glob", "**/*"))
        max_matches, arg_error = _coerce_int(
            params.get("max_matches"),
            name="max_matches",
            default=200,
        )
        if arg_error is not None:
            return arg_error
        case_sensitive = bool(params.get("case_sensitive", True))
        if not pattern:
            return _error_result("Missing pattern", "missing_pattern")

        root, path_error = _resolve_path(sandbox, start_path)
        if path_error is not None:
            return path_error
        if not root.exists():
            return _error_result(f"Path not found: {start_path}", "path_not_found")

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            return _error_result(f"Invalid regex: {exc}", "invalid_regex")

        matches: list[str] = []
        files = sorted(
            (
                p
                for p in root.glob(glob_pattern)
                if p.is_file() and not _is_ignored(p, root)
            ),
            key=lambda path: path.as_posix(),
        )
        scanned = 0
        skipped_binary = 0
        scan_truncated = len(files) > _MAX_SCAN_FILES
        for file_path in files[:_MAX_SCAN_FILES]:
            scanned += 1
            try:
                if file_path.stat().st_size > _MAX_FILE_BYTES or _is_binary(file_path):
                    skipped_binary += 1
                    continue
                text = file_path.read_text(encoding="utf-8")
            except Exception:
                continue
            for idx, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    rel = file_path.relative_to(workspace).as_posix()
                    matches.append(f"{rel}:{idx}:{line[:220]}")
                    if len(matches) >= max_matches:
                        break
            if len(matches) >= max_matches:
                break

        return AgentToolResult(
            content=[TextContent(text="\n".join(matches) if matches else "(no matches)")],
            details={
                "matches": len(matches),
                "scanned_files": scanned,
                "skipped_binary_files": skipped_binary,
            },
            metadata={
                "truncated": len(matches) >= max_matches or scan_truncated,
                "match_limit": max_matches,
                "scan_file_limit": _MAX_SCAN_FILES,
            },
        )

    async def find_tool(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
        _ = tool_call_id, signal, on_update
        start_path = str(params.get("path", "."))
        pattern = str(params.get("pattern", "**/*"))
        max_results, arg_error = _coerce_int(
            params.get("max_results"),
            name="max_results",
            default=200,
        )
        if arg_error is not None:
            return arg_error
        root, path_error = _resolve_path(sandbox, start_path)
        if path_error is not None:
            return path_error
        if not root.exists():
            return _error_result(f"Path not found: {start_path}", "path_not_found")

        candidates = sorted(
            (
                path
                for path in root.glob(pattern)
                if not _is_ignored(path, root)
            ),
            key=lambda path: path.as_posix(),
        )
        results = []
        for path in candidates[:_MAX_SCAN_FILES]:
            rel = path.relative_to(workspace).as_posix()
            results.append(rel + ("/" if path.is_dir() else ""))
            if len(results) >= max_results:
                break
        return AgentToolResult(
            content=[TextContent(text="\n".join(results) if results else "(no files)")],
            details={"count": len(results)},
            metadata={
                "truncated": (
                    len(results) >= max_results
                    or len(candidates) > _MAX_SCAN_FILES
                ),
                "result_limit": max_results,
                "scan_file_limit": _MAX_SCAN_FILES,
            },
        )

    if allow("grep"):
        tools.append(
            AgentTool(
                name="grep",
                label="Search Content",
                description="在文件内容里按正则搜索。",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "正则表达式"},
                        "path": {"type": "string", "description": "起始目录，默认 ."},
                        "glob": {"type": "string", "description": "glob 过滤，默认 **/*"},
                        "max_matches": {"type": "number", "description": "最大匹配条数，默认 200"},
                        "case_sensitive": {"type": "boolean", "description": "是否大小写敏感，默认 true"},
                    },
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
                execute=grep_tool,
            )
        )

    if allow("find"):
        tools.append(
            AgentTool(
                name="find",
                label="Find Files",
                description="按 glob 查找文件/目录路径。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "起始目录，默认 ."},
                        "pattern": {"type": "string", "description": "glob 表达式，默认 **/*"},
                        "max_results": {"type": "number", "description": "最大返回条数，默认 200"},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
                execute=find_tool,
            )
        )

    return tools


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
