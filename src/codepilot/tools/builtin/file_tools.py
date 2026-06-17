from __future__ import annotations

from typing import Any, Callable

from codepilot.core import AgentTool, AgentToolResult
from codepilot.llm.types import TextContent
from codepilot.tools.sandbox import WorkspaceSandbox


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

    async def ls_tool(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
        _ = tool_call_id, signal, on_update
        path_text = str(params.get("path", "."))
        max_entries = int(params.get("max_entries", 100))
        target = sandbox.resolve_path(path_text)
        if not target.exists():
            return AgentToolResult(content=[TextContent(text=f"Path not found: {path_text}")], details={})
        if not target.is_dir():
            return AgentToolResult(content=[TextContent(text=f"Not a directory: {path_text}")], details={})

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
        if not path_text:
            return AgentToolResult(content=[TextContent(text="Missing path")], details={})
        target = sandbox.resolve_path(path_text)
        if not target.exists():
            return AgentToolResult(content=[TextContent(text=f"Path not found: {path_text}")], details={})
        if not target.is_file():
            return AgentToolResult(content=[TextContent(text=f"Not a file: {path_text}")], details={})

        text = target.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...<truncated>..."
        return AgentToolResult(content=[TextContent(text=text)], details={})

    async def write_tool(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
        _ = tool_call_id, signal, on_update
        path_text = str(params.get("path", ""))
        content = str(params.get("content", ""))
        overwrite = bool(params.get("overwrite", True))
        if not path_text:
            return AgentToolResult(content=[TextContent(text="Missing path")], details={})

        target = sandbox.resolve_path(path_text)
        if target.exists() and not overwrite:
            return AgentToolResult(content=[TextContent(text=f"File exists: {path_text}")], details={})
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return AgentToolResult(content=[TextContent(text=f"Wrote file: {path_text}")], details={})

    async def edit_tool(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
        _ = tool_call_id, signal, on_update
        path_text = str(params.get("path", ""))
        old_text = str(params.get("old_text", ""))
        new_text = str(params.get("new_text", ""))
        replace_all = bool(params.get("replace_all", False))
        occurrence_index_raw = params.get("occurrence_index")
        expected_occurrences_raw = params.get("expected_occurrences")
        if not path_text:
            return AgentToolResult(content=[TextContent(text="Missing path")], details={})
        if old_text == "":
            return AgentToolResult(content=[TextContent(text="old_text cannot be empty")], details={})
        occurrence_index = None if occurrence_index_raw is None else int(occurrence_index_raw)
        expected_occurrences = None if expected_occurrences_raw is None else int(expected_occurrences_raw)
        if occurrence_index is not None and occurrence_index <= 0:
            return AgentToolResult(content=[TextContent(text="occurrence_index must be >= 1")], details={})
        if expected_occurrences is not None and expected_occurrences < 0:
            return AgentToolResult(content=[TextContent(text="expected_occurrences must be >= 0")], details={})
        target = sandbox.resolve_path(path_text)
        if not target.exists() or not target.is_file():
            return AgentToolResult(content=[TextContent(text=f"Path not found or not file: {path_text}")], details={})

        original = target.read_text(encoding="utf-8", errors="replace")
        count = original.count(old_text)
        if expected_occurrences is not None and count != expected_occurrences:
            return AgentToolResult(
                content=[TextContent(text=f"Expected {expected_occurrences} matches, but found {count}")],
                details={"matches": count, "expected_occurrences": expected_occurrences},
            )
        if count == 0:
            return AgentToolResult(content=[TextContent(text="No match found")], details={"replacements": 0})
        if not replace_all and count > 1 and occurrence_index is None and edit_require_unique_match:
            return AgentToolResult(
                content=[TextContent(text="Multiple matches found; set replace_all=true or provide more unique old_text")],
                details={"matches": count},
            )
        if replace_all:
            updated = original.replace(old_text, new_text)
            replaced = count
        else:
            if occurrence_index is not None:
                if occurrence_index > count:
                    return AgentToolResult(
                        content=[TextContent(text=f"occurrence_index={occurrence_index} is out of range (matches={count})")],
                        details={"matches": count, "occurrence_index": occurrence_index},
                    )
                updated = _replace_nth(original, old_text, new_text, occurrence_index)
            else:
                updated = original.replace(old_text, new_text, 1)
            replaced = 1
        target.write_text(updated, encoding="utf-8")
        return AgentToolResult(
            content=[TextContent(text=f"Edited file: {path_text} (replacements={replaced})")],
            details={"replacements": replaced},
        )

    if allow("ls") or allow("list_dir"):
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
        tools.append(
            AgentTool(
                name="list_dir",
                label="List Directory (compat)",
                description="兼容别名：等价于 ls。",
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

    if allow("read") or allow("read_file"):
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
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                execute=read_tool,
            )
        )
        tools.append(
            AgentTool(
                name="read_file",
                label="Read File (compat)",
                description="兼容别名：等价于 read。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对 workspace 的文件路径"},
                        "max_chars": {"type": "number", "description": "最大返回字符数，默认 4000"},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                execute=read_tool,
            )
        )

    if allow("write") or allow("write_file"):
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
        tools.append(
            AgentTool(
                name="write_file",
                label="Write File (compat)",
                description="兼容别名：等价于 write。",
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
                    },
                    "required": ["path", "old_text", "new_text"],
                    "additionalProperties": False,
                },
                execute=edit_tool,
            )
        )

    return tools
