from __future__ import annotations

import re
from typing import Any, Callable

from codepilot.core import AgentTool, AgentToolResult
from codepilot.llm.types import TextContent
from codepilot.tools.sandbox import WorkspaceSandbox


def create_search_tools(sandbox: WorkspaceSandbox, *, allow: Callable[[str], bool]) -> list[AgentTool]:
    workspace = sandbox.root
    tools: list[AgentTool] = []

    async def grep_tool(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
        _ = tool_call_id, signal, on_update
        pattern = str(params.get("pattern", ""))
        start_path = str(params.get("path", "."))
        glob_pattern = str(params.get("glob", "**/*"))
        max_matches = int(params.get("max_matches", 200))
        case_sensitive = bool(params.get("case_sensitive", True))
        if not pattern:
            return AgentToolResult(content=[TextContent(text="Missing pattern")], details={})

        root = sandbox.resolve_path(start_path)
        if not root.exists():
            return AgentToolResult(content=[TextContent(text=f"Path not found: {start_path}")], details={})

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            return AgentToolResult(content=[TextContent(text=f"Invalid regex: {exc}")], details={})

        matches: list[str] = []
        files = [p for p in root.glob(glob_pattern) if p.is_file()]
        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
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
            details={"matches": len(matches)},
        )

    async def find_tool(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
        _ = tool_call_id, signal, on_update
        start_path = str(params.get("path", "."))
        pattern = str(params.get("pattern", "**/*"))
        max_results = int(params.get("max_results", 200))
        root = sandbox.resolve_path(start_path)
        if not root.exists():
            return AgentToolResult(content=[TextContent(text=f"Path not found: {start_path}")], details={})

        results = []
        for path in root.glob(pattern):
            rel = path.relative_to(workspace).as_posix()
            results.append(rel + ("/" if path.is_dir() else ""))
            if len(results) >= max_results:
                break
        return AgentToolResult(
            content=[TextContent(text="\n".join(results) if results else "(no files)")],
            details={"count": len(results)},
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
