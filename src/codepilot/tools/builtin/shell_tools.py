from __future__ import annotations

import asyncio
from typing import Any, Callable

from codepilot.core import AgentTool, AgentToolResult
from codepilot.llm.types import TextContent
from codepilot.tools.sandbox import WorkspaceSandbox


def create_shell_tools(
    sandbox: WorkspaceSandbox,
    *,
    allow: Callable[[str], bool],
) -> list[AgentTool]:
    tools: list[AgentTool] = []

    async def bash_tool(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
        _ = tool_call_id, signal
        command = str(params.get("command", "")).strip()
        timeout_seconds = int(params.get("timeout_seconds", 30))
        cwd_text = str(params.get("cwd", "."))
        if not command:
            return AgentToolResult(content=[TextContent(text="Missing command")], details={})

        cwd = sandbox.resolve_path(cwd_text)
        if not cwd.exists() or not cwd.is_dir():
            return AgentToolResult(content=[TextContent(text=f"Invalid cwd: {cwd_text}")], details={})

        if on_update:
            on_update(AgentToolResult(content=[TextContent(text=f"Running command: {command}")], details={"phase": "start"}))
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
            out_text = stdout.decode("utf-8", errors="replace")
            err_text = stderr.decode("utf-8", errors="replace")
            merged = f"$ {command}\n{out_text}"
            if err_text:
                merged += ("\n[stderr]\n" + err_text)
            return AgentToolResult(
                content=[TextContent(text=merged.strip() or "(no output)")],
                details={"exit_code": proc.returncode},
            )
        except asyncio.TimeoutError:
            return AgentToolResult(
                content=[TextContent(text=f"Command timed out after {timeout_seconds}s")],
                details={"timeout": True},
            )

    if allow("bash"):
        tools.append(
            AgentTool(
                name="bash",
                label="Run Command",
                description="执行 shell 命令。",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "要执行的命令"},
                        "cwd": {"type": "string", "description": "命令执行目录，默认 ."},
                        "timeout_seconds": {"type": "number", "description": "超时时间（秒），默认 30"},
                        "allow_dangerous": {"type": "boolean", "description": "是否允许高风险命令（默认 false）"},
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
                execute=bash_tool,
            )
        )

    return tools
