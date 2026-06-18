from __future__ import annotations

import asyncio
from typing import Any, Callable

from codepilot.protocols import TextContent
from codepilot.tools.sandbox import WorkspaceSandbox
from codepilot.tools.types import AgentTool, AgentToolResult


def _is_verification_command(command: str) -> bool:
    text = command.lower()
    markers = (
        "pytest",
        "unittest",
        "npm test",
        "npm run test",
        "npm run lint",
        "npm run build",
        "ruff ",
        "mypy ",
        "pyright ",
        "compileall",
        "cargo test",
        "go test",
    )
    return any(marker in text for marker in markers)


def _shell_result(
    message: str,
    *,
    command: str,
    status: str,
    exit_code: int | None = None,
    error_code: str | None = None,
) -> AgentToolResult:
    verification = None
    if _is_verification_command(command):
        verification = {
            "status": (
                "passed"
                if status == "success"
                else "cancelled"
                if status == "cancelled"
                else "failed"
            ),
            "command": command,
            "exit_code": exit_code,
            "summary": message[:500],
        }
    return AgentToolResult(
        content=[TextContent(text=message)],
        status=status,  # type: ignore[arg-type]
        error_code=error_code,
        exit_code=exit_code,
        workspace_changed=None,
        verification=verification,
        details={"command": command, "exit_code": exit_code},
    )


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
            return _shell_result(
                "Missing command",
                command=command,
                status="error",
                error_code="missing_command",
            )

        cwd = sandbox.resolve_path(cwd_text)
        if not cwd.exists() or not cwd.is_dir():
            return _shell_result(
                f"Invalid cwd: {cwd_text}",
                command=command,
                status="error",
                error_code="invalid_cwd",
            )

        if on_update:
            on_update(AgentToolResult(content=[TextContent(text=f"Running command: {command}")], details={"phase": "start"}))
        proc: asyncio.subprocess.Process | None = None
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
            status = "success" if proc.returncode == 0 else "error"
            return _shell_result(
                merged.strip() or "(no output)",
                command=command,
                status=status,
                exit_code=proc.returncode,
                error_code=None if proc.returncode == 0 else "shell_exit_nonzero",
            )
        except asyncio.TimeoutError:
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.communicate()
            return _shell_result(
                f"Command timed out after {timeout_seconds}s",
                command=command,
                status="error",
                error_code="shell_timeout",
            )
        except asyncio.CancelledError:
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.communicate()
            return _shell_result(
                "Command cancelled",
                command=command,
                status="cancelled",
                error_code="shell_cancelled",
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
