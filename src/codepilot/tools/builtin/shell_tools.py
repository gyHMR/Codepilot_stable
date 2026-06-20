from __future__ import annotations

"""内置 shell 工具：bash（受限命令执行，含超时、环境过滤和工作区变更检测）。"""

import asyncio
import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from codepilot.protocols import TextContent
from codepilot.tools.sandbox import WorkspaceSandbox
from codepilot.tools.shell_policy import (
    ShellExecutionPolicy,
    build_shell_environment,
    classify_shell_command,
    truncate_output,
)
from codepilot.tools.types import AgentTool, AgentToolResult


@dataclass(frozen=True)
class _WorkspaceEffects:
    available: bool
    status: dict[str, str]
    hashes: dict[str, str]


def _is_verification_command(command: str) -> bool:
    return classify_shell_command(command) == "verification"


def _shell_result(
    message: str,
    *,
    command: str,
    status: str,
    exit_code: int | None = None,
    error_code: str | None = None,
    affected_paths: list[str] | None = None,
    workspace_changed: bool | None = None,
    diff_summary: str | None = None,
    metadata: dict[str, Any] | None = None,
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
            "summary": message[-500:],
        }
    return AgentToolResult(
        content=[TextContent(text=message)],
        status=status,  # type: ignore[arg-type]
        error_code=error_code,
        exit_code=exit_code,
        affected_paths=affected_paths or [],
        workspace_changed=workspace_changed,
        diff_summary=diff_summary,
        verification=verification,
        details={
            "command": command,
            "exit_code": exit_code,
            "shell_class": classify_shell_command(command),
        },
        metadata=metadata or {},
    )


def create_shell_tools(
    sandbox: WorkspaceSandbox,
    *,
    allow: Callable[[str], bool],
    policy: ShellExecutionPolicy | None = None,
) -> list[AgentTool]:
    tools: list[AgentTool] = []
    execution_policy = policy or ShellExecutionPolicy()

    async def bash_tool(
        tool_call_id: str,
        params: dict[str, Any],
        signal=None,
        on_update=None,
    ) -> AgentToolResult:
        _ = tool_call_id, signal
        command = str(params.get("command", "")).strip()
        cwd_text = str(params.get("cwd", "."))
        timeout_seconds, timeout_error = execution_policy.validate_timeout(
            params.get("timeout_seconds")
        )
        if not command:
            return _shell_result(
                "Missing command",
                command=command,
                status="error",
                error_code="missing_command",
            )
        if timeout_error or timeout_seconds is None:
            return _shell_result(
                (
                    "timeout_seconds must be between 1 and "
                    f"{execution_policy.max_timeout_seconds}"
                ),
                command=command,
                status="error",
                error_code="invalid_timeout",
            )

        cwd = sandbox.resolve_path(cwd_text)
        if not cwd.exists() or not cwd.is_dir():
            return _shell_result(
                f"Invalid cwd: {cwd_text}",
                command=command,
                status="error",
                error_code="invalid_cwd",
            )

        before = _workspace_effects(sandbox.root)
        if on_update:
            on_update(
                AgentToolResult(
                    content=[TextContent(text=f"Running command: {command}")],
                    details={"phase": "start"},
                )
            )
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(cwd),
                env=build_shell_environment(execution_policy.allowed_env),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_seconds,
            )
            out = truncate_output(
                stdout.decode("utf-8", errors="replace"),
                execution_policy.stdout_limit,
            )
            err = truncate_output(
                stderr.decode("utf-8", errors="replace"),
                execution_policy.stderr_limit,
            )
            after = _workspace_effects(sandbox.root)
            affected, changed, diff_summary = _compare_effects(
                sandbox.root,
                before,
                after,
            )
            merged = f"$ {command}\n{out.text}"
            if err.text:
                merged += "\n[stderr]\n" + err.text
            status = "success" if proc.returncode == 0 else "error"
            return _shell_result(
                merged.strip() or "(no output)",
                command=command,
                status=status,
                exit_code=proc.returncode,
                error_code=None if proc.returncode == 0 else "shell_exit_nonzero",
                affected_paths=affected,
                workspace_changed=changed,
                diff_summary=diff_summary,
                metadata={
                    "timed_out": False,
                    "stdout_truncated": out.truncated,
                    "stderr_truncated": err.truncated,
                    "stdout_original_chars": out.original_chars,
                    "stderr_original_chars": err.original_chars,
                    "stdout_returned_chars": out.returned_chars,
                    "stderr_returned_chars": err.returned_chars,
                    "effect_detection": "git" if after.available else "unavailable",
                    "timeout_seconds": timeout_seconds,
                },
            )
        except asyncio.TimeoutError:
            await _terminate_process(proc)
            after = _workspace_effects(sandbox.root)
            affected, changed, diff_summary = _compare_effects(
                sandbox.root,
                before,
                after,
            )
            return _shell_result(
                f"Command timed out after {timeout_seconds}s",
                command=command,
                status="error",
                error_code="shell_timeout",
                affected_paths=affected,
                workspace_changed=changed,
                diff_summary=diff_summary,
                metadata={
                    "timed_out": True,
                    "effect_detection": "git" if after.available else "unavailable",
                    "timeout_seconds": timeout_seconds,
                },
            )
        except asyncio.CancelledError:
            await _terminate_process(proc)
            after = _workspace_effects(sandbox.root)
            affected, changed, diff_summary = _compare_effects(
                sandbox.root,
                before,
                after,
            )
            return _shell_result(
                "Command cancelled",
                command=command,
                status="cancelled",
                error_code="shell_cancelled",
                affected_paths=affected,
                workspace_changed=changed,
                diff_summary=diff_summary,
                metadata={
                    "timed_out": False,
                    "cancelled": True,
                    "effect_detection": "git" if after.available else "unavailable",
                },
            )
        except Exception as exc:
            await _terminate_process(proc)
            after = _workspace_effects(sandbox.root)
            affected, changed, diff_summary = _compare_effects(
                sandbox.root,
                before,
                after,
            )
            return _shell_result(
                f"Command execution failed: {exc}",
                command=command,
                status="error",
                error_code="shell_execution_error",
                affected_paths=affected,
                workspace_changed=changed,
                diff_summary=diff_summary,
                metadata={
                    "exception_type": type(exc).__name__,
                    "effect_detection": "git" if after.available else "unavailable",
                },
            )

    if allow("bash"):
        tools.append(
            AgentTool(
                name="bash",
                label="Run Command",
                description=(
                    "在工作区内执行受限 shell 命令。危险命令会被拒绝，"
                    "未知或修改型命令可能需要用户审批。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "要执行的命令"},
                        "cwd": {
                            "type": "string",
                            "description": "工作区内的命令执行目录，默认 .",
                        },
                        "timeout_seconds": {
                            "type": "number",
                            "description": (
                                "超时时间（秒），范围 1-"
                                f"{execution_policy.max_timeout_seconds}"
                            ),
                        },
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
                execute=bash_tool,
            )
        )
    return tools


def _workspace_effects(root: Path) -> _WorkspaceEffects:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", "."],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return _WorkspaceEffects(False, {}, {})
    if result.returncode != 0:
        return _WorkspaceEffects(False, {}, {})
    status: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if len(line) >= 4:
            status[line[3:].split(" -> ")[-1]] = line[:2]
    hashes = {
        path: _path_fingerprint(root / path)
        for path in status
    }
    return _WorkspaceEffects(True, status, hashes)


def _compare_effects(
    root: Path,
    before: _WorkspaceEffects,
    after: _WorkspaceEffects,
) -> tuple[list[str], bool | None, str | None]:
    if not before.available or not after.available:
        return [], None, "Workspace effect detection unavailable (not a Git repository)"
    paths = sorted(
        path
        for path in set(before.status) | set(after.status)
        if (
            before.status.get(path) != after.status.get(path)
            or before.hashes.get(path) != after.hashes.get(path)
        )
    )
    # A dirty file can remain " M" before and after. Include current dirty paths
    # conservatively so downstream freshness checks revalidate them.
    if before.status != after.status:
        paths = sorted(set(paths) | set(after.status))
    changed = bool(paths)
    stat = _git_diff_stat(root)
    summary = stat or (
        f"{len(paths)} workspace path(s) changed" if paths else "No Git status change"
    )
    return paths, changed, summary


def _path_fingerprint(path: Path) -> str:
    if not path.exists():
        return "<missing>"
    if not path.is_file():
        return "<non-file>"
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return "<unreadable>"
    return digest.hexdigest()


def _git_diff_stat(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "diff", "--stat", "--", "."],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = result.stdout.strip()
    return text[-1000:] if text else None


async def _terminate_process(proc: asyncio.subprocess.Process | None) -> None:
    if proc is None or proc.returncode is not None:
        return
    try:
        if os.name == "nt":
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(proc.pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(killer.communicate(), timeout=5)
        else:
            proc.kill()
        await asyncio.wait_for(proc.communicate(), timeout=5)
    except Exception:
        try:
            proc.kill()
        except ProcessLookupError:
            pass


__all__ = ["create_shell_tools"]
