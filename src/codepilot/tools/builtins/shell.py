from __future__ import annotations

"""Built-in bash tool with workspace effects and verification summaries."""

import asyncio
import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from codepilot.protocols import TextContent
from codepilot.tools.contracts import ToolCallRequest, ToolDefinition, ToolResult
from codepilot.tools.registry import get_builtin_tool_metadata
from codepilot.tools.sandbox import (
    ShellExecutionPolicy,
    WorkspaceSandbox,
    build_shell_environment,
    classify_shell_command,
    truncate_output,
)


@dataclass(frozen=True)
class _WorkspaceEffects:
    available: bool
    status: dict[str, str]
    hashes: dict[str, str]


def create_shell_tools(
    sandbox: WorkspaceSandbox,
    *,
    allow: Callable[[str], bool],
    policy: ShellExecutionPolicy | None = None,
) -> list[ToolDefinition]:
    if not allow("bash"):
        return []
    execution_policy = policy or ShellExecutionPolicy()

    async def bash_tool(
        request: ToolCallRequest,
        signal=None,
        on_update=None,
    ) -> ToolResult:
        _ = signal
        params = request.arguments
        command = str(params.get("command", "")).strip()
        cwd_text = str(params.get("cwd", "."))
        timeout_seconds, timeout_error = execution_policy.validate_timeout(
            params.get("timeout_seconds")
        )
        if not command:
            return _shell_result("Missing command", command=command, status="error", error_code="missing_command")
        if timeout_error or timeout_seconds is None:
            return _shell_result(
                f"timeout_seconds must be between 1 and {execution_policy.max_timeout_seconds}",
                command=command,
                status="error",
                error_code="invalid_timeout",
            )
        alias_error = _workspace_alias_error(command, cwd_text, sandbox.root)
        if alias_error is not None:
            return _shell_result(
                alias_error,
                command=command,
                status="error",
                error_code="workspace_path_alias_not_supported",
                metadata={"workspace": str(sandbox.root.resolve()), "invalid_alias": "/workspace"},
            )
        try:
            cwd = sandbox.resolve_path(cwd_text)
        except ValueError:
            return _shell_result(
                f"Invalid cwd outside workspace: {cwd_text}",
                command=command,
                status="error",
                error_code="invalid_cwd",
            )
        if not cwd.exists() or not cwd.is_dir():
            return _shell_result(f"Invalid cwd: {cwd_text}", command=command, status="error", error_code="invalid_cwd")
        before = _workspace_effects(sandbox.root)
        if on_update:
            on_update(ToolResult(content=[TextContent(text=f"Running command: {command}")]))
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(cwd),
                env=build_shell_environment(execution_policy.allowed_env),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
            stdout_text, stdout_status = _decode_utf8(stdout)
            stderr_text, stderr_status = _decode_utf8(stderr)
            out = truncate_output(stdout_text, execution_policy.stdout_limit)
            err = truncate_output(stderr_text, execution_policy.stderr_limit)
            after = _workspace_effects(sandbox.root)
            affected, changed, diff_summary = _compare_effects(sandbox.root, before, after)
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
                    "output_quality": _output_quality(
                        stdout_status=stdout_status,
                        stderr_status=stderr_status,
                        stdout_truncated=out.truncated,
                        stderr_truncated=err.truncated,
                        stdout_original_chars=out.original_chars,
                        stderr_original_chars=err.original_chars,
                        stdout_returned_chars=out.returned_chars,
                        stderr_returned_chars=err.returned_chars,
                    ),
                    "change_evidence": _change_evidence(sandbox.root, before, after, affected),
                },
            )
        except asyncio.TimeoutError:
            await _terminate_process(proc)
            after = _workspace_effects(sandbox.root)
            affected, changed, diff_summary = _compare_effects(sandbox.root, before, after)
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
                    "change_evidence": _change_evidence(sandbox.root, before, after, affected),
                },
            )
        except asyncio.CancelledError:
            await _terminate_process(proc)
            after = _workspace_effects(sandbox.root)
            affected, changed, diff_summary = _compare_effects(sandbox.root, before, after)
            return _shell_result(
                "Command cancelled",
                command=command,
                status="cancelled",
                error_code="shell_cancelled",
                affected_paths=affected,
                workspace_changed=changed,
                diff_summary=diff_summary,
                metadata={
                    "cancelled": True,
                    "effect_detection": "git" if after.available else "unavailable",
                    "change_evidence": _change_evidence(sandbox.root, before, after, affected),
                },
            )
        except Exception as exc:
            await _terminate_process(proc)
            after = _workspace_effects(sandbox.root)
            affected, changed, diff_summary = _compare_effects(sandbox.root, before, after)
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
                    "change_evidence": _change_evidence(sandbox.root, before, after, affected),
                },
            )

    metadata = get_builtin_tool_metadata("bash")
    if metadata is None:
        raise ValueError("Missing builtin metadata for bash")
    return [
        ToolDefinition(
            name="bash",
            label="Run Command",
            description="在工作区内执行受限 shell 命令，危险命令会被拒绝。",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string"},
                    "timeout_seconds": {"type": "integer"},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            metadata=metadata,
            execute=bash_tool,
        )
    ]


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
) -> ToolResult:
    verification = None
    if classify_shell_command(command) == "verification":
        verification = {
            "status": "passed" if status == "success" else "cancelled" if status == "cancelled" else "failed",
            "command": command,
            "exit_code": exit_code,
            "summary": message[-500:],
        }
    effective_metadata = dict(metadata or {})
    hint = _recovery_hint(error_code)
    if hint is not None:
        effective_metadata.setdefault("recovery_hint", hint)
    return ToolResult(
        content=[TextContent(text=message)],
        status=status,  # type: ignore[arg-type]
        is_error=status != "success",
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
        metadata=effective_metadata,
    )


def _decode_utf8(raw: bytes) -> tuple[str, str]:
    try:
        return raw.decode("utf-8"), "ok"
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace"), "decoded_with_replacement"


def _output_quality(
    *,
    stdout_status: str = "ok",
    stderr_status: str = "ok",
    stdout_truncated: bool = False,
    stderr_truncated: bool = False,
    stdout_original_chars: int | None = None,
    stderr_original_chars: int | None = None,
    stdout_returned_chars: int | None = None,
    stderr_returned_chars: int | None = None,
) -> dict[str, Any]:
    decoded_with_replacement = stdout_status == "decoded_with_replacement" or stderr_status == "decoded_with_replacement"
    truncated = stdout_truncated or stderr_truncated
    return {
        "encoding": "utf-8",
        "decode_status": "decoded_with_replacement" if decoded_with_replacement else "ok",
        "truncated": truncated,
        "original_chars": (stdout_original_chars or 0) + (stderr_original_chars or 0),
        "returned_chars": (stdout_returned_chars or 0) + (stderr_returned_chars or 0),
        "may_be_binary": decoded_with_replacement,
        "reliable_for_reasoning": not decoded_with_replacement and not truncated,
    }


def _recovery_hint(error_code: str | None) -> dict[str, Any] | None:
    hints = {
        "shell_exit_nonzero": "Use stderr and verification summary to debug the failure before retrying.",
        "shell_timeout": "Narrow the command or ask before increasing scope.",
        "shell_execution_error": "Shell execution failed before a normal exit code was available.",
        "invalid_timeout": "Use a timeout within the configured allowed range.",
        "missing_command": "Provide a concrete command before invoking bash.",
        "workspace_path_alias_not_supported": "Remove hard-coded /workspace paths and rerun.",
    }
    if error_code not in hints:
        return None
    return {"message": hints[error_code]}


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
    hashes = {path: _path_fingerprint(root / path) for path in status}
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
        if before.status.get(path) != after.status.get(path)
        or before.hashes.get(path) != after.hashes.get(path)
    )
    if before.status != after.status:
        paths = sorted(set(paths) | set(after.status))
    stat = _git_diff_stat(root)
    summary = stat or (f"{len(paths)} workspace path(s) changed" if paths else "No Git status change")
    return paths, bool(paths), summary


def _change_evidence(
    root: Path,
    before: _WorkspaceEffects,
    after: _WorkspaceEffects,
    affected_paths: list[str],
) -> dict[str, Any]:
    before_hashes = {
        path: before.hashes.get(path) or _git_head_fingerprint(root, path) or "<missing>"
        for path in affected_paths
    }
    after_hashes = {path: after.hashes.get(path, "<missing>") for path in affected_paths}
    return {
        "change_kind": "unknown",
        "before_hashes": before_hashes,
        "after_hashes": after_hashes,
        "affected_paths": list(affected_paths),
        "effect_detection": "git" if after.available else "unavailable",
        "effect_detection_confidence": "medium" if after.available else "low",
        "safe_revert_available": False,
    }


def _workspace_alias_error(command: str, cwd_text: str, workspace: Path) -> str | None:
    workspace_posix = workspace.resolve().as_posix().rstrip("/")
    if workspace_posix == "/workspace" or workspace_posix.startswith("/workspace/"):
        return None
    command_text = command.replace("\\", "/")
    cwd = str(cwd_text).strip().replace("\\", "/").rstrip("/")
    if "/workspace" not in command_text and cwd != "/workspace" and not cwd.startswith("/workspace/"):
        return None
    return (
        "Hard-coded /workspace is not available for this session. "
        f"Commands already run in the workspace cwd: {workspace.resolve()}."
    )


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


def _git_head_fingerprint(root: Path, relative_path: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "show", f"HEAD:./{relative_path}"],
            cwd=root,
            capture_output=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


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
