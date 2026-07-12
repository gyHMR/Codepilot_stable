from __future__ import annotations

"""Canonical system command tool."""

import asyncio
from dataclasses import dataclass
from typing import Any

from ..codecs import DataclassCodec
from ..contracts import ToolExecutionContext, ToolHandlerError, ToolRegistration, ToolSpec
from ..results import TextContent
from ..sandbox import (
    ShellExecutionPolicy,
    WorkspaceSandbox,
    build_shell_environment,
    truncate_output,
    validate_shell_command,
)
from ..security import (
    ConcurrencyPolicy,
    OutputLimits,
    OutputTrustPolicy,
    TimeoutPolicy,
    ToolAccessRequest,
    ToolAccessResolution,
    ToolEffect,
    ToolPolicy,
    ToolResource,
)

_DRAFT = "https://json-schema.org/draft/2020-12/schema"


@dataclass(frozen=True)
class BashInput:
    command: str
    timeout_seconds: int | None = None


@dataclass(frozen=True)
class BashOutput:
    text: str
    stdout: str
    stderr: str
    exit_code: int
    details: dict[str, Any]
    metadata: dict[str, Any]


def create_shell_registration(
    sandbox: WorkspaceSandbox,
    *,
    policy: ShellExecutionPolicy | None = None,
) -> ToolRegistration:
    execution_policy = policy or ShellExecutionPolicy()
    input_schema = {
        "$schema": _DRAFT,
        "type": "object",
        "properties": {
            "command": {"type": "string", "minLength": 1},
            "timeout_seconds": {
                "type": ["integer", "null"],
                "minimum": 1,
                "maximum": execution_policy.max_timeout_seconds,
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }
    output_schema = {
        "$schema": _DRAFT,
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "stdout": {"type": "string"},
            "stderr": {"type": "string"},
            "exit_code": {"type": "integer"},
            "details": {"type": "object"},
            "metadata": {"type": "object"},
        },
        "required": ["text", "stdout", "stderr", "exit_code", "details", "metadata"],
        "additionalProperties": False,
    }

    class Resolver:
        def resolve(self, input: BashInput, request):
            _ = request
            shell_class = validate_shell_command(input.command)
            effects = {"process_spawn", "filesystem_read"}
            if shell_class in {"mutation", "unknown"}:
                effects.add("filesystem_write")
            return ToolAccessResolution(
                input=input,
                access=ToolAccessRequest(
                    actions=("shell.execute",),
                    resources=(ToolResource("workspace:///"),),
                    effects=frozenset(effects),
                    risk="low" if shell_class in {"verification", "read_only"} else "medium",
                    reason=f"Execute {shell_class} shell command",
                    safe_preview={
                        "command": input.command,
                        "shell_class": shell_class,
                    },
                ),
            )

    async def handler(input: BashInput, context: ToolExecutionContext) -> BashOutput:
        context.cancellation.raise_if_cancelled()
        shell_class = validate_shell_command(input.command)
        timeout, error = execution_policy.validate_timeout(input.timeout_seconds)
        if error or timeout is None:
            raise ToolHandlerError("shell.invalid_timeout", "Invalid shell timeout")
        proc = await asyncio.create_subprocess_shell(
            input.command,
            cwd=str(sandbox.root),
            env=build_shell_environment(execution_policy.allowed_env),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def cleanup() -> None:
            if proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()

        context.cleanup.push(cleanup)
        context.effects.report(
            ToolEffect(
                kind="process_spawn",
                resource=ToolResource("workspace:///"),
                operation=input.command,
                status="started",
                certainty="observed",
            )
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            await cleanup()
            raise ToolHandlerError("shell.timeout", f"Shell command timed out after {timeout}s") from exc
        context.effects.report(
            ToolEffect(
                kind="filesystem_read",
                resource=ToolResource("workspace:///"),
                operation="shell workspace access",
                status="completed",
                certainty="inferred",
            )
        )
        if shell_class in {"mutation", "unknown"}:
            context.effects.report(
                ToolEffect(
                    kind="filesystem_write",
                    resource=ToolResource("workspace:///"),
                    operation="shell workspace mutation",
                    status="unknown",
                    certainty="inferred",
                )
            )
        stdout_raw = stdout_bytes.decode("utf-8", errors="replace")
        stderr_raw = stderr_bytes.decode("utf-8", errors="replace")
        stdout = truncate_output(stdout_raw, execution_policy.stdout_limit)
        stderr = truncate_output(stderr_raw, execution_policy.stderr_limit)
        text = stdout.text
        if stderr.text:
            text = f"{text}\n[stderr]\n{stderr.text}" if text else stderr.text
        details = {
            "shell_class": shell_class,
            "timeout_seconds": timeout,
            "stdout_truncated": stdout.truncated,
            "stderr_truncated": stderr.truncated,
        }
        if proc.returncode:
            raise ToolHandlerError(
                "shell_exit_nonzero",
                text or f"Shell command exited with code {proc.returncode}",
                details={**details, "exit_code": proc.returncode},
            )
        return BashOutput(
            text=text,
            stdout=stdout.text,
            stderr=stderr.text,
            exit_code=proc.returncode or 0,
            details=details,
            metadata={
                "output_quality": {
                    "truncated": stdout.truncated or stderr.truncated,
                    "reliable_for_reasoning": not stdout.truncated and not stderr.truncated,
                }
            },
        )

    class Renderer:
        def render(self, data):
            return (TextContent(text=data["text"]),)

    return ToolRegistration(
        version="1.0.0",
        implementation_version="2",
        spec=ToolSpec(
            "bash",
            "Run one bounded system command in the workspace and return stdout, stderr, and exit code.",
            input_schema,
            output_schema,
        ),
        category="command",
        source="builtin",
        owner="codepilot.builtin",
        policy=ToolPolicy(
            allowed_modes=frozenset({"execute"}),
            declared_effects=frozenset({"process_spawn", "filesystem_read", "filesystem_write"}),
            required_permissions=frozenset({"shell.execute"}),
            base_risk="medium",
            approval="on_risk",
            timeout=TimeoutPolicy(
                execution_policy.timeout_seconds * 1_000,
                execution_policy.max_timeout_seconds * 1_000,
            ),
            concurrency=ConcurrencyPolicy(mode="serial", group="system_command"),
            output_limits=OutputLimits(
                max_content_bytes=execution_policy.stdout_limit + execution_policy.stderr_limit + 4_096
            ),
            output_trust=OutputTrustPolicy(),
        ),
        input_codec=DataclassCodec(BashInput, input_schema),
        output_codec=DataclassCodec(BashOutput, output_schema),
        handler=handler,
        renderer=Renderer(),
        access_resolver=Resolver(),
    )


__all__ = ["create_shell_registration"]
