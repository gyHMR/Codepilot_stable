from __future__ import annotations

"""Canonical system command tool."""

import asyncio
from dataclasses import dataclass
from typing import Any

from ..codecs import DataclassCodec
from ..contracts import ToolExecutionContext, ToolHandlerError, ToolRegistration, ToolSpec
from ..results import TextContent
from ..sandbox import (
    CommandProfile,
    ShellExecutionPolicy,
    WorkspaceSandbox,
    build_shell_environment,
    truncate_output,
    validate_controlled_command,
    validate_shell_command,
)
from ..security import (
    ConcurrencyPolicy,
    OutputLimits,
    OutputTrustPolicy,
    RiskLevel,
    TimeoutPolicy,
    ToolAccessRequest,
    ToolAccessResolution,
    ToolEffect,
    ToolEffectKind,
    ToolPolicy,
    ToolResource,
)

_DRAFT = "https://json-schema.org/draft/2020-12/schema"


@dataclass(frozen=True)
class CommandInput:
    argv: list[str]
    cwd: str = "."
    timeout_seconds: int | None = None


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


def create_command_registration(
    sandbox: WorkspaceSandbox,
    *,
    policy: ShellExecutionPolicy | None = None,
) -> ToolRegistration:
    execution_policy = policy or ShellExecutionPolicy()
    input_schema = {
        "$schema": _DRAFT,
        "type": "object",
        "properties": {
            "argv": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "maxItems": 128,
                "description": "Executable followed by literal arguments; do not include Shell operators.",
            },
            "cwd": {
                "type": "string",
                "minLength": 1,
                "default": ".",
                "description": "Workspace-relative working directory.",
            },
            "timeout_seconds": {
                "type": ["integer", "null"],
                "minimum": 1,
                "maximum": execution_policy.max_timeout_seconds,
            },
        },
        "required": ["argv"],
        "additionalProperties": False,
    }
    output_schema = _output_schema()

    class Resolver:
        def resolve(self, input: CommandInput, request):
            _ = request
            argv, cwd, profile = validate_controlled_command(
                input.argv,
                sandbox=sandbox,
                cwd=input.cwd,
            )
            effects = _command_effects(profile)
            return ToolAccessResolution(
                input=input,
                access=ToolAccessRequest(
                    actions=(f"command.{profile}",),
                    resources=(ToolResource("workspace:///" + sandbox.relative_path(cwd)),),
                    effects=effects,
                    risk=_command_risk(profile),
                    reason=f"Run controlled {profile} command",
                    safe_preview={
                        "argv": argv,
                        "cwd": sandbox.relative_path(cwd) or ".",
                        "command_profile": profile,
                    },
                    approval_scopes=(
                        frozenset({"once"})
                        if profile == "external_effect"
                        else frozenset({"once", "session", "project"})
                    ),
                ),
            )

    async def handler(input: CommandInput, context: ToolExecutionContext) -> BashOutput:
        context.cancellation.raise_if_cancelled()
        argv, cwd, profile = validate_controlled_command(
            input.argv,
            sandbox=sandbox,
            cwd=input.cwd,
        )
        timeout, error = execution_policy.validate_timeout(input.timeout_seconds)
        if error or timeout is None:
            raise ToolHandlerError("command.invalid_timeout", "Invalid command timeout")
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            env=build_shell_environment(execution_policy.allowed_env),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return await _collect_process_result(
            proc,
            context=context,
            execution_policy=execution_policy,
            timeout=timeout,
            operation=" ".join(argv),
            profile=profile,
            effects=_command_effects(profile),
            error_prefix="command",
            resource=ToolResource("workspace:///" + sandbox.relative_path(cwd)),
            details={"cwd": sandbox.relative_path(cwd) or "."},
        )

    class Renderer:
        def render(self, data):
            return (TextContent(text=data["text"]),)

    return ToolRegistration(
        version="1.0.0",
        implementation_version="1",
        spec=ToolSpec(
            "command",
            "Run a recognized project command as an argv array without Shell parsing. Prefer this for inspection, tests, lint, builds, and bounded formatting; use bash only when Shell syntax is required.",
            input_schema,
            output_schema,
        ),
        category="command",
        source="builtin",
        owner="codepilot.builtin",
        policy=ToolPolicy(
            allowed_modes=frozenset({"execute"}),
            declared_effects=frozenset(
                {
                    "process_spawn",
                    "filesystem_read",
                    "filesystem_write",
                    "network_access",
                    "external_state_write",
                }
            ),
            required_permissions=frozenset({"command.execute"}),
            base_risk="low",
            approval="on_risk",
            timeout=TimeoutPolicy(
                execution_policy.timeout_seconds * 1_000,
                execution_policy.max_timeout_seconds * 1_000,
            ),
            concurrency=ConcurrencyPolicy(mode="serial", group="system_command"),
            output_limits=OutputLimits(
                max_content_bytes=execution_policy.stdout_limit
                + execution_policy.stderr_limit
                + 4_096
            ),
            output_trust=OutputTrustPolicy(),
        ),
        input_codec=DataclassCodec(CommandInput, input_schema),
        output_codec=DataclassCodec(BashOutput, output_schema),
        handler=handler,
        renderer=Renderer(),
        access_resolver=Resolver(),
    )


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
            "command": {
                "type": "string",
                "minLength": 1,
                "description": "Raw Shell text; approval is required and destructive commands are rejected.",
            },
            "timeout_seconds": {
                "type": ["integer", "null"],
                "minimum": 1,
                "maximum": execution_policy.max_timeout_seconds,
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }
    output_schema = _output_schema()

    class Resolver:
        def resolve(self, input: BashInput, request):
            _ = request
            shell_class = validate_shell_command(input.command)
            return ToolAccessResolution(
                input=input,
                access=ToolAccessRequest(
                    actions=("shell.execute",),
                    resources=(ToolResource("workspace:///"),),
                    effects=frozenset(
                        {
                            "process_spawn",
                            "filesystem_read",
                            "filesystem_write",
                            "network_access",
                            "external_state_write",
                        }
                    ),
                    risk="high",
                    reason=f"Execute {shell_class} shell command",
                    safe_preview={
                        "command": input.command,
                        "shell_class": shell_class,
                    },
                    approval_scopes=frozenset({"once"}),
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
        for kind, operation in (
            ("filesystem_write", "shell workspace mutation"),
            ("network_access", "shell network access"),
            ("external_state_write", "shell external state mutation"),
        ):
            context.effects.report(
                ToolEffect(
                    kind=kind,
                    resource=ToolResource("workspace:///"),
                    operation=operation,
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
            "Run one approved raw Shell command with workspace cwd. Use only when pipes, redirection, command chaining, or other Shell syntax cannot be expressed with the command tool.",
            input_schema,
            output_schema,
        ),
        category="command",
        source="builtin",
        owner="codepilot.builtin",
        policy=ToolPolicy(
            allowed_modes=frozenset({"execute"}),
            declared_effects=frozenset(
                {
                    "process_spawn",
                    "filesystem_read",
                    "filesystem_write",
                    "network_access",
                    "external_state_write",
                }
            ),
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


def _command_effects(profile: CommandProfile) -> frozenset[ToolEffectKind]:
    effects = {"process_spawn", "filesystem_read"}
    if profile in {"repository_execution", "bounded_mutation", "external_effect"}:
        effects.add("filesystem_write")
    if profile == "external_effect":
        effects.update({"network_access", "external_state_write"})
    return frozenset(effects)


def _command_risk(profile: CommandProfile) -> RiskLevel:
    if profile == "inspection":
        return "low"
    if profile in {"repository_execution", "bounded_mutation"}:
        return "medium"
    return "high"


async def _collect_process_result(
    proc,
    *,
    context: ToolExecutionContext,
    execution_policy: ShellExecutionPolicy,
    timeout: int,
    operation: str,
    profile: CommandProfile,
    effects: frozenset[ToolEffectKind],
    error_prefix: str,
    resource: ToolResource,
    details: dict[str, Any],
) -> BashOutput:
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
            resource=resource,
            operation=operation,
            status="started",
            certainty="observed",
        )
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        await cleanup()
        raise ToolHandlerError(
            f"{error_prefix}.timeout",
            f"Command timed out after {timeout}s",
        ) from exc

    effect_operations = {
        "filesystem_read": "command workspace access",
        "filesystem_write": "command workspace mutation",
        "network_access": "command network access",
        "external_state_write": "command external state mutation",
    }
    for kind in sorted(effects - {"process_spawn"}):
        context.effects.report(
            ToolEffect(
                kind=kind,
                resource=resource,
                operation=effect_operations[kind],
                status="completed" if kind == "filesystem_read" else "unknown",
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
    result_details = {
        **details,
        "command_profile": profile,
        "timeout_seconds": timeout,
        "stdout_truncated": stdout.truncated,
        "stderr_truncated": stderr.truncated,
    }
    if proc.returncode:
        raise ToolHandlerError(
            f"{error_prefix}_exit_nonzero",
            text or f"Command exited with code {proc.returncode}",
            details={**result_details, "exit_code": proc.returncode},
        )
    return BashOutput(
        text=text,
        stdout=stdout.text,
        stderr=stderr.text,
        exit_code=proc.returncode or 0,
        details=result_details,
        metadata={
            "output_quality": {
                "truncated": stdout.truncated or stderr.truncated,
                "reliable_for_reasoning": not stdout.truncated and not stderr.truncated,
            }
        },
    )


def _output_schema() -> dict[str, Any]:
    return {
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


__all__ = ["create_command_registration", "create_shell_registration"]
