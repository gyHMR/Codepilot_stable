"""规范的系统命令工具 —— command（受控命令）和 bash（完整 Shell）。

本文件实现两个命令执行工具：
1. command — 受控命令（argv 数组，无 Shell 解析）
    - 参数通过显式的 argv 列表传递，避免 Shell 注入
    - 只支持已知的可识别命令（通过 validate_controlled_command 验证）
    - 自动路径参数沙箱验证
    - 适合 git、pytest、npm 等已知命令

2. bash — 完整 Shell 命令
    - 通过子进程的 Shell 解析执行（需要显式审批）
    - 拒绝高风险命令（rm -rf, git push --force 等）
    - 拒绝修改 .codepilot 内部状态的命令
    - 拒绝涉及敏感文件的命令
    - 输出截断（head-tail 策略）
    - 适合管道、重定向、命令链等 Shell 语法场景

两者共享：
- ShellExecutionPolicy（超时、输出截断限制）
- build_shell_environment（环境变量白名单过滤）
- _collect_process_result（异步进程管理 + 输出截断）
"""

import asyncio
import subprocess
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
    TimeoutPolicy,
    ToolAccessRequest,
    ToolAccessResolution,
    ToolEffect,
    ToolEffectKind,
    ToolPolicy,
    ToolResource,
)

_DRAFT = "https://json-schema.org/draft/2020-12/schema"


# ── 输入类型 ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CommandInput:
    """受控命令的输入参数。

    参数:
        argv: 命令参数列表（如 ["git", "status"]）
        cwd: 工作目录（相对于工作区，默认 "."）
        timeout_seconds: 超时时间（可选，受 ShellExecutionPolicy 限制）
    """
    argv: list[str]
    cwd: str = "."
    timeout_seconds: int | None = None


@dataclass(frozen=True)
class BashInput:
    """Shell 命令的输入参数。

    参数:
        command: 原始 Shell 命令文本
        timeout_seconds: 超时时间（可选，受 ShellExecutionPolicy 限制）
    """
    command: str
    timeout_seconds: int | None = None


@dataclass(frozen=True)
class BashOutput:
    """命令执行的统一输出类型。

    参数:
        text: 合并后的输出文本（stdout + stderr）
        stdout: 标准输出文本
        stderr: 标准错误文本
        exit_code: 进程退出码
        details: 结构化详情（命令分类、超时、截断信息）
        metadata: 元数据（输出质量信息）
    """
    text: str
    stdout: str
    stderr: str
    exit_code: int
    details: dict[str, Any]
    metadata: dict[str, Any]
    verification: dict[str, Any] | None = None


# ── 注册创建函数 ──────────────────────────────────────────────────────────────


def create_command_registration(
    sandbox: WorkspaceSandbox,
    *,
    policy: ShellExecutionPolicy | None = None,
) -> ToolRegistration:
    """创建受控命令工具（command）。

    受控命令通过 argv 数组传递参数，不经过 Shell 解析，
    因此不会受到 Shell 注入攻击。但只能执行已知的安全命令。

    安全策略:
    - 只允许 execute 模式
    - 串行执行（serial, group="system_command"）
    - 审批策略 on_risk
    - 破坏性命令（rm -rf 等）被拒绝
    - 未知命令被拒绝
    - 每个路径参数都经过沙箱验证

    参数:
        sandbox: 工作区沙箱
        policy: Shell 执行策略（默认使用 ShellExecutionPolicy 默认值）

    返回:
        command 工具的 ToolRegistration
    """
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
        """受控命令的访问解析器。"""
        def resolve(self, input: CommandInput, request):
            _ = request
            timeout, error = execution_policy.validate_timeout(input.timeout_seconds)
            if error or timeout is None:
                raise ValueError("Invalid command timeout")
            argv, cwd, assessment = validate_controlled_command(
                input.argv,
                sandbox=sandbox,
                cwd=input.cwd,
            )
            profile = assessment.profile
            effects = assessment.effects
            return ToolAccessResolution(
                input=input,
                access=ToolAccessRequest(
                    actions=(f"command.{profile}",),
                    resources=(ToolResource("workspace:///" + sandbox.relative_path(cwd)),),
                    effects=effects,
                    risk=assessment.risk,
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
                execution_timeout_ms=timeout * 1_000,
            )

    async def handler(input: CommandInput, context: ToolExecutionContext) -> BashOutput:
        """受控命令处理器 —— 使用 create_subprocess_exec 执行。

        不经过 Shell 解析，直接执行 argv 中的可执行文件。
        """
        context.cancellation.raise_if_cancelled()
        argv, cwd, assessment = validate_controlled_command(
            input.argv,
            sandbox=sandbox,
            cwd=input.cwd,
        )
        profile = assessment.profile
        timeout, error = execution_policy.validate_timeout(input.timeout_seconds)
        if error or timeout is None:
            raise ToolHandlerError("command.invalid_timeout", "Invalid command timeout")
        proc = await _spawn_process(
            argv,
            cwd=str(cwd),
            env=build_shell_environment(execution_policy.allowed_env),
        )
        return await _collect_process_result(
            proc,
            context=context,
            execution_policy=execution_policy,
            timeout=timeout,
            operation=" ".join(argv),
            profile=profile,
            verification=assessment.verification,
            effects=assessment.effects,
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
            "Run one recognized project or inspection command as an argv array without Shell parsing. Prefer this for tests, lint, builds, version-control inspection, and other commands that do not require pipes, redirects, expansion, or chaining. Each call has bounded output and timeout metadata; inspect the exit code before claiming success. Capability approval may be required for the first matching command. Use bash only when Shell syntax is essential.",
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
    """创建 Shell 命令工具（bash）。

    Shell 命令通过子进程 Shell 执行，支持管道、重定向、命令链等 Shell 语法。
    但受严格的安全控制：
    - 高风险命令（rm -rf, git push --force 等）被拒绝
    - 修改 .codepilot 内部状态的命令被拒绝
    - 涉及敏感文件的命令被拒绝
    - 环境变量被严格过滤（白名单 + 敏感关键字）
    - 输出被截断（head-tail 策略）
    - 审批范围固定为 once（每次需要重新审批）

    参数:
        sandbox: 工作区沙箱
        policy: Shell 执行策略（默认使用 ShellExecutionPolicy 默认值）

    返回:
        bash 工具的 ToolRegistration
    """
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
        """Shell 命令的访问解析器。"""
        def resolve(self, input: BashInput, request):
            _ = request
            timeout, error = execution_policy.validate_timeout(input.timeout_seconds)
            if error or timeout is None:
                raise ValueError("Invalid shell timeout")
            assessment = validate_shell_command(input.command)
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
                    risk=assessment.risk,
                    reason=f"Execute {assessment.profile} shell command",
                    safe_preview={
                        "command": input.command,
                        "shell_class": assessment.profile,
                    },
                    approval_scopes=frozenset({"once"}),
                ),
                execution_timeout_ms=timeout * 1_000,
            )

    async def handler(input: BashInput, context: ToolExecutionContext) -> BashOutput:
        """Shell 命令处理器 —— 使用 create_subprocess_shell 执行。

        通过系统 Shell 执行命令，支持完整的 Shell 语法。
        """
        context.cancellation.raise_if_cancelled()
        assessment = validate_shell_command(input.command)
        timeout, error = execution_policy.validate_timeout(input.timeout_seconds)
        if error or timeout is None:
            raise ToolHandlerError("shell.invalid_timeout", "Invalid shell timeout")
        proc = await _spawn_process(
            input.command,
            cwd=str(sandbox.root),
            env=build_shell_environment(execution_policy.allowed_env),
            shell=True,
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
            "shell_class": assessment.profile,
            "timeout_seconds": timeout,
            "stdout_truncated": stdout.truncated,
            "stderr_truncated": stderr.truncated,
        }
        if proc.returncode:
            raise ToolHandlerError(
                "shell_exit_nonzero",
                text or f"Shell command exited with code {proc.returncode}",
                details={
                    **details,
                    "exit_code": proc.returncode,
                    **_verification_detail(
                        assessment.verification,
                        input.command,
                        "failed",
                    ),
                },
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
            verification=_verification_value(
                assessment.verification,
                input.command,
                "passed",
            ),
        )

    class Renderer:
        def render(self, data):
            return (TextContent(text=data["text"]),)

    return ToolRegistration(
        version="1.0.0",
        implementation_version="2",
        spec=ToolSpec(
            "bash",
            "Run one approved raw Shell command with the workspace as cwd. Use only when pipes, redirection, command chaining, variable expansion, or another Shell feature cannot be expressed by command or a dedicated file/search tool. Keep the command scoped and inspect exit code plus truncation metadata; do not use Shell syntax to bypass tool permissions or overwrite unrelated user work.",
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


# ── 共享辅助函数 ──────────────────────────────────────────────────────────────


class _ThreadedProcess:
    """Expose the small asyncio-process surface used by the tool handlers."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self._process = process

    @property
    def returncode(self) -> int | None:
        return self._process.poll()

    def terminate(self) -> None:
        self._process.terminate()

    def kill(self) -> None:
        self._process.kill()

    async def wait(self) -> int:
        return await asyncio.to_thread(self._process.wait)

    async def communicate(self) -> tuple[bytes, bytes]:
        stdout, stderr = await asyncio.to_thread(self._process.communicate)
        return stdout or b"", stderr or b""


async def _spawn_process(
    command: list[str] | str,
    *,
    cwd: str,
    env: dict[str, str],
    shell: bool = False,
) -> _ThreadedProcess:
    """Start a process without depending on the event loop's subprocess transport."""

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = await asyncio.to_thread(
        subprocess.Popen,
        command,
        cwd=cwd,
        env=env,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creationflags,
    )
    return _ThreadedProcess(process)


async def _collect_process_result(
    proc,
    *,
    context: ToolExecutionContext,
    execution_policy: ShellExecutionPolicy,
    timeout: int,
    operation: str,
    profile: CommandProfile,
    verification: bool,
    effects: frozenset[ToolEffectKind],
    error_prefix: str,
    resource: ToolResource,
    details: dict[str, Any],
) -> BashOutput:
    """收集异步进程的执行结果（command 和 bash 共享）。

    处理流程:
    1. 注册清理回调（超时或取消时终止进程）
    2. 报告 process_spawn 副作用
    3. 等待进程完成（带超时）
    4. 报告各类副作用（filesystem_read/write, network_access 等）
    5. 解码并截断输出
    6. 检查进程退出码

    参数:
        proc: 异步子进程
        context: 执行上下文
        execution_policy: 执行策略
        timeout: 超时秒数
        operation: 操作描述
        profile: 命令执行画像
        effects: 声明副作用集合
        error_prefix: 错误码前缀
        resource: 进程资源标识
        details: 额外详情

    返回:
        BashOutput 执行结果

    抛出:
        ToolHandlerError: 超时或进程退出码非零
    """
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

    # 报告副作用（根据声明的 effects 集合）
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

    # 解码输出并截断
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
            details={
                **result_details,
                "exit_code": proc.returncode,
                **_verification_detail(verification, operation, "failed"),
            },
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
        verification=_verification_value(verification, operation, "passed"),
    )


def _output_schema() -> dict[str, Any]:
    """命令输出的统一 JSON Schema。"""
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
            "verification": {
                "anyOf": [
                    {"type": "object"},
                    {"type": "null"},
                ]
            },
        },
        "required": [
            "text",
            "stdout",
            "stderr",
            "exit_code",
            "details",
            "metadata",
            "verification",
        ],
        "additionalProperties": False,
    }


def _verification_value(
    verification: bool,
    operation: str,
    status: str,
) -> dict[str, str] | None:
    if not verification:
        return None
    return {"status": status, "command": operation}


def _verification_detail(
    verification: bool,
    operation: str,
    status: str,
) -> dict[str, object]:
    value = _verification_value(verification, operation, status)
    return {"verification": value} if value is not None else {}


__all__ = ["create_command_registration", "create_shell_registration"]
