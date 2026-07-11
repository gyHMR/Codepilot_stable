from __future__ import annotations

"""
内置 Bash 工具：工作区命令执行与变更检测

本模块是 Codepilot 工具链中的核心 Shell 执行组件，提供了安全受限的 Bash 命令执行能力。
它不仅执行命令，还会自动检测命令对工作区文件系统的影响，并返回结构化的执行结果。

核心能力：
  1. 安全沙箱执行 —— 通过 WorkspaceSandbox 限定命令可访问的路径和资源
  2. 超时与取消控制 —— 支持可配置的超时与异步取消（CancelledError）
  3. 工作区变更检测 —— 利用 Git 状态对比，自动识别命令修改/新增/删除了哪些文件
  4. 输出质量评估 —— 判断 stdout/stderr 的编码可用性和截断情况
  5. 跨平台进程终止 —— Windows 使用 taskkill，Unix 使用 SIGKILL
  6. 工作区路径别名检测 —— 拦截硬编码的 /workspace 路径，避免非标准路径的混淆

整体执行流程：
  参数校验 → 路径解析 → 执行前快照(before) → 启动 asyncio 子进程 → 执行后快照(after)
  → 对比变更(diff) → 组装结构化 ToolResult 返回
  任何环节出现异常（超时、取消、进程崩溃）均会尝试终止子进程并返回带有变更信息的错误结果。

依赖关系：
  - codepilot.protocols.TextContent: 标准文本内容协议
  - codepilot.tools.contracts: ToolDefinition / ToolCallRequest / ToolResult 等核心契约
  - codepilot.tools.sandbox: 沙箱策略、环境变量构建、命令分类、输出截断
  - codepilot.tools.registry: 获取内置工具的元数据（显示名、分类等）
"""

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
    """
    工作区快照（不可变数据类）

    存储某一时刻工作区所有 Git 追踪文件的状态快照，用于执行前后对比。

    属性:
        available (bool):
            True 表示 Git 可用，成功获取了状态快照。
            False 表示当前目录不是 Git 仓库或 git 命令执行失败，此时 status 和 hashes 为空。
        status (dict[str, str]):
            Git 状态映射: 文件相对路径 -> Git 状态码（如 " M", "??", "D " 等 git status --porcelain 格式）。
            键为相对于工作区根目录的文件路径。
        hashes (dict[str, str]):
            文件内容哈希映射: 文件相对路径 -> SHA-256 哈希值（通过 _path_fingerprint 计算）。
            用于检测文件内容级别的精确变化（不仅依赖 Git 状态码）。
    """
    available: bool
    status: dict[str, str]
    hashes: dict[str, str]


def create_shell_tools(
    sandbox: WorkspaceSandbox,
    *,
    allow: Callable[[str], bool],
    policy: ShellExecutionPolicy | None = None,
) -> list[ToolDefinition]:
    """
    创建 Bash 工具的工厂函数。

    这是本模块的唯一公开入口。根据给定的沙箱、权限策略和执行策略，
    返回一个包含 ToolDefinition 的列表。如果用户未授权 "bash" 工具，
    则返回空列表（工具不可用）。

    Args:
        sandbox (WorkspaceSandbox):
            工作区沙箱，定义了命令可以访问的根路径、路径解析规则等。
            所有命令执行前都会通过沙箱校验 cwd 和命令内容。
        allow (Callable[[str], bool]):
            权限检查函数。传入工具名 "bash"，返回布尔值表示是否允许使用。
            例如: allow("bash") -> True/False。
        policy (ShellExecutionPolicy | None):
            Shell 执行策略，控制超时范围、允许的环境变量、输出长度限制等。
            如果为 None，则使用默认的 ShellExecutionPolicy()。

    Returns:
        list[ToolDefinition]:
            包含一个 ToolDefinition 的列表（如果 bash 被允许），否则为空列表。
            每个 ToolDefinition 包含:
            - name: 工具名称 "bash"
            - label: 显示名 "Run Command"
            - description: 中文描述
            - parameters: JSON Schema 定义的参数（command, cwd, timeout_seconds）
            - metadata: 从 registry 获取的内置工具元数据
            - execute: 异步执行函数 bash_tool
    """
    # 权限检查: 如果用户未授权 bash 工具，直接返回空列表
    if not allow("bash"):
        return []
    # 使用传入的执行策略或默认策略
    execution_policy = policy or ShellExecutionPolicy()

    async def bash_tool(
        request: ToolCallRequest,
        signal=None,
        on_update=None,
    ) -> ToolResult:
        """
        Bash 工具的异步执行函数。

        完整的执行流程分为以下阶段：

        【阶段 1: 参数校验】
          - 从 request.arguments 中提取 command 和 cwd
          - 校验 command 非空
          - 通过 execution_policy.validate_timeout() 校验 timeout_seconds 合法性
          - 检测 cwd 或 command 中是否硬编码了 /workspace 路径别名

        【阶段 2: 路径解析】
          - 通过 sandbox.resolve_path() 将用户提供的 cwd 转换为绝对路径
          - 验证路径存在且为目录

        【阶段 3: 执行前快照】
          - 调用 _workspace_effects() 捕获当前工作区的 Git 状态和文件哈希

        【阶段 4: 异步子进程执行】
          - 通过 asyncio.create_subprocess_shell() 启动子进程
          - 使用 build_shell_environment() 构建受限环境变量
          - 通过 asyncio.wait_for() 施加超时控制
          - 捕获 stdout 和 stderr 并分别截断

        【阶段 5: 执行后快照与对比】
          - 再次调用 _workspace_effects() 获取执行后的 Git 状态
          - 通过 _compare_effects() 对比前后快照，识别变更的文件

        【阶段 6: 组装返回结果】
          - 通过 _shell_result() 构建结构化的 ToolResult
          - 包含输出内容、退出码、影响路径、diff 摘要、输出质量评估等

        【异常处理】
          - asyncio.TimeoutError: 命令超时，终止进程，返回 timeout 结果
          - asyncio.CancelledError: 用户取消，终止进程，返回 cancelled 结果
          - Exception: 未知错误，终止进程，返回 execution_error 结果

        所有异常路径均会捕获 after 快照，以便向调用者报告命令在超时/取消前已产生的部分影响。

        Args:
            request (ToolCallRequest):
                工具调用请求，包含 arguments 字典（command, cwd, timeout_seconds）。
            signal:
                保留参数，当前未使用（用下划线忽略）。
            on_update (Callable | None):
                可选回调，用于向 UI 层推送执行状态更新（如 "Running command: ..."）。

        Returns:
            ToolResult:
                结构化的执行结果，包含内容、状态、退出码、影响路径、
                diff 摘要、验证信息、元数据等完整信息。
        """
        _ = signal  # 保留参数，当前未使用
        params = request.arguments  # 提取参数: command, cwd, timeout_seconds
        # ---------- 阶段 1: 参数校验 ----------
        command = str(params.get("command", "")).strip()
        cwd_text = str(params.get("cwd", "."))  # 默认当前目录
        # 校验超时参数，返回 (有效超时秒数, 错误消息)
        timeout_seconds, timeout_error = execution_policy.validate_timeout(
            params.get("timeout_seconds")
        )
        # 命令不能为空
        if not command:
            return _shell_result("Missing command", command=command, status="error", error_code="missing_command")
        # 超时参数无效
        if timeout_error or timeout_seconds is None:
            return _shell_result(
                f"timeout_seconds must be between 1 and {execution_policy.max_timeout_seconds}",
                command=command,
                status="error",
                error_code="invalid_timeout",
            )
        # 检测硬编码的 /workspace 路径别名
        alias_error = _workspace_alias_error(command, cwd_text, sandbox.root)
        if alias_error is not None:
            return _shell_result(
                alias_error,
                command=command,
                status="error",
                error_code="workspace_path_alias_not_supported",
                metadata={"workspace": str(sandbox.root.resolve()), "invalid_alias": "/workspace"},
            )

        # ---------- 阶段 2: 路径解析 ----------
        try:
            cwd = sandbox.resolve_path(cwd_text)
        except ValueError:
            return _shell_result(
                f"Invalid cwd outside workspace: {cwd_text}",
                command=command,
                status="error",
                error_code="invalid_cwd",
            )
        # 验证路径存在且为目录
        if not cwd.exists() or not cwd.is_dir():
            return _shell_result(f"Invalid cwd: {cwd_text}", command=command, status="error", error_code="invalid_cwd")

        # ---------- 阶段 3: 执行前快照 ----------
        # 捕获当前工作区 Git 状态，作为变更对比的基准
        before = _workspace_effects(sandbox.root)

        # 通知 UI 层命令即将执行
        if on_update:
            on_update(ToolResult(content=[TextContent(text=f"Running command: {command}")]))

        proc: asyncio.subprocess.Process | None = None
        try:
            # ---------- 阶段 4: 异步子进程执行 ----------
            # 创建异步子进程（shell 模式: 命令通过系统 shell 解释执行）
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(cwd),  # 设置工作目录
                env=build_shell_environment(execution_policy.allowed_env),  # 构建受限环境变量
                stdout=asyncio.subprocess.PIPE,  # 捕获标准输出
                stderr=asyncio.subprocess.PIPE,  # 捕获标准错误
            )
            # 等待进程完成，施加超时控制
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
            # 解码输出（UTF-8，如果失败则使用 replacement 字符）
            stdout_text, stdout_status = _decode_utf8(stdout)
            stderr_text, stderr_status = _decode_utf8(stderr)
            # 根据执行策略截断过长的输出
            out = truncate_output(stdout_text, execution_policy.stdout_limit)
            err = truncate_output(stderr_text, execution_policy.stderr_limit)

            # ---------- 阶段 5: 执行后快照与对比 ----------
            after = _workspace_effects(sandbox.root)
            # 对比前后快照: 返回受影响文件列表、是否有变更、diff 摘要
            affected, changed, diff_summary = _compare_effects(sandbox.root, before, after)

            # ---------- 阶段 6: 组装返回结果 ----------
            # 拼接命令和标准输出（模拟终端展示）
            merged = f"$ {command}\n{out.text}"
            if err.text:
                merged += "\n[stderr]\n" + err.text  # 附加标准错误输出
            # 根据进程退出码决定状态
            status = "success" if proc.returncode == 0 else "error"
            return _shell_result(
                merged.strip() or "(no output)",  # 如果无输出则提供占位文本
                command=command,
                status=status,
                exit_code=proc.returncode,
                error_code=None if proc.returncode == 0 else "shell_exit_nonzero",
                affected_paths=affected,  # 受影响的文件路径列表
                workspace_changed=changed,  # 工作区是否有变更
                diff_summary=diff_summary,  # Git diff 摘要
                metadata={
                    "timed_out": False,
                    "stdout_truncated": out.truncated,  # stdout 是否被截断
                    "stderr_truncated": err.truncated,  # stderr 是否被截断
                    "stdout_original_chars": out.original_chars,  # stdout 原始字符数
                    "stderr_original_chars": err.original_chars,  # stderr 原始字符数
                    "stdout_returned_chars": out.returned_chars,  # stdout 实际返回字符数
                    "stderr_returned_chars": err.returned_chars,  # stderr 实际返回字符数
                    "effect_detection": "git" if after.available else "unavailable",  # 变更检测方式
                    "timeout_seconds": timeout_seconds,
                    "output_quality": _output_quality(  # 输出质量评估
                        stdout_status=stdout_status,
                        stderr_status=stderr_status,
                        stdout_truncated=out.truncated,
                        stderr_truncated=err.truncated,
                        stdout_original_chars=out.original_chars,
                        stderr_original_chars=err.original_chars,
                        stdout_returned_chars=out.returned_chars,
                        stderr_returned_chars=err.returned_chars,
                    ),
                    "change_evidence": _change_evidence(sandbox.root, before, after, affected),  # 变更证据详情
                },
            )

        # ---------- 异常处理: 超时 ----------
        except asyncio.TimeoutError:
            # 尝试终止未完成的子进程（跨平台）
            await _terminate_process(proc)
            # 即使超时，仍然捕获 after 快照 —— 命令可能已产生部分文件变更
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

        # ---------- 异常处理: 用户取消 ----------
        except asyncio.CancelledError:
            # 用户通过上层调度取消了任务
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

        # ---------- 异常处理: 其他未知错误 ----------
        except Exception as exc:
            # 捕获所有未预期的异常（如进程被操作系统杀死、管道断裂等）
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
                    "exception_type": type(exc).__name__,  # 记录异常类型便于诊断
                    "effect_detection": "git" if after.available else "unavailable",
                    "change_evidence": _change_evidence(sandbox.root, before, after, affected),
                },
            )

    # 从内置工具注册表中获取 bash 工具的元数据（显示名、分类、描述等）
    metadata = get_builtin_tool_metadata("bash")
    if metadata is None:
        raise ValueError("Missing builtin metadata for bash")

    # 构建并返回 ToolDefinition 列表
    return [
        ToolDefinition(
            name="bash",
            label="Run Command",
            description=(
                "在工作区内执行受限 shell 命令，危险命令会被拒绝。"
                "代码定位和文件阅读优先使用 read/grep/find；shell 主要用于运行测试、构建、"
                "项目命令或内置工具无法覆盖的检查。Windows 环境不要默认使用 Unix grep/head/file。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},       # 要执行的 shell 命令
                    "cwd": {"type": "string"},            # 工作目录（相对于沙箱根目录）
                    "timeout_seconds": {"type": "integer"},  # 超时时间（秒）
                },
                "required": ["command"],  # command 是必填参数
                "additionalProperties": False,  # 不允许额外参数
            },
            metadata=metadata,
            execute=bash_tool,  # 绑定异步执行函数
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
    """
    构建结构化的 Shell 执行结果。

    将命令执行的各项信息（输出、状态、变更、错误码等）组装为统一的 ToolResult，
    同时补充验证信息（verification）和恢复建议（recovery_hint）。

    Args:
        message (str):
            用户可见的结果文本，包含命令回显和输出内容。
        command (str):
            执行的原始命令字符串。
        status (str):
            执行状态: "success"（成功）、"error"（错误）、"cancelled"（取消）。
        exit_code (int | None):
            进程退出码。成功时为 0，错误时为非零，超时/取消时为 None。
        error_code (str | None):
            结构化错误码，便于上层程序化处理：
            - "missing_command": 未提供命令
            - "invalid_timeout": 超时参数不合法
            - "workspace_path_alias_not_supported": 硬编码了 /workspace
            - "invalid_cwd": 无效的工作目录
            - "shell_exit_nonzero": 命令执行失败（退出码非零）
            - "shell_timeout": 命令超时
            - "shell_cancelled": 用户取消
            - "shell_execution_error": 其他执行错误
        affected_paths (list[str] | None):
            受影响的文件路径列表（相对于工作区根目录）。
        workspace_changed (bool | None):
            工作区是否有文件变更。None 表示无法检测（非 Git 仓库）。
        diff_summary (str | None):
            Git diff 摘要文本（如 "3 files changed, 42 insertions(+), 8 deletions(-)"）。
        metadata (dict[str, Any] | None):
            额外的元数据字典（超时信息、截断信息、输出质量、变更证据等）。

    Returns:
        ToolResult:
            完整的结构化结果对象，包含:
            - content: TextContent 列表（消息文本）
            - status: 执行状态
            - is_error: 是否错误
            - error_code: 错误码
            - exit_code: 退出码
            - affected_paths: 影响路径
            - workspace_changed: 工作区变更标记
            - diff_summary: diff 摘要
            - verification: 验证信息（仅 verification 类命令）
            - details: 命令执行详情
            - metadata: 元数据（含恢复建议 recovery_hint）
    """
    # 验证信息: 如果命令被分类为 "verification"（验证类命令），
    # 则根据执行状态生成验证结果，帮助 AI 判断验证是否通过
    verification = None
    if classify_shell_command(command) == "verification":
        verification = {
            "status": "passed" if status == "success" else "cancelled" if status == "cancelled" else "failed",
            "command": command,
            "exit_code": exit_code,
            "summary": message[-500:],  # 截取最后 500 字符作为验证摘要
        }

    # 合并元数据，附加恢复建议
    effective_metadata = dict(metadata or {})
    hint = _recovery_hint(error_code)
    if hint is not None:
        effective_metadata.setdefault("recovery_hint", hint)  # 只在未设置时添加，避免覆盖调用者提供的提示

    return ToolResult(
        content=[TextContent(text=message)],
        status=status,  # type: ignore[arg-type]
        is_error=status != "success",  # 任何非 success 状态视为错误
        error_code=error_code,
        exit_code=exit_code,
        affected_paths=affected_paths or [],
        workspace_changed=workspace_changed,
        diff_summary=diff_summary,
        verification=verification,
        details={
            "command": command,
            "exit_code": exit_code,
            "shell_class": classify_shell_command(command),  # 命令分类: mutation/verification/unknown
        },
        metadata=effective_metadata,
    )


def _decode_utf8(raw: bytes) -> tuple[str, str]:
    """
    将原始字节解码为 UTF-8 字符串。

    如果标准解码失败（例如输出包含二进制数据），则使用 replacement 字符
    （U+FFFD）替换无法解码的字节，以保证始终返回可用的字符串。

    Args:
        raw (bytes): 子进程 stdout 或 stderr 的原始字节。

    Returns:
        tuple[str, str]:
            解码后的字符串和解码状态。
            - 成功: (decoded_text, "ok")
            - 失败后降级: (decoded_text_with_replacements, "decoded_with_replacement")
    """
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
    """
    评估命令输出的质量，帮助 AI 判断输出内容是否可靠。

    评估维度:
    - 编码完整性: 是否发生了解码降级（使用了 replacement 字符）
    - 截断情况: stdout/stderr 是否因超出限制而被截断
    - 字符数量: 原始字符数 vs 实际返回的字符数
    - 二进制风险: 是否可能包含二进制数据
    - 推理可用性: 综合判断输出是否可靠用于后续推理

    Args:
        stdout_status (str):
            stdout 解码状态（"ok" 或 "decoded_with_replacement"）。
        stderr_status (str):
            stderr 解码状态（"ok" 或 "decoded_with_replacement"）。
        stdout_truncated (bool):
            stdout 是否被截断。
        stderr_truncated (bool):
            stderr 是否被截断。
        stdout_original_chars (int | None):
            stdout 原始字符总数。
        stderr_original_chars (int | None):
            stderr 原始字符总数。
        stdout_returned_chars (int | None):
            stdout 实际返回的字符数（截断后）。
        stderr_returned_chars (int | None):
            stderr 实际返回的字符数（截断后）。

    Returns:
        dict[str, Any]:
            包含以下键的质量评估字典:
            - encoding: 编码格式（固定 "utf-8"）
            - decode_status: 整体解码状态
            - truncated: 是否发生了截断
            - original_chars: 原始字符总数（stdout + stderr）
            - returned_chars: 返回字符总数（stdout + stderr）
            - may_be_binary: 是否可能包含二进制数据
            - reliable_for_reasoning: 输出是否可靠用于后续推理
    """
    # 判断是否有任何流发生了降级解码或截断
    decoded_with_replacement = stdout_status == "decoded_with_replacement" or stderr_status == "decoded_with_replacement"
    truncated = stdout_truncated or stderr_truncated
    return {
        "encoding": "utf-8",
        "decode_status": "decoded_with_replacement" if decoded_with_replacement else "ok",
        "truncated": truncated,
        "original_chars": (stdout_original_chars or 0) + (stderr_original_chars or 0),
        "returned_chars": (stdout_returned_chars or 0) + (stderr_returned_chars or 0),
        "may_be_binary": decoded_with_replacement,  # 降级解码通常意味着存在非文本字节
        # 只有未降级解码且未截断的输出才被认为是"可靠"的
        "reliable_for_reasoning": not decoded_with_replacement and not truncated,
    }


def _recovery_hint(error_code: str | None) -> dict[str, Any] | None:
    """
    为已知错误码提供恢复建议。

    当命令执行失败时，根据错误码返回对应的恢复提示消息，
    帮助 AI 或用户理解失败原因并采取纠正措施。

    Args:
        error_code (str | None):
            错误码字符串（与 _shell_result 中定义的 error_code 对应）。

    Returns:
        dict[str, Any] | None:
            包含 "message" 键的字典（恢复建议），如果错误码未识别则返回 None。
    """
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
    """
    捕获当前工作区的 Git 状态快照。

    执行 `git status --porcelain -- .` 获取所有文件的 Git 状态码，
    并为每个受追踪的文件计算 SHA-256 内容哈希。

    这是变更检测机制的基础 —— 将执行前和执行后的两次快照进行对比，
    即可识别命令对工作区产生的所有文件级影响。

    Args:
        root (Path):
            工作区根目录（Git 仓库根目录）。

    Returns:
        _WorkspaceEffects:
            包含 Git 状态快照的数据对象。
            - 如果 Git 不可用或执行失败，返回 available=False 的空快照。
            - 如果成功，status 映射文件路径到 Git 状态码，
              hashes 映射文件路径到 SHA-256 哈希值。
    """
    try:
        # 运行 git status --porcelain 获取简洁的机器可读状态
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", "."],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,  # git status 通常很快，3 秒超时足够
            check=False,  # 不抛出异常，由 returncode 判断
        )
    except (OSError, subprocess.SubprocessError):
        # Git 不可用或执行失败（如 git 未安装、权限不足、仓库损坏等）
        return _WorkspaceEffects(False, {}, {})

    # 即使执行成功但 returncode 非零（如不在 Git 仓库中），也返回空快照
    if result.returncode != 0:
        return _WorkspaceEffects(False, {}, {})

    # 解析 git status --porcelain 输出
    # 格式: XY filename (X=暂存区状态, Y=工作区状态, 如 " M foo.py" 表示工作区修改)
    # 如果文件名包含 " -> "，说明是重命名，取箭头后的新文件名
    status: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if len(line) >= 4:
            # line[3:] 跳过前两个状态字符和一个空格，获取文件路径
            # split(" -> ") 处理重命名情况: "R  old -> new" -> 取 "new"
            status[line[3:].split(" -> ")[-1]] = line[:2]

    # 为所有受影响的文件计算内容哈希，便于精确检测文件内容变化
    hashes = {path: _path_fingerprint(root / path) for path in status}
    return _WorkspaceEffects(True, status, hashes)


def _compare_effects(
    root: Path,
    before: _WorkspaceEffects,
    after: _WorkspaceEffects,
) -> tuple[list[str], bool | None, str | None]:
    """
    对比执行前后的工作区快照，检测文件级变更。

    对比逻辑分为三层:
    1. 状态码对比: 同一文件在 before 和 after 中的 Git 状态码是否相同。
       不同则标记为受影响。
    2. 哈希对比: 即使状态码相同，内容哈希不同也说明文件被修改。
       （例如文件在 before 中状态为 "M "，执行后被进一步修改，
         Git 状态码不变但内容确实变了。）
    3. 新增文件检测: after 中存在但 before 中不存在的文件也加入受影响列表。

    此外，还生成 Git diff --stat 摘要，以人类可读的方式展示变更规模。

    Args:
        root (Path):
            工作区根目录。
        before (_WorkspaceEffects):
            命令执行前的工作区快照。
        after (_WorkspaceEffects):
            命令执行后的工作区快照。

    Returns:
        tuple[list[str], bool | None, str | None]:
            - affected (list[str]): 受影响的文件路径列表（已排序去重）。
            - changed (bool | None): 是否有文件变更（None 表示无法检测）。
            - summary (str | None): 变更摘要文本。
              如果 Git 不可用，返回 "Workspace effect detection unavailable (not a Git repository)"。
              如果有 diff 统计，返回 git diff --stat 的输出（截断至最后 1000 字符）。
              否则返回 "N workspace path(s) changed" 或 "No Git status change"。
    """
    # 如果前后任一快照不可用，无法进行变更检测
    if not before.available or not after.available:
        return [], None, "Workspace effect detection unavailable (not a Git repository)"

    # 第一层: 找出状态码或哈希发生变化的文件
    paths = sorted(
        path
        for path in set(before.status) | set(after.status)  # 取前后状态中所有出现过文件的并集
        # 状态码不同 OR 哈希不同 -> 文件发生了变化
        if before.status.get(path) != after.status.get(path)
        or before.hashes.get(path) != after.hashes.get(path)
    )

    # 第二层: 如果整体状态字典不同，将 after 中所有文件加入影响列表
    if before.status != after.status:
        paths = sorted(set(paths) | set(after.status))

    # 第三层: 生成 Git diff --stat 摘要（更直观的变更描述）
    stat = _git_diff_stat(root)
    summary = stat or (f"{len(paths)} workspace path(s) changed" if paths else "No Git status change")
    return paths, bool(paths), summary


def _change_evidence(
    root: Path,
    before: _WorkspaceEffects,
    after: _WorkspaceEffects,
    affected_paths: list[str],
) -> dict[str, Any]:
    """
    生成变更证据详情，提供受影响文件的前后哈希对比。

    对于每个受影响的文件，分别记录:
    - before 哈希: 优先使用文件系统快照中的哈希，如果缺失则从 Git HEAD 获取
      （如果文件在 before 快照中不存在但 after 中存在，则通过 git show HEAD:./path
      获取原始版本哈希，标记为 "<missing>" 表示新文件）
    - after 哈希: 使用 after 快照中的哈希（标记为 "<missing>" 如果文件已删除）

    这些证据可用于审计、调试和了解命令对文件的精确影响。

    Args:
        root (Path):
            工作区根目录。
        before (_WorkspaceEffects):
            执行前快照。
        after (_WorkspaceEffects):
            执行后快照。
        affected_paths (list[str]):
            受影响的文件路径列表。

    Returns:
        dict[str, Any]:
            包含以下键的变更证据字典:
            - change_kind: 变更类型（当前固定 "unknown"，保留扩展性）
            - before_hashes: {文件路径 -> 哈希值} 映射（执行前）
            - after_hashes: {文件路径 -> 哈希值} 映射（执行后）
            - affected_paths: 受影响路径列表
            - effect_detection: 检测方式（"git" 或 "unavailable"）
            - effect_detection_confidence: 检测置信度（"medium" 或 "low"）
            - safe_revert_available: 是否可安全回滚（当前始终 False）
    """
    # 构建 before 哈希: 优先用快照信息，缺失时回退到 Git HEAD 版本
    before_hashes = {
        path: before.hashes.get(path) or _git_head_fingerprint(root, path) or "<missing>"
        for path in affected_paths
    }
    # 构建 after 哈希: 直接使用 after 快照（文件不存在则为 "<missing>"，如被删除）
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
    """
    检测命令或 cwd 中是否硬编码了 /workspace 路径别名。

    在容器化或沙箱环境中，实际工作区路径可能不是 /workspace（例如
    用户指定了自定义目录或系统自动分配了不同路径）。如果命令中
    硬编码了 /workspace，会导致路径解析失败。

    检测规则:
    - 如果实际工作区路径本身就是 /workspace（或子目录），则不报错（允许使用）
    - 如果实际工作区路径不是 /workspace，但命令文本中或 cwd 参数中包含
      "/workspace" 字符串，则返回错误提示

    命令中的反斜杠会被统一转换为正斜杠，以兼容 Windows 路径。

    Args:
        command (str):
            用户输入的 shell 命令。
        cwd_text (str):
            用户指定的工作目录（cwd 参数值）。
        workspace (Path):
            实际的沙箱工作区根路径。

    Returns:
        str | None:
            如果检测到硬编码的 /workspace 路径且实际路径不匹配，返回错误提示消息。
            否则返回 None（表示路径使用正确）。
    """
    # 获取实际工作区路径的 POSIX 格式，去除末尾斜杠
    workspace_posix = workspace.resolve().as_posix().rstrip("/")
    # 如果实际路径就是 /workspace，则不需要报错——路径别名是合法的
    if workspace_posix == "/workspace" or workspace_posix.startswith("/workspace/"):
        return None

    # 将命令中的反斜杠统一转换为正斜杠，以便跨平台匹配
    command_text = command.replace("\\", "/")
    # 处理 cwd 文本，统一格式
    cwd = str(cwd_text).strip().replace("\\", "/").rstrip("/")

    # 检查命令文本或 cwd 中是否包含 /workspace
    if "/workspace" not in command_text and cwd != "/workspace" and not cwd.startswith("/workspace/"):
        return None

    # 返回错误提示，告知用户实际路径
    return (
        "Hard-coded /workspace is not available for this session. "
        f"Commands already run in the workspace cwd: {workspace.resolve()}."
    )


def _path_fingerprint(path: Path) -> str:
    """
    通过 SHA-256 计算文件的唯一内容指纹（哈希）。

    用于精确检测文件内容的字节级别变化。即使 Git 状态码相同，
    内容哈希不同也意味着文件被实际修改了。

    哈希计算策略:
    - 以 1MB 为块大小分块读取文件，避免大文件一次性加载占用过多内存
    - 使用 SHA-256（而非更快的 MD5 或 SHA-1）以确保足够的抗碰撞性

    Args:
        path (Path):
            要计算哈希的文件路径。

    Returns:
        str:
            文件的 SHA-256 十六进制哈希字符串。
            特殊情况:
            - "<missing>": 文件不存在
            - "<non-file>": 路径存在但不是普通文件（如目录、符号链接）
            - "<unreadable>": 文件存在但无法读取（权限不足等）
    """
    if not path.exists():
        return "<missing>"
    if not path.is_file():
        return "<non-file>"
    digest = hashlib.sha256()
    try:
        # 以 1MB 块大小分块读取，逐步更新哈希
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return "<unreadable>"
    return digest.hexdigest()


def _git_head_fingerprint(root: Path, relative_path: str) -> str | None:
    """
    从 Git HEAD 提交中获取文件的原始内容哈希。

    当 before 快照中某个文件不在 _workspace_effects 的 status 中
    （例如文件在命令执行前未被 Git 追踪），但 after 快照中出现时，
    此函数尝试从 Git HEAD 获取该文件的基准版本作为对比参考。

    通过 `git show HEAD:./relative_path` 获取 HEAD 提交中该文件的内容，
    然后计算其 SHA-256 哈希。

    Args:
        root (Path):
            Git 仓库根目录。
        relative_path (str):
            相对于仓库根目录的文件路径。

    Returns:
        str | None:
            HEAD 版本文件内容的 SHA-256 哈希。
            如果 Git 不可用、文件不在 HEAD 中或执行失败，返回 None。
    """
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
    # 文件可能不在 HEAD 中（新文件），此时 returncode 非零
    if result.returncode != 0:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def _git_diff_stat(root: Path) -> str | None:
    """
    获取 Git diff --stat 摘要，以人类可读的方式展示变更规模。

    执行 `git diff --stat -- .` 获取暂存区和工作区的合并差异统计。
    输出格式示例: "3 files changed, 42 insertions(+), 8 deletions(-)"

    结果被截断至最后 1000 字符，防止极端情况（如大量文件变更）下
    输出过长。

    Args:
        root (Path):
            Git 仓库根目录。

    Returns:
        str | None:
            git diff --stat 的输出文本（截断至最后 1000 字符）。
            如果 Git 不可用、不在仓库中或执行失败，返回 None。
            如果没有任何差异，返回 None。
    """
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
    # 截断至最后 1000 字符，保留最重要的摘要信息
    return text[-1000:] if text else None


async def _terminate_process(proc: asyncio.subprocess.Process | None) -> None:
    """
    跨平台终止异步子进程。

    当命令超时或被用户取消时调用，确保子进程和其所有子进程树被彻底终止。

    实现策略:
    - Windows (os.name == "nt"):
      使用 `taskkill /PID <pid> /T /F` 强制终止指定进程及其所有子进程。
      /T: 终止进程树（包括所有子进程）
      /F: 强制终止（不等待进程响应）
      由于 taskkill 本身也是子进程，使用 asyncio 创建并等待其完成。
    - Unix/Linux/macOS:
      使用 subprocess.Process.kill() 发送 SIGKILL 信号。

    容错处理:
    - 如果进程已经结束（proc.returncode is not None），直接返回
    - 如果终止失败（如进程已自行退出、PID 无效等），尝试再次 kill
    - 捕获 ProcessLookupError（进程不存在），静默忽略

    超时控制:
    - taskkill 命令有 5 秒超时
    - proc.communicate() 等待有 5 秒超时
    - 如果整体异常，再次尝试 proc.kill() 做最后的兜底

    Args:
        proc (asyncio.subprocess.Process | None):
            要终止的异步子进程。如果为 None 或进程已结束，不做任何操作。
    """
    # 进程不存在或已经结束，无需终止
    if proc is None or proc.returncode is not None:
        return
    try:
        if os.name == "nt":
            # Windows: 使用 taskkill 强制终止进程树
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(proc.pid),
                "/T",  # 终止进程树（进程及所有子进程）
                "/F",  # 强制终止
                stdout=asyncio.subprocess.DEVNULL,  # 丢弃输出
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(killer.communicate(), timeout=5)
        else:
            # Unix: 发送 SIGKILL 信号
            proc.kill()
        # 等待进程彻底退出（最多 5 秒）
        await asyncio.wait_for(proc.communicate(), timeout=5)
    except Exception:
        # 兜底: 如果以上步骤失败，再次尝试直接 kill
        try:
            proc.kill()
        except ProcessLookupError:
            # 进程可能已经不存在了，静默忽略
            pass


__all__ = ["create_shell_tools"]
