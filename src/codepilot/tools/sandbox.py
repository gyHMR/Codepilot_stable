"""工作区和 Shell 安全辅助模块（供工具系统使用）。

=============================================================================
模块概述
=============================================================================

本模块为 Codepilot 的 Shell 命令执行和文件系统操作提供了一套安全沙箱机制，
主要包含以下核心功能：

1. 工作区沙箱（WorkspaceSandbox）
   —— 将所有文件路径操作限制在工作区目录内，防止路径遍历攻击和越权访问。

2. Shell/argv 命令解析与统一 CommandAssessment
   —— 对用户/LLM 提交的 Shell 命令进行安全分级，分为验证类、只读类、
      变更类、高风险类和未知类五类，以决定命令的执行策略和审批流程。

3. Shell 执行策略（ShellExecutionPolicy）
   —— 定义 Shell 命令的超时限制、输出截断等运行时约束。

4. 环境变量过滤（build_shell_environment）
   —— 从系统环境变量中只暴露安全的白名单变量，防止密钥泄露。

5. 输出截断（truncate_output）
   —— 对过长的命令输出进行截断处理，保留头部和尾部信息。

6. 内部状态保护（command_mentions_internal_state）
   —— 检测命令是否试图修改 .codepilot 内部状态文件。

7. 文件状态采集（file_state_for_path）
   —— 以工作区相对路径为键，收集文件的元数据（大小、修改时间、SHA256 哈希）。

8. 敏感路径检查（is_sensitive_workspace_path, command_mentions_sensitive_path）
   —— 保护 .env、SSH 密钥、AWS 凭据等敏感文件不被工具读取或修改。

=============================================================================
安全设计原则
=============================================================================

- 纵深防御：路径解析、命令分类、环境过滤各层独立工作，任一层失效仍能兜底。
- 默认拒绝：无法明确归类为安全的命令一律标记为 "unknown"，交由上层审批。
- 最小权限：Shell 环境只暴露运行所必需的最少变量，不继承用户会话的全部环境。
- 白名单策略：命令分类使用前缀白名单匹配，而非黑名单排除，避免遗漏新危险命令。
"""

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

# ---------------------------------------------------------------------------
# ShellCommandClass 类型别名
# ---------------------------------------------------------------------------
# 定义 Shell 命令的安全分类级别：
#   - "verification": 验证类命令（如测试、lint、编译检查），仅读取/检查，无副作用
#   - "read_only":    只读类命令（如 ls、cat、grep），不修改文件系统
#   - "mutation":     变更类命令（如格式化、构建、代码生成），会修改文件但通常是安全的
#   - "high_risk":    高风险命令（如递归删除、强制推送），需要显式审批
#   - "unknown":      未知命令，无法自动判定安全性，需要人工审核
ShellCommandClass = Literal["verification", "read_only", "mutation", "high_risk", "unknown"]
CommandProfile = Literal[
    "inspection",
    "repository_execution",
    "bounded_mutation",
    "external_effect",
    "destructive",
    "unknown",
]


@dataclass(frozen=True)
class CommandAssessment:
    """命令评估结果 —— 对命令的安全等级、风险和副作用的综合评估。

    参数:
        profile: 命令的执行画像分类
        risk: 风险级别（low / medium / high / critical）
        effects: 命令可能产生的副作用集合
        requires_shell: 是否需要通过 Shell 执行（True=Shell命令，False=受控命令）
        destructive: 是否具有破坏性
        network: 是否需要网络访问
    """
    profile: CommandProfile
    risk: Literal["low", "medium", "high", "critical"]
    effects: frozenset[str]
    requires_shell: bool
    destructive: bool
    network: bool

# ---------------------------------------------------------------------------
# 内部保留目录名集合
# ---------------------------------------------------------------------------
# .codepilot 是 Codepilot 内部元数据和状态的存储目录，
# 禁止通过 Shell 命令直接修改其中的文件，必须通过 Session Store API 操作。
_INTERNAL_ROOTS = {".codepilot", ".git"}
_SENSITIVE_FILE_NAMES = {
    ".env",
    "credentials",
    "credentials.json",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "known_hosts",
}

# ---------------------------------------------------------------------------
# 高风险命令的正则模式列表
# ---------------------------------------------------------------------------
# 每个模式对应一类具有破坏性或不可逆操作的命令：
#   - rm -rf / rm -fr / del /f / rmdir /s: 递归强制删除
#   - Remove-Item -Recurse: PowerShell 递归删除
#   - format / mkfs: 磁盘格式化
#   - shutdown / reboot: 系统关机/重启
#   - git reset --hard / git clean -f: 强制丢弃工作区修改
#   - git push --force / -f: 强制推送覆盖远程历史
# 这些命令即使在正确的上下文中也应触发显式审批。
_HIGH_RISK_PATTERNS = (
    r"\brm\s+-(?:[a-z]*r[a-z]*f|[a-z]*f[a-z]*r)\b",
    r"\brm\s+-r\b",
    r"\bdel\s+/f\b",
    r"\brmdir\s+/s\b",
    r"\bremove-item\b.*\b-recurse\b",
    r"\bformat\s+[a-z]:",
    r"\bmkfs\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\s+-[a-z]*f",
    r"\bgit\s+push\b.*(?:--force|-f)\b",
)

# ---------------------------------------------------------------------------
# 验证类命令前缀白名单
# ---------------------------------------------------------------------------
# 这些命令仅进行检查/验证操作，不会产生副作用：
#   - Python 测试与编译检查（pytest, compileall）
#   - 代码质量检查（ruff check, mypy, pyright）
#   - Git 信息查询（status, diff, log, show, rev-parse）
#   - 其他语言的测试与构建检查（go test, cargo test, npm test/lint/build）
# 验证类命令可以自动执行，不需要审批。
_VERIFICATION_PREFIXES = (
    "pytest",
    "python -m pytest",
    "python -m compileall",
    "ruff check",
    "mypy",
    "pyright",
    "git status",
    "git diff",
    "git log",
    "git show",
    "git rev-parse",
    "go test",
    "cargo test",
    "npm test",
    "npm run test",
    "npm run lint",
    "npm run build",
)

# ---------------------------------------------------------------------------
# 只读类命令前缀白名单
# ---------------------------------------------------------------------------
# 这些命令只读取信息，不修改文件系统：
#   - 文件查看（type, cat, head, tail, Get-Content, gc）
#   - 目录浏览（dir, ls, pwd）
#   - 文本搜索（grep, rg, findstr, Select-String）
#   - 统计（wc）、路径查找（where）
#   - sed -n: sed 的只读模式（不进行原地替换）
#   - 版本信息（python --version, node --version, pip --version 等）
#   - Git 查询（git branch, git remote）
_READ_ONLY_PREFIXES = (
    "dir",
    "ls",
    "pwd",
    "type",
    "cat",
    "head",
    "tail",
    "wc",
    "grep",
    "rg",
    "sed -n",
    "where",
    "findstr",
    "select-string",
    "get-content",
    "gc",
    "python --version",
    "python -v",
    "py --version",
    "py -v",
    "pip --version",
    "node --version",
    "npm --version",
    "git branch",
    "git remote",
)

# ---------------------------------------------------------------------------
# 变更类命令前缀白名单
# ---------------------------------------------------------------------------
# 这些命令会修改文件，但属于安全的、有边界的操作：
#   - 代码格式化（ruff format, black, prettier, npm run format）
#   - Git 暂存（git add）
#   - 项目构建与代码生成（python -m build, npm run generate）
# 变更类命令通常不需要审批，但会被记录日志。
_MUTATION_PREFIXES = (
    "ruff format",
    "black ",
    "prettier ",
    "npm run format",
    "git add",
    "python -m build",
    "npm run generate",
)

# ---------------------------------------------------------------------------
# 安全环境变量名白名单
# ---------------------------------------------------------------------------
# 在构建 Shell 子进程环境时，只允许继承这些变量：
#   - PATH / PATHEXT: 可执行文件搜索路径
#   - SYSTEMROOT / WINDIR / COMSPEC: Windows 系统路径
#   - TEMP / TMP: 临时目录
#   - HOME / USERPROFILE: 用户主目录
#   - VIRTUAL_ENV: Python 虚拟环境路径
#   - PYTHONPATH: Python 模块搜索路径
# 所有其他环境变量（尤其是可能包含密钥的变量）将被过滤掉。
_SAFE_ENV_NAMES = {
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "TEMP",
    "TMP",
    "HOME",
    "USERPROFILE",
    "VIRTUAL_ENV",
    "PYTHONPATH",
}

# ---------------------------------------------------------------------------
# 敏感环境变量的关键字标记
# ---------------------------------------------------------------------------
# 即使变量名在安全白名单中，如果名称包含以下任一关键字，
# 也会被排除，以防止无意中泄露密钥或凭据：
#   TOKEN, SECRET, PASSWORD, API_KEY, CREDENTIAL, COOKIE
_SECRET_ENV_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "CREDENTIAL", "COOKIE")


# =========================================================================
# WorkspaceSandbox —— 工作区沙箱
# =========================================================================


@dataclass(frozen=True)
class WorkspaceSandbox:
    """工作区沙箱 —— 将所有文件路径操作限制在工作区目录内。

    核心职责：
    1. 路径解析（resolve_path）：将相对路径或绝对路径解析为工作区内的
       规范化的绝对路径。
    2. 边界检查（ensure_within_workspace）：验证路径确实在工作区树内，
       如果路径逃逸则抛出 ValueError。
    3. 相对路径计算（relative_path）：将工作区内的路径转换为相对于
       工作区根目录的 POSIX 风格路径字符串。
    4. 可变路径检查（ensure_mutable_path）：在边界检查基础上，额外禁止
       修改 .codepilot 内部目录下的文件。
    5. 可读路径检查（ensure_readable_path）：在边界检查基础上，额外禁止
       读取敏感文件（如 .env、SSH 密钥等）。

    使用示例:
        sandbox = WorkspaceSandbox("/home/user/project")
        safe = sandbox.resolve_path("../outside")  # 抛出 ValueError

    参数:
        workspace_dir: 工作区根目录的路径（字符串或 Path 对象）
    """

    workspace_dir: str | Path

    @property
    def root(self) -> Path:
        """获取工作区根目录的规范化绝对路径。

        每次访问都会调用 Path.resolve() 以确保符号链接和相对路径
        都被展开为真实的绝对路径，作为后续所有路径检查的基准。

        返回:
            规范化后的工作区绝对路径
        """
        return Path(self.workspace_dir).resolve()

    def resolve_path(self, path_text: str | Path) -> Path:
        """将用户输入的路径解析为工作区内的安全绝对路径。

        解析逻辑：
        1. 如果是绝对路径，直接规范化（resolve）
        2. 如果是相对路径，先拼接到工作区根目录再规范化
        3. 规范化后的结果通过 ensure_within_workspace 验证边界

        参数:
            path_text: 用户提供的路径（可以是字符串或 Path 对象）

        返回:
            验证通过的工作区内绝对路径

        抛出:
            ValueError: 如果路径逃逸到工作区之外（如 "../../etc/passwd"）
        """
        path = Path(path_text)
        target = path.resolve() if path.is_absolute() else (self.root / path).resolve()
        return self.ensure_within_workspace(target)

    def ensure_within_workspace(self, path: str | Path) -> Path:
        """验证给定路径是否在工作区边界内。

        使用 Path.relative_to() 来判断目标路径是否以工作区根目录为前缀。
        如果 relative_to 抛出 ValueError，说明路径逃逸到了工作区外部。

        注意：Path.relative_to() 不解析 ".." 符号 —— 但在此之前
        resolve_path() 已经对路径做了 resolve() 处理（展开 ".."），
        因此这里的检查是可靠的。

        参数:
            path: 待验证的路径

        返回:
            规范化后的路径（与输入路径相同，仅做类型保证）

        抛出:
            ValueError: 如果路径不在工作区内（消息为 "Path escapes workspace boundary"）
        """
        target = Path(path).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("Path escapes workspace boundary") from exc
        return target

    def relative_path(self, path: str | Path) -> str:
        """将工作区内的路径转换为相对路径的 POSIX 风格字符串。

        先通过 ensure_within_workspace 做边界检查，然后将路径转换为
        相对于工作区根目录的相对路径，最后用 as_posix() 统一为正斜杠格式
        （跨平台兼容，Windows 上反斜杠会被转为正斜杠）。

        参数:
            path: 工作区内的绝对路径或相对路径

        返回:
            相对于工作区根目录的 POSIX 风格路径字符串，如 "src/main.py"

        抛出:
            ValueError: 如果路径不在工作区内
        """
        return self.ensure_within_workspace(path).relative_to(self.root).as_posix()

    def ensure_mutable_path(self, path: str | Path) -> Path:
        """验证路径是否在可修改的范围内。

        在 ensure_within_workspace 边界检查基础上，额外限制：
        1. 不能修改 .codepilot 内部目录中的文件
           （.codepilot 存储会话状态、工具审批记录等内部数据，
           必须通过 Session Store API 修改）
        2. 不能修改敏感文件（如 .env、SSH 密钥等）
        3. 不能直接修改 .git 目录下的文件

        参数:
            path: 待验证的路径

        返回:
            验证通过的可修改路径

        抛出:
            ValueError: 如果路径在工作区外、指向内部状态目录或敏感文件
        """
        target = self.ensure_within_workspace(path)
        relative = target.relative_to(self.root)
        if relative.parts and relative.parts[0].lower() in _INTERNAL_ROOTS:
            raise ValueError("Internal workspace state cannot be modified by tools")
        self._ensure_not_sensitive(relative)
        return target

    def ensure_readable_path(self, path: str | Path) -> Path:
        """验证路径是否可读 —— 在工作区内且非敏感文件。

        用于只读操作（如 read、ls），确保路径在工作区内，
        且不指向 .env、SSH 密钥等敏感文件。

        参数:
            path: 待验证的路径

        返回:
            验证通过的可读路径

        抛出:
            ValueError: 如果路径在工作区外或指向敏感文件
        """
        target = self.ensure_within_workspace(path)
        self._ensure_not_sensitive(target.relative_to(self.root))
        return target

    @staticmethod
    def _ensure_not_sensitive(relative: Path) -> None:
        """静态方法 —— 检查路径是否指向敏感文件。

        通过 is_sensitive_workspace_path() 检测是否匹配敏感文件模式。

        参数:
            relative: 工作区相对路径

        抛出:
            ValueError: 如果路径被识别为敏感文件
        """
        if is_sensitive_workspace_path(relative):
            raise ValueError(f"Sensitive workspace file is protected: {relative.as_posix()}")


# =========================================================================
# ShellExecutionPolicy —— Shell 执行策略
# =========================================================================


@dataclass(frozen=True)
class ShellExecutionPolicy:
    """Shell 命令执行的运行时策略配置。

    定义了 Shell 命令执行的各项限制参数：
    - timeout_seconds: 默认超时时间（秒），如果用户未指定超时则使用此值
    - max_timeout_seconds: 允许的最大超时时间，用户指定的超时不能超过此值
    - stdout_limit: 标准输出最大字符数，超过则截断
    - stderr_limit: 标准错误输出最大字符数，超过则截断
    - allowed_env: 额外允许传递给子进程的环境变量名元组

    参数:
        timeout_seconds: 默认超时时间，默认 30 秒
        max_timeout_seconds: 最大允许超时时间，默认 120 秒
        stdout_limit: 标准输出截断限制，默认 20,000 字符
        stderr_limit: 标准错误截断限制，默认 10,000 字符
        allowed_env: 额外允许的环境变量名元组
    """

    timeout_seconds: int = 30
    max_timeout_seconds: int = 120
    stdout_limit: int = 20_000
    stderr_limit: int = 10_000
    allowed_env: tuple[str, ...] = ()

    def validate_timeout(self, requested: object) -> tuple[int | None, str | None]:
        """验证用户请求的超时值是否合法。

        验证规则（按顺序）：
        1. 如果 requested 为 None，返回默认超时时间
        2. 如果 requested 是布尔值，这显然是错误类型，返回 "invalid_timeout"
        3. 尝试将 requested 转为整数，如果失败返回 "invalid_timeout"
        4. 如果整数值 < 1 或 > max_timeout_seconds，返回 "invalid_timeout"
        5. 所有检查通过，返回 (验证后的超时值, None)

        参数:
            requested: 用户请求的超时值（None、int 或其他类型）

        返回:
            (有效的超时秒数 | None, 错误消息 | None)
        """
        if requested is None:
            return self.timeout_seconds, None
        if isinstance(requested, bool):
            return None, "invalid_timeout"
        try:
            value = int(requested)
        except (TypeError, ValueError):
            return None, "invalid_timeout"
        if value < 1 or value > self.max_timeout_seconds:
            return None, "invalid_timeout"
        return value, None


# =========================================================================
# TruncatedOutput —— 截断输出
# =========================================================================


@dataclass(frozen=True)
class TruncatedOutput:
    """截断后的输出描述。

    当 Shell 命令的输出超过指定长度限制时，输出会被截断，
    此类记录截断前后的元信息，以便上层了解输出是否完整。

    参数:
        text: 截断后的文本内容（如果未截断则为完整原文）
        truncated: 是否发生了截断
        original_chars: 原始输出的总字符数
        returned_chars: 截断后返回的字符数（包含截断标记文本）
    """

    text: str
    truncated: bool
    original_chars: int
    returned_chars: int


# =========================================================================
# 公共函数
# =========================================================================


def file_state_for_path(workspace_dir: str | Path, path: str | Path) -> dict[str, Any]:
    """收集指定路径文件的状态元数据。

    此函数在工作区沙箱内安全地解析路径，并返回文件的元数据字典，
    包括文件是否存在、大小、修改时间（纳秒精度）和 SHA256 哈希值。

    SHA256 计算使用流式读取（1MB 块），适用于大文件。

    如果文件不存在或不是常规文件，返回的字典中 "exists" 为 False。

    参数:
        workspace_dir: 工作区根目录路径
        path: 要检查的文件路径（绝对或相对路径）

    返回:
        包含以下键的字典：
        - "path": 文件相对于工作区根目录的 POSIX 路径
        - "exists": 文件是否存在且为常规文件
        - "size": 文件大小（字节），仅当 exists=True
        - "mtime_ns": 最后修改时间（纳秒），仅当 exists=True
        - "sha256": 文件内容的 SHA256 哈希，仅当 exists=True
        - "workspace_path": 工作区根目录的字符串表示
    """
    sandbox = WorkspaceSandbox(workspace_dir)
    target = sandbox.resolve_path(path)
    relative = target.relative_to(sandbox.root).as_posix()

    if not target.exists() or not target.is_file():
        return {"path": relative, "exists": False, "workspace_path": str(sandbox.root)}

    stat = target.stat()
    return {
        "path": relative,
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(target),
        "workspace_path": str(sandbox.root),
    }


def _classify_shell_syntax(command: str) -> ShellCommandClass:
    """Shell 命令安全分级 —— 对 Shell 命令进行安全分类。

    分类流程（按优先级从高到低）：
    1. 高风险检测 —— 用正则模式匹配检查是否包含 rm -rf、强制推送等操作
    2. 提取第一条有效命令 —— 处理命令链（&&、||、;、换行）和环境设置前缀
    3. 如果无法提取有效命令，返回 "unknown"
    4. 如果命令包含 Shell 重定向/管道（>、>>、<、|），返回 "unknown"
       （这些操作使白名单匹配不可靠，因为重定向目标可能产生副作用）
    5. 按顺序匹配白名单前缀：verification → read_only → mutation
    6. 都不匹配 → "unknown"

    参数:
        command: 用户提交的原始 Shell 命令字符串

    返回:
        "verification" | "read_only" | "mutation" | "high_risk" | "unknown"
    """
    normalized = " ".join(command.strip().lower().split())

    # 第一关：高风险模式匹配（优先级最高）
    if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in _HIGH_RISK_PATTERNS):
        return "high_risk"

    # 第二关：提取第一条有效命令
    first = _first_command(normalized)
    if not first:
        return "unknown"

    # 第三关：Shell 重定向/管道检测
    if _has_shell_redirection(first):
        return "unknown"

    # 第四关：白名单匹配
    if any(_matches_command_prefix(first, prefix) for prefix in _VERIFICATION_PREFIXES):
        return "verification"
    if any(_matches_command_prefix(first, prefix) for prefix in _READ_ONLY_PREFIXES):
        return "read_only"
    if any(_matches_command_prefix(first, prefix) for prefix in _MUTATION_PREFIXES):
        return "mutation"

    # 第五关：工作区内 Python 脚本检测
    if _is_python_workspace_script(first):
        return "mutation"

    return "unknown"


def validate_shell_command(command: str) -> CommandAssessment:
    """验证 Shell 命令 —— 执行不可绕过的安全检查并返回评估结果。

    三步检查：
    1. 调用 assess_command 进行安全分级
    2. 如果命令被归类为 destructive（破坏性），抛出异常
    3. 检查命令是否涉及内部状态文件或敏感文件

    参数:
        command: Shell 命令字符串

    返回:
        CommandAssessment 评估结果

    抛出:
        ValueError: 命令是高风险的、涉及内部状态或敏感文件
    """
    assessment = assess_command(command, requires_shell=True)
    if assessment.destructive:
        raise ValueError("Shell command is high-risk and cannot be approved")
    if command_mentions_internal_state(command):
        raise ValueError("Shell command targets internal workspace state")
    if command_mentions_sensitive_path(command):
        raise ValueError("Shell command targets a sensitive workspace file")
    return assessment


def _parse_argv_profile(argv: Sequence[str]) -> CommandProfile:
    """解析 argv 命令的执行画像 —— 基于参数列表分类命令。

    适用于受控命令工具（如 "command" 工具），
    通过检视 argv[0]（可执行文件）和参数来分类。

    参数:
        argv: 命令参数序列（如 ["git", "status"]）

    返回:
        CommandProfile 执行画像分类
    """
    if isinstance(argv, (str, bytes)) or not argv:
        return "unknown"
    parts = tuple(str(item).strip() for item in argv)
    if any(not item for item in parts):
        return "unknown"
    executable = _command_executable(parts[0])
    args = tuple(item.lower() for item in parts[1:])

    # 破坏性命令
    if executable in {"rm", "rmdir", "del", "format", "mkfs", "shutdown", "reboot"}:
        return "destructive"
    # Git 子命令分类
    if executable in {"git"} and args:
        action = args[0]
        if action == "reset" and "--hard" in args:
            return "destructive"
        if action == "clean" and any(item.startswith("-") and "f" in item for item in args[1:]):
            return "destructive"
        if action == "push" and any(item in {"--force", "-f", "--force-with-lease"} for item in args[1:]):
            return "destructive"
        if action in {"push", "pull", "fetch", "clone"}:
            return "external_effect"
        if action in {"status", "diff", "log", "show", "rev-parse", "branch", "remote"}:
            return "inspection"
        if action == "add":
            return "external_effect"

    # 网络命令
    if executable in {"curl", "wget"}:
        return "external_effect"
    # pip
    if executable in {"pip", "pip3"}:
        if args and args[0] in {"install", "uninstall", "download", "wheel"}:
            return "external_effect"
        if args and args[0] in {"--version", "-v", "list", "show"}:
            return "inspection"
    # npm/pnpm/yarn
    if executable in {"npm", "pnpm", "yarn"}:
        if args and args[0] in {"install", "uninstall", "update", "publish", "ci"}:
            return "external_effect"
        if args and args[0] == "run" and len(args) > 1:
            if args[1] in {"format", "generate", "test", "lint", "build"}:
                return "repository_execution"
        if args and args[0] == "test":
            return "repository_execution"
        if args and args[0] in {"--version", "-v"}:
            return "inspection"
    # 测试/检查工具
    if executable in {"pytest", "mypy", "pyright"}:
        return "repository_execution"
    if executable == "ruff":
        if args and args[0] == "format":
            return "bounded_mutation"
        if args and args[0] == "check":
            return "repository_execution"
    if executable in {"black", "prettier"}:
        return "bounded_mutation"
    if executable == "go" and args and args[0] == "test":
        return "repository_execution"
    if executable == "cargo" and args and args[0] in {"test", "check"}:
        return "repository_execution"
    if executable in {"rg", "grep", "findstr"}:
        return "inspection"
    # Python 调用
    if executable in {"python", "python3", "py"}:
        if args and args[0] in {"--version", "-v"}:
            return "inspection"
        if len(args) >= 2 and args[0] == "-m":
            module = args[1]
            if module in {"pytest", "compileall"}:
                return "repository_execution"
            if module == "build":
                return "repository_execution"
            if module == "pip" and len(args) >= 3:
                if args[2] in {"install", "uninstall", "download", "wheel"}:
                    return "external_effect"
                if args[2] in {"--version", "list", "show"}:
                    return "inspection"
    if args and args[0] in {"--version", "-v"}:
        return "inspection"
    return "unknown"


def validate_controlled_command(
    argv: Sequence[str],
    *,
    sandbox: WorkspaceSandbox,
    cwd: str | Path = ".",
) -> tuple[tuple[str, ...], Path, CommandAssessment]:
    """验证受控命令 —— 对一个参数式命令及其路径参数做全面安全验证。

    受控命令（command 工具）相比完整的 Shell 命令更安全，
    因为参数是显式提供的，不需要 Shell 解析。本函数：
    1. 过滤空参数
    2. 评估命令的 SecurityProfile
    3. 拒绝破坏性命令和未知命令
    4. 验证工作目录在工作区内
    5. 对每个路径参数做沙箱检查

    参数:
        argv: 命令参数序列
        sandbox: 工作区沙箱实例
        cwd: 命令的工作目录（默认当前目录）

    返回:
        (规范化后的参数元组, 验证通过的工作目录, 命令评估结果)
    """
    if isinstance(argv, (str, bytes)) or not argv:
        raise ValueError("Controlled command argv must be a non-empty string array")
    normalized = tuple(str(item).strip() for item in argv)
    if any(not item for item in normalized):
        raise ValueError("Controlled command argv cannot contain empty values")
    assessment = assess_command(normalized, requires_shell=False)
    if assessment.destructive:
        raise ValueError("Destructive commands are not available through the command tool")
    if assessment.profile == "unknown":
        raise ValueError("Unsupported controlled command; use bash with approval")

    working_dir = sandbox.ensure_mutable_path(sandbox.resolve_path(cwd))
    if not working_dir.is_dir():
        raise ValueError("Controlled command cwd must be a workspace directory")
    for raw in normalized[1:]:
        _validate_command_path_argument(
            raw,
            sandbox=sandbox,
            cwd=working_dir,
            mutating=assessment.profile != "inspection",
        )
    return normalized, working_dir, assessment


def assess_command(
    command: str | Sequence[str],
    *,
    requires_shell: bool,
) -> CommandAssessment:
    """统一命令评估函数 —— 对 Shell 命令和 argv 命令都适用的安全评估。

    根据 requires_shell 参数选择不同的评估路径：
    - True: 使用 Shell 语法分类（_classify_shell_syntax）
    - False: 使用 argv 画像解析（_parse_argv_profile）

    然后统一计算风险等级、副作用集合和网络需求。

    参数:
        command: Shell 命令字符串（requires_shell=True）或参数序列（requires_shell=False）
        requires_shell: True=Shell 命令评估，False=argv 命令评估

    返回:
        CommandAssessment 包含完整的安全评估结果
    """
    if requires_shell:
        if not isinstance(command, str):
            raise TypeError("Shell command assessment expects a string")
        shell_class = _classify_shell_syntax(command)
        profile: CommandProfile = {
            "verification": "repository_execution",
            "read_only": "inspection",
            "mutation": "repository_execution",
            "high_risk": "destructive",
            "unknown": "unknown",
        }[shell_class]
    else:
        if isinstance(command, (str, bytes)):
            raise TypeError("argv command assessment expects a sequence")
        profile = _parse_argv_profile(command)
    effects = {"process_spawn", "filesystem_read"}
    if profile in {"repository_execution", "bounded_mutation", "external_effect", "unknown"}:
        effects.add("filesystem_write")
    network = profile in {"repository_execution", "external_effect", "unknown"}
    if network:
        effects.update({"network_access", "external_state_write"})
    risk = (
        "low" if profile == "inspection"
        else "medium" if profile in {"repository_execution", "bounded_mutation"}
        else "high"
    )
    return CommandAssessment(
        profile=profile,
        risk=risk,
        effects=frozenset(effects),
        requires_shell=requires_shell,
        destructive=profile == "destructive",
        network=network,
    )


def is_sensitive_workspace_path(path: str | Path) -> bool:
    """检查路径是否是敏感的工作区文件。

    敏感文件包括：
    - .env 及其变体（.env.local 等，但 .env.example 不算）
    - SSH 密钥文件（id_rsa, id_ed25519 等）
    - 凭据文件（credentials, credentials.json）
    - 证书文件（*.key, *.p12, *.pfx）
    - SSH 配置目录下的文件（.ssh/）
    - AWS 凭据文件（.aws/credentials）
    - Git 配置文件（.git/config, .git/credentials）

    参数:
        path: 待检查的路径（字符串或 Path）

    返回:
        True 表示路径是敏感文件
    """
    relative = Path(path)
    name = relative.name.lower()
    parts = tuple(part.lower() for part in relative.parts)
    if name == ".env" or (
        name.startswith(".env.")
        and not name.endswith((".example", ".sample", ".template"))
    ):
        return True
    if name in _SENSITIVE_FILE_NAMES or name.endswith((".key", ".p12", ".pfx")):
        return True
    if ".ssh" in parts or (".aws" in parts and name == "credentials"):
        return True
    if len(parts) >= 2 and parts[0] == ".git" and name in {"config", "credentials"}:
        return True
    return False


def command_mentions_sensitive_path(command: str) -> bool:
    """检查 Shell 命令是否引用了敏感文件路径。

    通过字符串匹配检测常见的敏感文件路径模式。

    参数:
        command: 原始 Shell 命令字符串

    返回:
        True 表示命令中出现了敏感路径
    """
    text = command.replace("\\", "/").lower()
    markers = (
        "/.env",
        " .env",
        "/id_rsa",
        "/id_ed25519",
        "/credentials",
        ".ssh/",
        ".aws/credentials",
    )
    return any(marker in text for marker in markers)


def build_shell_environment(extra_allowed: tuple[str, ...] = ()) -> dict[str, str]:
    """构建安全的 Shell 子进程环境变量字典。

    过滤逻辑（对每个系统环境变量依次检查）：
    1. 变量名必须在 _SAFE_ENV_NAMES 白名单或 extra_allowed 中
    2. 变量名不能包含 _SECRET_ENV_MARKERS 中的敏感关键字
    3. 通过检查的值原样保留

    安全考量：
    - 白名单策略确保不会意外将包含密钥的环境变量传给子进程
    - 敏感关键字过滤作为第二道防线
    - 返回的字典仅包含安全且非敏感的变量

    参数:
        extra_allowed: 额外允许的变量名元组（自动转为大写匹配）

    返回:
        过滤后的安全环境变量字典
    """
    names = _SAFE_ENV_NAMES | {name.upper() for name in extra_allowed}

    result: dict[str, str] = {}
    for name, value in os.environ.items():
        upper = name.upper()

        if upper not in names:
            continue

        if any(marker in upper for marker in _SECRET_ENV_MARKERS):
            continue

        result[name] = value

    return result


def truncate_output(text: str, limit: int) -> TruncatedOutput:
    """对过长的输出文本进行截断处理（head-tail 截断策略）。

    截断策略：
    - 如果文本长度 <= limit，直接返回
    - 如果长度 > limit：
      1. 保留前 head_size 个字符
      2. 保留后 tail_size 个字符
      3. 中间插入截断标记 "...<truncated N chars>..."
    - 截断标记的长度不计入 limit

    这种策略确保用户能看到输出的开头（关键信息）和结尾（错误信息）。

    注意：此截断是字符级别，适用于中文等多字节字符。

    参数:
        text: 原始输出文本
        limit: 允许的最大字符数

    返回:
        TruncatedOutput 对象（含截断后的文本和元数据）
    """
    original = len(text)
    if original <= limit:
        return TruncatedOutput(text, False, original, original)

    head_size = max(1, limit // 2)
    tail_size = max(1, limit - head_size)
    omitted = max(0, original - head_size - tail_size)

    marker = f"\n...<truncated {omitted} chars>...\n"
    rendered = text[:head_size] + marker + text[-tail_size:]

    return TruncatedOutput(rendered, True, original, len(rendered))


def command_mentions_internal_state(command: str) -> bool:
    """检测 Shell 命令是否试图修改 .codepilot 内部状态文件。

    两个条件必须同时满足：
    1. 命令文本中包含 ".codepilot/"
    2. 命令中包含写入/修改操作（>>、>、copy、move、mv、sed -i）

    参数:
        command: 原始 Shell 命令

    返回:
        True 表示命令可能修改内部状态，应被阻止
    """
    text = command.replace("\\", "/").lower()
    return ".codepilot/" in text and bool(re.search(r"(?:>>?|copy|move|mv|sed\s+-i)", text))


# =========================================================================
# 内部辅助函数
# =========================================================================


def _sha256_file(path: Path) -> str:
    """计算文件的 SHA256 哈希值（流式读取，适用于大文件）。

    使用 iter(lambda: f.read(1MB), b"") 的惯用法实现流式读取，
    内存占用恒定（约 1MB），不会因为大文件导致 OOM。

    参数:
        path: 目标文件 Path

    返回:
        SHA256 十六进制摘要字符串（64 字符）
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_executable(value: str) -> str:
    """从命令字符串中提取规范化后的可执行文件名。

    去除扩展名（.exe, .cmd, .bat）并转小写。

    参数:
        value: 可执行文件路径字符串

    返回:
        规范化的可执行文件名
    """
    name = Path(value).name.lower()
    for suffix in (".exe", ".cmd", ".bat"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _validate_command_path_argument(
    raw: str,
    *,
    sandbox: WorkspaceSandbox,
    cwd: Path,
    mutating: bool,
) -> None:
    """验证受控命令的一个路径参数是否在沙箱内。

    验证逻辑：
    1. 去除 Shell 引号
    2. 跳过以 "-" 开头的标志（除非有 "=" 的值绑定）
    3. 跳过 URL 格式的参数
    4. 跳过看起来不像路径的参数
    5. 对看起来像路径的参数（绝对路径、含 ".."、含 "/" 或 "\"）
       执行沙箱边界检查

    参数:
        raw: 原始参数值
        sandbox: 工作区沙箱
        cwd: 当前工作目录
        mutating: 是否为可变操作

    抛出:
        ValueError: 路径参数逃逸出工作区或涉及内部状态
    """
    value = _strip_shell_quotes(raw.strip())
    if not value or value.startswith("-") and "=" not in value:
        return
    if value.startswith("-") and "=" in value:
        value = value.split("=", 1)[1]
    if not value or "://" in value:
        return
    candidate = Path(value)
    looks_like_path = (
        candidate.is_absolute()
        or ".." in candidate.parts
        or "/" in value
        or "\\" in value
    )
    if not looks_like_path:
        return
    target = candidate if candidate.is_absolute() else cwd / candidate
    target = sandbox.ensure_within_workspace(target)
    if mutating:
        sandbox.ensure_mutable_path(target)
    else:
        sandbox.ensure_readable_path(target)


def _matches_command_prefix(command: str, prefix: str) -> bool:
    """检查命令是否以指定前缀开头（精确前缀匹配）。

    匹配规则：
    - 规范前缀后（去空格、小写、压缩空格）匹配命令
    - 完全匹配或前缀后跟空格都视为匹配
    - "git status" 匹配 "git status" 和 "git status --short"
    - "git status" 不匹配 "git statuscheck"

    参数:
        command: 规范化后的命令
        prefix: 白名单前缀

    返回:
        True 表示匹配成功
    """
    prefix = " ".join(prefix.strip().lower().split())
    return command == prefix or command.startswith(prefix + " ")


def _has_shell_redirection(command: str) -> bool:
    """检测命令是否包含 Shell 重定向或管道操作。

    检测操作符：>>、>（输出重定向）、<（输入重定向）、|（管道）

    参数:
        command: 规范化后的命令

    返回:
        True 表示包含重定向或管道
    """
    return bool(re.search(r"(?:>>?|<|\|)", command))


def _first_command(command: str) -> str:
    """从命令链中提取第一条有效命令。

    处理流程：
    1. 按 &&、||、;、换行拆分为命令段
    2. 过滤安全的前置设置命令（set PATH=..., cd, export 等）
    3. 对剩余命令段做递归分类，综合判定安全性

    参数:
        command: 规范化后的命令字符串

    返回:
        提取的第一个有效命令，或 "<compound>" 表示无法判定
    """
    segments = [
        segment.strip()
        for segment in re.split(r"(?:&&|\|\||;|\r?\n)", command)
        if segment.strip()
    ]
    if not segments:
        return ""

    if len(segments) == 1:
        return segments[0]

    command_segments = [
        segment
        for segment in segments
        if not _is_safe_env_setup(segment) and not _is_safe_directory_setup(segment)
    ]

    if not command_segments:
        return "unknown"

    classes = [_classify_shell_syntax(segment) for segment in command_segments]

    if all(item in {"verification", "read_only", "mutation"} for item in classes):
        if "mutation" in classes:
            return command_segments[classes.index("mutation")]
        if "verification" in classes:
            return command_segments[classes.index("verification")]
        return command_segments[0]

    return "<compound>"


def _is_safe_env_setup(command: str) -> bool:
    """检查命令是否仅为安全的环境变量设置。

    支持的语法：
    - set PYTHONPATH=...（CMD）
    - $env:PYTHONPATH = ...（PowerShell）
    - export PYTHONPATH=...（Unix）

    参数:
        command: 规范化后的命令

    返回:
        True 表示是安全的环境变量设置
    """
    normalized = " ".join(command.strip().lower().split())
    return bool(
        re.fullmatch(r"set\s+pythonpath=.*", normalized)
        or re.fullmatch(r"\$env:pythonpath\s*=.*", normalized)
        or re.fullmatch(r"export\s+pythonpath=.*", normalized)
    )


def _is_safe_directory_setup(command: str) -> bool:
    """检查命令是否仅为安全的目录切换。

    参数:
        command: 规范化后的命令

    返回:
        True 表示是安全的目录切换
    """
    target = _directory_setup_target(command)
    return target is not None and _is_safe_relative_shell_path(target)


def _directory_setup_target(command: str) -> str | None:
    """从目录切换命令中提取目标路径。

    支持：cd、chdir、pushd 及 PowerShell 变体。

    参数:
        command: 规范化后的命令

    返回:
        提取的路径字符串，或 None
    """
    normalized = " ".join(command.strip().lower().split())
    match = re.fullmatch(r"(?:cd|chdir|pushd)\s+(?:/d\s+)?(.+)", normalized)
    if match is None:
        return None
    return _strip_shell_quotes(match.group(1))


def _is_python_workspace_script(command: str) -> bool:
    """检查命令是否是对工作区内 Python 脚本的调用。

    匹配模式：python/python3/py 后跟 .py 文件路径。

    参数:
        command: 规范化后的命令

    返回:
        True 表示是工作区内的 Python 脚本调用
    """
    normalized = " ".join(command.strip().lower().split())
    match = re.fullmatch(r"(?:python|python3|py)\s+([^\s]+\.py)(?:\s+.*)?", normalized)
    if match is None:
        return False
    return _is_safe_relative_shell_path(_strip_shell_quotes(match.group(1)))


def _strip_shell_quotes(value: str) -> str:
    """去除字符串两端的 Shell 引号（单引号或双引号）。

    如 '"hello"' → "hello", "'hello'" → "hello"

    参数:
        value: 可能含引号的字符串

    返回:
        去除引号后的字符串
    """
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1].strip()
    return text


def _is_safe_relative_shell_path(value: str) -> bool:
    """验证字符串是否为安全的相对 Shell 路径。

    安全检查：
    1. 不能为空
    2. 不能以 "/" 开头（Unix 绝对路径）
    3. 不能以 "~" 开头（用户主目录）
    4. 不能匹配驱动器号（如 "C:"）
    5. 不能含 Shell 特殊字符（$、%、`、|、&、;、<、>）
    6. 不能含 ".."（路径回溯）
    7. 必须至少有一个有效部分

    参数:
        value: 待验证的路径字符串

    返回:
        True 表示是安全的相对路径
    """
    text = value.strip().replace("\\", "/")

    if not text or text.startswith(("/", "~")) or re.match(r"^[a-z]:", text):
        return False

    if any(marker in text for marker in ("$", "%", "`", "|", "&", ";", "<", ">")):
        return False

    parts = [part for part in text.split("/") if part]
    return bool(parts) and ".." not in parts


# ---------------------------------------------------------------------------
# 模块导出列表
# ---------------------------------------------------------------------------
__all__ = [
    "CommandAssessment",
    "CommandProfile",
    "ShellCommandClass",
    "ShellExecutionPolicy",
    "TruncatedOutput",
    "WorkspaceSandbox",
    "build_shell_environment",
    "assess_command",
    "command_mentions_internal_state",
    "file_state_for_path",
    "truncate_output",
    "validate_controlled_command",
    "validate_shell_command",
]