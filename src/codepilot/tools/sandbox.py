from __future__ import annotations

"""
工作区和 Shell 安全辅助模块（供工具系统使用）。

=============================================================================
模块概述
=============================================================================

本模块为 Codepilot 的 Shell 命令执行和文件系统操作提供了一套安全沙箱机制，
主要包含以下核心功能：

1. 工作区沙箱（WorkspaceSandbox）
   —— 将所有文件路径操作限制在工作区目录内，防止路径遍历攻击和越权访问。

2. Shell 命令分类（classify_shell_command）
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
from typing import Any, Literal

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


@dataclass(frozen=True)
class WorkspaceSandbox:
    """
    工作区沙箱 —— 将所有文件路径操作限制在工作区目录内。

    核心职责：
    1. 路径解析（resolve_path）：将相对路径或绝对路径解析为工作区内的
       规范化的绝对路径。
    2. 边界检查（ensure_within_workspace）：验证路径确实在工作区树内，
       如果路径逃逸则抛出 ValueError。
    3. 相对路径计算（relative_path）：将工作区内的路径转换为相对于
       工作区根目录的 POSIX 风格路径字符串。
    4. 可变路径检查（ensure_mutable_path）：在边界检查基础上，额外禁止
       修改 .codepilot 内部目录下的文件。

    使用示例:
        sandbox = WorkspaceSandbox("/home/user/project")
        safe = sandbox.resolve_path("../outside")  # 抛出 ValueError
    """

    #: 工作区根目录的绝对路径（字符串或 Path 对象）
    workspace_dir: str | Path

    @property
    def root(self) -> Path:
        """
        返回工作区根目录的规范化绝对路径。

        每次访问都会调用 Path.resolve() 以确保符号链接和相对路径
        都被展开为真实的绝对路径，作为后续所有路径检查的基准。
        """
        return Path(self.workspace_dir).resolve()

    def resolve_path(self, path_text: str | Path) -> Path:
        """
        将用户输入的路径解析为工作区内的安全绝对路径。

        解析逻辑：
        1. 如果是绝对路径，直接规范化（resolve）
        2. 如果是相对路径，先拼接到工作区根目录再规范化
        3. 规范化后的结果通过 ensure_within_workspace 验证边界

        参数:
            path_text: 用户提供的路径（可以是字符串或 Path 对象）

        返回:
            验证通过的工作区内绝对路径

        抛出:
            ValueError: 如果路径逃逸到工作区之外
        """
        path = Path(path_text)
        target = path.resolve() if path.is_absolute() else (self.root / path).resolve()
        return self.ensure_within_workspace(target)

    def ensure_within_workspace(self, path: str | Path) -> Path:
        """
        验证给定路径是否在工作区边界内。

        使用 Path.relative_to() 来判断目标路径是否以工作区根目录为前缀。
        如果 relative_to 抛出 ValueError，说明路径逃逸到了工作区外部，
        此时会重新抛出带有明确错误消息的 ValueError。

        注：Path.relative_to() 不解析 ".." 符号 —— 但在此之前
        resolve_path() 已经对路径做了 resolve() 处理，将 ".." 展开为
        实际目录，因此这里的检查是可靠的。

        参数:
            path: 待验证的路径

        返回:
            规范化后的路径（与输入相同，仅做类型保证）

        抛出:
            ValueError: 如果路径不在工作区内（"Path escapes workspace boundary"）
        """
        target = Path(path).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("Path escapes workspace boundary") from exc
        return target

    def relative_path(self, path: str | Path) -> str:
        """
        将工作区内的路径转换为相对于工作区根目录的 POSIX 风格路径字符串。

        先通过 ensure_within_workspace 做边界检查，然后将路径转换为
        相对于工作区根目录的相对路径，最后用 as_posix() 统一为正斜杠格式
        （跨平台兼容）。

        参数:
            path: 工作区内的绝对路径或相对路径

        返回:
            相对于工作区根目录的 POSIX 风格路径字符串，例如 "src/main.py"
        """
        return self.ensure_within_workspace(path).relative_to(self.root).as_posix()

    def ensure_mutable_path(self, path: str | Path) -> Path:
        """
        验证路径是否在可修改的范围内 —— 在边界检查基础上，
        额外禁止修改 .codepilot 内部目录中的文件。

        设计原因：
        .codepilot 目录存储会话状态、工具审批记录等关键内部数据，
        必须通过 Session Store API 进行修改以保证数据一致性，
        不允许外部 Shell 命令直接操作。

        参数:
            path: 待验证的路径

        返回:
            验证通过的可修改路径

        抛出:
            ValueError: 如果路径在工作区外，或指向 .codepilot 内部文件
        """
        target = self.ensure_within_workspace(path)
        relative = target.relative_to(self.root)
        if relative.parts and relative.parts[0].lower() in _INTERNAL_ROOTS:
            raise ValueError("Internal workspace state cannot be modified by tools")
        self._ensure_not_sensitive(relative)
        return target

    def ensure_readable_path(self, path: str | Path) -> Path:
        """Validate a workspace path and reject credential-like sensitive files."""

        target = self.ensure_within_workspace(path)
        self._ensure_not_sensitive(target.relative_to(self.root))
        return target

    @staticmethod
    def _ensure_not_sensitive(relative: Path) -> None:
        if is_sensitive_workspace_path(relative):
            raise ValueError(f"Sensitive workspace file is protected: {relative.as_posix()}")


@dataclass(frozen=True)
class ShellExecutionPolicy:
    """
    Shell 命令执行的运行时策略配置。

    定义了 Shell 命令执行的各项限制参数：
    - timeout_seconds: 默认超时时间（秒），如果用户未指定超时则使用此值
    - max_timeout_seconds: 允许的最大超时时间，用户指定的超时不能超过此值
    - stdout_limit: 标准输出最大字符数，超过则截断
    - stderr_limit: 标准错误输出最大字符数，超过则截断
    - allowed_env: 额外允许传递给子进程的环境变量名元组
    """

    #: 默认超时时间（秒），用户未指定超时时的回退值
    timeout_seconds: int = 30
    #: 允许的最大超时时间（秒），防止用户设置过长超时导致资源占用
    max_timeout_seconds: int = 120
    #: 标准输出最大保留字符数
    stdout_limit: int = 20_000
    #: 标准错误输出最大保留字符数
    stderr_limit: int = 10_000
    #: 除了 _SAFE_ENV_NAMES 白名单之外额外允许的环境变量名
    allowed_env: tuple[str, ...] = ()

    def validate_timeout(self, requested: object) -> tuple[int | None, str | None]:
        """
        验证用户请求的超时值是否合法。

        验证规则（按顺序）：
        1. 如果 requested 为 None，返回默认超时时间
        2. 如果 requested 是布尔值，这显然是错误类型，返回 "invalid_timeout"
        3. 尝试将 requested 转为整数，如果失败（如字符串、浮点数等），返回 "invalid_timeout"
        4. 如果整数值 < 1 或 > max_timeout_seconds，返回 "invalid_timeout"
        5. 所有检查通过，返回 (验证后的超时值, None)

        返回值是一个元组 (timeout_value, error_message)：
        - 成功时 error_message 为 None，timeout_value 为有效的超时秒数
        - 失败时 timeout_value 为 None，error_message 为 "invalid_timeout"

        参数:
            requested: 用户请求的超时值（可以是 None、int、或其他类型）

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


@dataclass(frozen=True)
class TruncatedOutput:
    """
    截断后的输出描述。

    当 Shell 命令的输出超过指定长度限制时，输出会被截断，
    此类记录截断前后的元信息，以便上层了解输出是否完整。

    属性:
        text: 截断后的文本内容（如果未截断则为完整原文）
        truncated: 是否发生了截断
        original_chars: 原始输出的总字符数
        returned_chars: 截断后返回的字符数（包含截断标记文本）
    """

    #: 截断后（或未截断的原始）文本
    text: str
    #: 是否发生了输出截断
    truncated: bool
    #: 原始输出的字符总数
    original_chars: int
    #: 截断后实际返回的字符总数（包含截断提示标记）
    returned_chars: int


# =========================================================================
# 公共函数
# =========================================================================


def file_state_for_path(workspace_dir: str | Path, path: str | Path) -> dict[str, Any]:
    """
    收集指定路径文件的状态元数据。

    此函数在工作区沙箱内安全地解析路径，并返回文件的元数据字典，
    包括文件是否存在、大小、修改时间（纳秒精度）和 SHA256 哈希值。

    使用场景：
    - 在执行 Shell 命令前后采集文件状态快照，用于变更检测
    - 为工具调用的文件参数提供元数据（如确认文件是否已有内容）
    - 计算文件内容哈希用于去重或完整性验证

    SHA256 计算使用流式读取（1MB 块），适用于大文件。

    如果文件不存在或不是常规文件，返回的字典中 "exists" 为 False，
    且不包含 size/mtime_ns/sha256 字段。

    参数:
        workspace_dir: 工作区根目录路径
        path: 要检查的文件路径（可以是绝对路径或相对路径）

    返回:
        包含以下键的字典：
        - "path": 文件相对于工作区根目录的路径（POSIX 风格）
        - "exists": 文件是否存在且为常规文件
        - "size": 文件大小（字节），仅当 exists=True 时存在
        - "mtime_ns": 最后修改时间（纳秒时间戳），仅当 exists=True 时存在
        - "sha256": 文件内容的 SHA256 十六进制哈希，仅当 exists=True 时存在
        - "workspace_path": 工作区根目录的字符串表示
    """
    # 使用 WorkspaceSandbox 确保路径在工作区内
    sandbox = WorkspaceSandbox(workspace_dir)
    target = sandbox.resolve_path(path)
    relative = target.relative_to(sandbox.root).as_posix()

    # 文件不存在或不是常规文件：返回最小化信息
    if not target.exists() or not target.is_file():
        return {"path": relative, "exists": False, "workspace_path": str(sandbox.root)}

    # 收集文件元数据：大小、修改时间、内容哈希
    stat = target.stat()
    return {
        "path": relative,
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(target),
        "workspace_path": str(sandbox.root),
    }


def classify_shell_command(command: str) -> ShellCommandClass:
    """
    对 Shell 命令进行安全分级。

    分类流程（按优先级从高到低）：
    1. 高风险检测 —— 用正则模式匹配检查命令是否包含高风险操作
       （如 rm -rf、强制推送等），匹配到则返回 "high_risk"
    2. 提取第一条有效命令 —— 通过 _first_command() 处理命令链
       （&&、||、;、换行）和环境设置前缀
    3. 如果无法提取有效命令，返回 "unknown"
    4. 如果命令包含 Shell 重定向/管道（>、>>、<、|），返回 "unknown"
       （因为有副作用的可能性增加，白名单匹配不再可靠）
    5. 按顺序匹配白名单前缀：
       a. 验证类前缀 → "verification"
       b. 只读类前缀 → "read_only"
       c. 变更类前缀 → "mutation"
       d. Python 工作区脚本 → "mutation"
    6. 都不匹配 → "unknown"

    命令预处理：去除首尾空格、全部转小写、压缩连续空格为单个空格，
    以确保匹配的一致性。

    参数:
        command: 用户提交的原始 Shell 命令字符串

    返回:
        ShellCommandClass 分类标签（"verification" | "read_only" |
        "mutation" | "high_risk" | "unknown"）
    """
    # 规范化：去首尾空格 → 小写 → 压缩连续空格
    normalized = " ".join(command.strip().lower().split())

    # 第一关：高风险模式匹配（优先级最高，直接返回）
    if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in _HIGH_RISK_PATTERNS):
        return "high_risk"

    # 第二关：提取第一条有效命令（剥离环境变量设置和目录切换前缀）
    first = _first_command(normalized)
    if not first:
        return "unknown"

    # 第三关：Shell 重定向/管道检测 —— 有这些操作则放弃白名单匹配
    if _has_shell_redirection(first):
        return "unknown"

    # 第四关：按顺序匹配安全命令白名单
    if any(_matches_command_prefix(first, prefix) for prefix in _VERIFICATION_PREFIXES):
        return "verification"
    if any(_matches_command_prefix(first, prefix) for prefix in _READ_ONLY_PREFIXES):
        return "read_only"
    if any(_matches_command_prefix(first, prefix) for prefix in _MUTATION_PREFIXES):
        return "mutation"

    # 第五关：检测是否为工作区内的 Python 脚本调用（如 python src/script.py）
    if _is_python_workspace_script(first):
        return "mutation"

    # 无法匹配任何已知安全模式，标记为未知
    return "unknown"


def validate_shell_command(command: str) -> ShellCommandClass:
    """Apply non-bypassable Shell safety checks before permission approval."""

    shell_class = classify_shell_command(command)
    if shell_class == "high_risk":
        raise ValueError("Shell command is high-risk and cannot be approved")
    if command_mentions_internal_state(command):
        raise ValueError("Shell command targets internal workspace state")
    if command_mentions_sensitive_path(command):
        raise ValueError("Shell command targets a sensitive workspace file")
    return shell_class


def is_sensitive_workspace_path(path: str | Path) -> bool:
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
    """
    构建安全的 Shell 子进程环境变量字典。

    过滤逻辑（按顺序对每个系统环境变量执行）：
    1. 变量名必须在 _SAFE_ENV_NAMES 白名单中，或者是 extra_allowed 中的额外变量
       （比较时均转换为大写进行大小写不敏感匹配）
    2. 变量名（大写形式）不能包含任何 _SECRET_ENV_MARKERS 中的敏感关键字
       （如 TOKEN、SECRET、PASSWORD 等），即使在白名单中也会被排除
    3. 通过上述两层检查的值原样保留（变量名保持原始大小写）

    安全考量：
    - 白名单策略确保不会意外将包含密钥的环境变量传给子进程
    - 敏感关键字过滤作为第二道防线，防止通过命名规范绕过白名单
      （例如某个名为 "BUILD_TOKEN" 的变量因包含 "TOKEN" 而被排除）
    - 使用 set 和 tuple 作为不可变参数类型，防止运行时篡改

    参数:
        extra_allowed: 除了默认白名单之外，额外允许传递的变量名元组
                       （名称将自动转为大写进行匹配）

    返回:
        过滤后的环境变量字典，仅包含安全且非敏感的变量
    """
    # 合并默认白名单和额外允许的变量（统一转大写以支持不区分大小写匹配）
    names = _SAFE_ENV_NAMES | {name.upper() for name in extra_allowed}

    result: dict[str, str] = {}
    for name, value in os.environ.items():
        upper = name.upper()

        # 第一道过滤：白名单检查
        if upper not in names:
            continue

        # 第二道过滤：敏感关键字检查（在白名单内也可能包含敏感变量）
        if any(marker in upper for marker in _SECRET_ENV_MARKERS):
            continue

        # 通过所有检查，保留此变量
        result[name] = value

    return result


def truncate_output(text: str, limit: int) -> TruncatedOutput:
    """
    对过长的输出文本进行截断处理。

    截断策略（head-tail 截断）：
    - 如果原始文本长度 <= limit，不截断，直接返回原文本
    - 如果原始文本长度 > limit：
      1. 保留前 head_size = max(1, limit // 2) 个字符
      2. 保留后 tail_size = max(1, limit - head_size) 个字符
      3. 中间插入截断标记 "...<truncated N chars>..." 并独占一行
    - 截断标记的长度不计入 limit 限制，因此返回的总长度会略大于 limit

    这种 "保留头尾、标记中间" 的策略确保：
    - 用户能看到输出的开头（通常包含最重要的信息）
    - 用户能看到输出的结尾（通常包含总结或错误信息）
    - 中间被省略部分的大小被明确告知

    注意：此截断是字符级别（len()）而非字节级别的，
    因此对于包含多字节字符（如中文）的文本也适用。

    参数:
        text: 原始输出文本
        limit: 允许的最大字符数（不含截断标记）

    返回:
        TruncatedOutput 对象，包含截断后的文本和元数据
    """
    original = len(text)
    if original <= limit:
        return TruncatedOutput(text, False, original, original)

    # 计算头部和尾部保留的字符数
    head_size = max(1, limit // 2)
    tail_size = max(1, limit - head_size)
    omitted = max(0, original - head_size - tail_size)

    # 构造截断提示标记并拼接最终文本
    marker = f"\n...<truncated {omitted} chars>...\n"
    rendered = text[:head_size] + marker + text[-tail_size:]

    return TruncatedOutput(rendered, True, original, len(rendered))


def command_mentions_internal_state(command: str) -> bool:
    """
    检测 Shell 命令是否试图修改 .codepilot 内部状态文件。

    检测条件（两个条件必须同时满足）：
    1. 命令文本中包含 ".codepilot/" 路径引用
       （先将反斜杠统一替换为斜杠、转小写后检查）
    2. 命令中包含写入或修改操作的关键字：
       - 重定向操作符：>> 或 >（追加/覆盖写入）
       - 文件操作命令：copy, move, mv（复制/移动）
       - 原地编辑命令：sed -i（sed 的原地修改模式）

    使用场景：
    - 在允许 Shell 命令执行前调用此函数，如有匹配则拒绝执行
    - .codepilot 内部文件必须通过 Session Store API 操作以保证数据一致性

    参数:
        command: 用户提交的原始 Shell 命令字符串

    返回:
        True: 命令可能修改 .codepilot 内部状态，应被阻止
        False: 命令不涉及 .codepilot 内部状态的修改
    """
    # 统一路径分隔符为斜杠，转小写以便不区分大小写匹配
    text = command.replace("\\", "/").lower()
    # 两个条件：提到了 .codepilot/ 目录 + 包含写入/修改操作
    return ".codepilot/" in text and bool(re.search(r"(?:>>?|copy|move|mv|sed\s+-i)", text))


# =========================================================================
# 内部辅助函数
# =========================================================================


def _sha256_file(path: Path) -> str:
    """
    计算文件的 SHA256 哈希值（十六进制字符串）。

    使用流式读取（每次读取 1MB 大小的块），内存占用恒定，
    不会因为大文件导致内存溢出。适用于任意大小的文件。

    参数:
        path: 目标文件的 Path 对象

    返回:
        文件内容的 SHA256 十六进制摘要字符串（64 个字符）
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        # iter(lambda: f.read(chunk_size), b"") 是 Python 中流式读取文件的标准惯用法
        # 当 handle.read(1MB) 返回空字节串 b"" 时，迭代终止
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches_command_prefix(command: str, prefix: str) -> bool:
    """
    检查命令是否以指定前缀开头（精确前缀匹配）。

    匹配规则：
    1. 将 prefix 规范化为去除首尾空格 → 小写 → 压缩空格的形式
    2. 如果 command 恰好等于 prefix，返回 True（完全匹配）
    3. 如果 command 以 "prefix + 空格" 开头，返回 True（前缀后跟参数）
    4. 否则返回 False

    这种匹配方式确保：
    - "git status" 匹配 "git status" 和 "git status --short"，但不匹配 "git statuscheck"
    - 对于带有子命令的命令（如 "npm run test"），只有当完整前缀匹配时才生效

    参数:
        command: 规范化后的命令字符串
        prefix: 白名单中的命令前缀

    返回:
        True: 匹配成功
    """
    # 规范化前缀（与 classify_shell_command 中的规范化方式一致）
    prefix = " ".join(prefix.strip().lower().split())
    # 精确匹配或前缀后跟空格（即后跟参数）
    return command == prefix or command.startswith(prefix + " ")


def _has_shell_redirection(command: str) -> bool:
    """
    检测命令是否包含 Shell 重定向或管道操作。

    检测以下三种 Shell 操作符：
    - >> 或 > : 输出重定向（追加或覆盖写入文件）
    - < : 输入重定向（从文件读取输入）
    - | : 管道（将一个命令的输出传给另一个命令）

    如果命令包含这些操作符，说明其行为无法仅通过第一个命令的白名单来
    准确判定（重定向到的目标文件、管道后的命令都可能引入未知行为），
    因此 classify_shell_command 会将其标记为 "unknown"。

    参数:
        command: 规范化后的命令字符串

    返回:
        True: 命令包含重定向或管道操作
    """
    return bool(re.search(r"(?:>>?|<|\|)", command))


def _first_command(command: str) -> str:
    """
    从命令链中提取第一条有效的非环境设置命令。

    提取策略（三层处理）：

    第 1 层：按命令分隔符拆分
    - 使用正则 r"(?:&&|\|\||;|\r?\n)" 将命令链拆分为独立段
      （&& = 逻辑与，|| = 逻辑或，; = 顺序执行，换行 = 自然分隔）
    - 跳过空段

    第 2 层：过滤安全的前置设置命令
    - 如果只有一个命令段，直接返回该段
    - 如果有多个命令段，剥离前置的环境变量设置（set PYTHONPATH=...）
      和目录切换命令（cd，chdir，pushd），因为它们不影响后续命令的安全性

    第 3 层：复合命令的安全性综合判定
    - 对所有剩余的命令段逐一递归进行分类
    - 如果所有段都属于 {verification, read_only, mutation} 中的某一类：
      * 如果有 mutation 类命令，返回第一个 mutation 命令（最严格的）
      * 否则如果有 verification 类命令，返回第一个 verification 命令
      * 否则返回第一个命令
    - 如果有任何段不属于这三类（即 unknown 或 high_risk），
      返回特殊标记 "<compound>"，导致 classify_shell_command 返回 "unknown"

    参数:
        command: 规范化后的命令字符串

    返回:
        提取出的第一个有效命令字符串，或 "<compound>" 表示无法判定
    """
    # 第 1 层：按 Shell 命令分隔符拆分
    segments = [
        segment.strip()
        for segment in re.split(r"(?:&&|\|\||;|\r?\n)", command)
        if segment.strip()
    ]
    if not segments:
        return ""

    # 只有一个命令段，直接返回
    if len(segments) == 1:
        return segments[0]

    # 第 2 层：过滤掉安全的环境设置和目录切换前缀命令
    command_segments = [
        segment
        for segment in segments
        if not _is_safe_env_setup(segment) and not _is_safe_directory_setup(segment)
    ]

    # 如果过滤后没有剩余命令（全是环境设置），无法确定安全类别
    if not command_segments:
        return "unknown"

    # 第 3 层：对每个剩余命令递归分类，综合判定安全性
    # 注意：此处递归调用 classify_shell_command，但每个段已经被确认不含命令分隔符，
    # 所以不会无限递归
    classes = [classify_shell_command(segment) for segment in command_segments]

    # 检查是否所有段的分类都在安全范围内
    if all(item in {"verification", "read_only", "mutation"} for item in classes):
        # 取最严格的分类作为代表命令
        if "mutation" in classes:
            return command_segments[classes.index("mutation")]
        if "verification" in classes:
            return command_segments[classes.index("verification")]
        return command_segments[0]

    # 存在不在安全类别中的段，返回特殊标记
    return "<compound>"


def _is_safe_env_setup(command: str) -> bool:
    """
    检查命令是否仅为安全的环境变量设置操作。

    支持三种 Shell 环境变量设置语法：
    - Windows CMD: set PYTHONPATH=...
    - PowerShell:  $env:PYTHONPATH = ...
    - Unix Shell:  export PYTHONPATH=...

    这些操作被认为是安全的，因为它们仅修改 PYTHONPATH，
    不会产生文件系统副作用，也不会泄露敏感信息。

    参数:
        command: 规范化后的命令字符串

    返回:
        True: 命令是安全的环境变量设置
    """
    normalized = " ".join(command.strip().lower().split())
    return bool(
        re.fullmatch(r"set\s+pythonpath=.*", normalized)
        or re.fullmatch(r"\$env:pythonpath\s*=.*", normalized)
        or re.fullmatch(r"export\s+pythonpath=.*", normalized)
    )


def _is_safe_directory_setup(command: str) -> bool:
    """
    检查命令是否仅为安全的目录切换操作。

    使用两步验证：
    1. 用 _directory_setup_target() 从命令中提取目标目录路径
    2. 用 _is_safe_relative_shell_path() 验证提取出的路径是否安全
       （相对路径、不包含特殊字符、不包含 ".." 回溯）

    安全的目录切换不会导致路径逃逸到工作区之外。

    参数:
        command: 规范化后的命令字符串

    返回:
        True: 命令是安全的目录切换操作
    """
    target = _directory_setup_target(command)
    return target is not None and _is_safe_relative_shell_path(target)


def _directory_setup_target(command: str) -> str | None:
    """
    从目录切换命令中提取目标路径。

    支持的命令：
    - cd <path>
    - chdir <path>
    - pushd <path>
    - cd /d <path>（Windows CMD 切换驱动器并切换目录）
    - 相应的 PowerShell 变体

    提取步骤：
    1. 用正则匹配命令前缀并提取路径参数
    2. 用 _strip_shell_quotes() 去除路径周围的 Shell 引号（单引号或双引号）

    参数:
        command: 规范化后的命令字符串

    返回:
        提取出的目标路径字符串，如果不是目录切换命令则返回 None
    """
    normalized = " ".join(command.strip().lower().split())
    # 匹配 cd / chdir / pushd，可选 /d 参数，后跟路径
    match = re.fullmatch(r"(?:cd|chdir|pushd)\s+(?:/d\s+)?(.+)", normalized)
    if match is None:
        return None
    # 去除 Shell 引号（如 cd "my dir" → my dir）
    return _strip_shell_quotes(match.group(1))


def _is_python_workspace_script(command: str) -> bool:
    """
    检查命令是否是对工作区内 Python 脚本的调用。

    匹配模式：python/python3/py 后跟一个 .py 文件路径，可选参数。

    安全验证使用两步法：
    1. 用正则提取 .py 脚本的路径
    2. 用 _is_safe_relative_shell_path() 验证：
       - 是相对路径（非绝对路径，非驱动器路径，非 ~ 路径）
       - 不包含路径回溯 ".."
       - 不包含 Shell 特殊字符（$、%、管道等）

    这种命令被视为 "mutation" 类，因为 Python 脚本可能会修改文件。
    实际上执行的脚本路径是经过安全验证的相对路径。

    参数:
        command: 规范化后的命令字符串

    返回:
        True: 命令是对工作区内 Python 脚本的安全调用
    """
    normalized = " ".join(command.strip().lower().split())
    # 匹配 python/python3/py 后跟 .py 文件路径
    match = re.fullmatch(r"(?:python|python3|py)\s+([^\s]+\.py)(?:\s+.*)?", normalized)
    if match is None:
        return False
    # 去除 Shell 引号后验证路径安全性
    return _is_safe_relative_shell_path(_strip_shell_quotes(match.group(1)))


def _strip_shell_quotes(value: str) -> str:
    """
    去除字符串两端的 Shell 引号（单引号或双引号）。

    规则：
    - 如果字符串以相同的引号字符开头和结尾（' 或 "），则去除这对引号
    - 去除后再次 strip 空白字符以处理引号内可能的空格
    - 如果两端引号不同（不匹配），则不做处理
    - 字符串长度必须 >= 2 才可能被引号包裹

    示例：
        '"hello"'  → "hello"
        "'hello'"  → "hello"
        "noquotes" → "noquotes"

    参数:
        value: 可能带有 Shell 引号的字符串

    返回:
        去除引号后的字符串
    """
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1].strip()
    return text


def _is_safe_relative_shell_path(value: str) -> bool:
    """
    验证字符串是否为安全的相对 Shell 路径。

    安全检查（任一项不通过则返回 False）：
    1. 空字符串：不安全
    2. 以 "/" 开头：绝对 Unix 路径，不安全
    3. 以 "~" 开头：用户主目录路径，不安全
    4. 匹配驱动器号模式（如 "C:"）：Windows 绝对路径，不安全
    5. 包含 Shell 特殊字符（$、%、`、|、&、;、<、>）：不安全，
       这些字符可能被 Shell 解释执行，导致命令注入
    6. 路径部分中包含 ".."：路径回溯，不安全
    7. 路径必须至少有一个有效部分

    此函数用于验证从命令参数中提取的文件路径，
    确保它们不会逃逸到工作区之外。

    参数:
        value: 待验证的路径字符串

    返回:
        True: 路径是安全的相对路径
    """
    # 统一路径分隔符为斜杠
    text = value.strip().replace("\\", "/")

    # 空路径 或 绝对路径（Unix / 开头、~ 开头、Windows 驱动器号开头）
    if not text or text.startswith(("/", "~")) or re.match(r"^[a-z]:", text):
        return False

    # Shell 特殊字符检查：防止命令注入
    if any(marker in text for marker in ("$", "%", "`", "|", "&", ";", "<", ">")):
        return False

    # 路径回溯检查：防止 .. 逃逸出工作区
    parts = [part for part in text.split("/") if part]
    return bool(parts) and ".." not in parts


# ---------------------------------------------------------------------------
# 模块导出列表
# ---------------------------------------------------------------------------
__all__ = [
    "ShellCommandClass",
    "ShellExecutionPolicy",
    "TruncatedOutput",
    "WorkspaceSandbox",
    "build_shell_environment",
    "classify_shell_command",
    "command_mentions_internal_state",
    "file_state_for_path",
    "truncate_output",
]
