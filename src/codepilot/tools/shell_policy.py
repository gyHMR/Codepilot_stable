from __future__ import annotations

"""保守的 shell 命令分类、环境变量过滤和输出截断控制。"""

import os
import re
from dataclasses import dataclass
from typing import Literal


# Shell 命令分类：验证型/修改型/高风险/未知
ShellCommandClass = Literal["verification", "mutation", "high_risk", "unknown"]

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
    r"\bcurl\b.*(?:--upload-file|-t)\b",
    r"\bwget\b.*--post-file\b",
)
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
_MUTATION_PREFIXES = (
    "ruff format",
    "black ",
    "prettier ",
    "npm run format",
    "git add",
    "python -m build",
    "npm run generate",
)
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
_SECRET_ENV_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "CREDENTIAL", "COOKIE")


@dataclass(frozen=True)
class ShellExecutionPolicy:
    """Shell 执行策略：控制超时、输出限制和环境变量白名单。"""
    timeout_seconds: int = 30           # 默认超时（秒）
    max_timeout_seconds: int = 120      # 最大允许超时（秒）
    stdout_limit: int = 20_000          # stdout 最大字符数
    stderr_limit: int = 10_000          # stderr 最大字符数
    allowed_env: tuple[str, ...] = ()   # 额外允许的环境变量名

    def validate_timeout(self, requested: object) -> tuple[int | None, str | None]:
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
    """截断后的输出结果。"""
    text: str              # 截断后的文本
    truncated: bool        # 是否发生了截断
    original_chars: int    # 原始字符数
    returned_chars: int    # 返回的字符数


def classify_shell_command(command: str) -> ShellCommandClass:
    """分类 shell 命令：验证型（测试/lint）、修改型（格式化/git add）、高风险（rm -rf）或未知。"""
    normalized = " ".join(command.strip().lower().split())
    if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in _HIGH_RISK_PATTERNS):
        return "high_risk"
    first = _first_command(normalized)
    if any(_matches_command_prefix(first, prefix) for prefix in _VERIFICATION_PREFIXES):
        return "verification"
    if any(_matches_command_prefix(first, prefix) for prefix in _MUTATION_PREFIXES):
        return "mutation"
    return "unknown"


def _matches_command_prefix(command: str, prefix: str) -> bool:
    prefix = " ".join(prefix.strip().lower().split())
    return command == prefix or command.startswith(prefix + " ")


def build_shell_environment(extra_allowed: tuple[str, ...] = ()) -> dict[str, str]:
    """构建安全的 shell 环境变量（过滤密钥，只保留白名单变量）。"""
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
    """截断输出文本：保留头部和尾部，中间用省略标记替代。"""
    original = len(text)
    if original <= limit:
        return TruncatedOutput(text, False, original, original)
    head_size = max(1, limit // 2)
    tail_size = max(1, limit - head_size)
    omitted = max(0, original - head_size - tail_size)
    marker = f"\n...<truncated {omitted} chars>...\n"
    rendered = text[:head_size] + marker + text[-tail_size:]
    return TruncatedOutput(rendered, True, original, len(rendered))


def _first_command(command: str) -> str:
    # Classification is intentionally conservative: if the command chains
    # several operations, only classify it safe when every segment is safe.
    segments = [
        segment.strip()
        for segment in re.split(r"(?:&&|\|\||;|\r?\n)", command)
        if segment.strip()
    ]
    if not segments:
        return ""
    if len(segments) > 1:
        classes = {classify_shell_command(segment) for segment in segments}
        if classes == {"verification"}:
            return segments[0]
        return "<compound>"
    return segments[0]


__all__ = [
    "ShellCommandClass",
    "ShellExecutionPolicy",
    "TruncatedOutput",
    "build_shell_environment",
    "classify_shell_command",
    "truncate_output",
]
