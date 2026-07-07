from __future__ import annotations

# 新手导读：WorkspaceSandbox 负责路径解析和工作区边界检查，是文件工具的安全底座。
# 关注点：所有文件读写都应通过它确认路径没有逃逸 workspace。

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class WorkspaceSandbox:
    """工作区沙箱：在解析路径的同时强制执行工作区边界检查。"""
    workspace_dir: str | Path

    @property
    def root(self) -> Path:
        return Path(self.workspace_dir).resolve()

    def resolve_path(self, path_text: str | Path) -> Path:
        path = Path(path_text)
        target = path.resolve() if path.is_absolute() else (self.root / path).resolve()
        self.ensure_within_workspace(target)
        return target

    def ensure_within_workspace(self, path: str | Path) -> Path:
        target = Path(path).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("Path escapes workspace boundary") from exc
        return target


def file_state_for_path(workspace_dir: str | Path, path: str | Path) -> dict[str, Any]:
    """获取文件状态快照（路径、存在性、大小、修改时间、SHA256）。"""
    sandbox = WorkspaceSandbox(workspace_dir)
    target = sandbox.resolve_path(path)
    relative = target.relative_to(sandbox.root).as_posix()
    if not target.exists() or not target.is_file():
        return {
            "path": relative,
            "exists": False,
            "workspace_path": str(sandbox.root),
        }
    stat = target.stat()
    return {
        "path": relative,
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(target),
        "workspace_path": str(sandbox.root),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ── Shell execution safety ─────────────────────────────────────────

# Shell 命令分类：验证型/只读型/修改型/高风险/未知
ShellCommandClass = Literal["verification", "read_only", "mutation", "high_risk", "unknown"]

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
    "ver",
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
    if _has_shell_redirection(first):
        return "unknown"
    if any(_matches_command_prefix(first, prefix) for prefix in _VERIFICATION_PREFIXES):
        return "verification"
    if any(_matches_command_prefix(first, prefix) for prefix in _READ_ONLY_PREFIXES):
        return "read_only"
    if any(_matches_command_prefix(first, prefix) for prefix in _MUTATION_PREFIXES):
        return "mutation"
    if _is_python_workspace_script(first):
        return "mutation"
    return "unknown"


def _matches_command_prefix(command: str, prefix: str) -> bool:
    prefix = " ".join(prefix.strip().lower().split())
    return command == prefix or command.startswith(prefix + " ")


def _has_shell_redirection(command: str) -> bool:
    return bool(re.search(r"(?:>>?|<|\|)", command))


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
        command_segments = [
            segment
            for segment in segments
            if not _is_safe_env_setup(segment) and not _is_safe_directory_setup(segment)
        ]
        if command_segments:
            classes = [classify_shell_command(segment) for segment in command_segments]
            if all(item in {"verification", "read_only", "mutation"} for item in classes):
                if "mutation" in classes:
                    return next(
                        segment
                        for segment, item in zip(command_segments, classes)
                        if item == "mutation"
                    )
                if "verification" in classes:
                    return next(
                        segment
                        for segment, item in zip(command_segments, classes)
                        if item == "verification"
                    )
                return command_segments[0]
        return "<compound>"
    return segments[0]


def _is_safe_env_setup(command: str) -> bool:
    normalized = " ".join(command.strip().lower().split())
    return bool(
        re.fullmatch(r"set\s+pythonpath=.*", normalized)
        or re.fullmatch(r"\$env:pythonpath\s*=.*", normalized)
        or re.fullmatch(r"export\s+pythonpath=.*", normalized)
    )


def _is_safe_directory_setup(command: str) -> bool:
    target = _directory_setup_target(command)
    return target is not None and _is_safe_relative_shell_path(target)


def _directory_setup_target(command: str) -> str | None:
    normalized = " ".join(command.strip().lower().split())
    match = re.fullmatch(r"(?:cd|chdir|pushd)\s+(?:/d\s+)?(.+)", normalized)
    if match is None:
        return None
    return _strip_shell_quotes(match.group(1))


def _is_python_workspace_script(command: str) -> bool:
    normalized = " ".join(command.strip().lower().split())
    match = re.fullmatch(r"(?:python|python3|py)\s+([^\s]+\.py)(?:\s+.*)?", normalized)
    if match is None:
        return False
    return _is_safe_relative_shell_path(_strip_shell_quotes(match.group(1)))


def _strip_shell_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1].strip()
    return text


def _is_safe_relative_shell_path(value: str) -> bool:
    text = value.strip().replace("\\", "/")
    if not text or text.startswith(("/", "~")):
        return False
    if re.match(r"^[a-z]:", text):
        return False
    if any(marker in text for marker in ("$", "%", "`", "|", "&", ";", "<", ">")):
        return False
    parts = [part for part in text.split("/") if part]
    return bool(parts) and ".." not in parts


__all__ = [
    "ShellCommandClass",
    "ShellExecutionPolicy",
    "TruncatedOutput",
    "WorkspaceSandbox",
    "build_shell_environment",
    "classify_shell_command",
    "file_state_for_path",
    "truncate_output",
]
