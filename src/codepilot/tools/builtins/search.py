"""规范的工作区搜索工具 —— grep / find（glob 匹配）。

本文件实现两个搜索工具：
1. grep: 在文件中搜索正则表达式匹配
2. find: 按 glob 模式匹配文件路径

两个工具都内置了安全限制：
- 跳过 .git / .codepilot / .venv / node_modules / __pycache__ 等目录
- 最多扫描 20,000 个文件
- 最多扫描 64MB 数据
- 路径经过 WorkspaceSandbox 验证
- 只读操作（plan 和 execute 模式均可使用）
"""

import asyncio
import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..codecs import DataclassCodec
from ..contracts import ToolExecutionContext, ToolHandlerError, ToolRegistration, ToolSpec
from ..results import TextContent
from ..sandbox import WorkspaceSandbox
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

# 被排除的目录（跳过这些不扫描）
_EXCLUDED_DIRS = frozenset({".git", ".codepilot", ".venv", "node_modules", "__pycache__"})
# 扫描限制：最多扫描文件数和总字节数，防止意外扫描大项目导致资源耗尽
_MAX_SCANNED_FILES = 20_000
_MAX_SCANNED_BYTES = 64 * 1024 * 1024


# ── 输入类型 ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GrepInput:
    """grep 工具的输入参数。

    参数:
        pattern: 要搜索的正则表达式
        path: 搜索根路径（默认 "."）
        glob: 文件匹配模式（默认 "**/*" 所有文件）
        max_results: 最多返回的匹配行数（默认 200）
        case_sensitive: 是否区分大小写（默认 True）
    """
    pattern: str
    path: str = "."
    glob: str = "**/*"
    max_results: int = 200
    case_sensitive: bool = True


@dataclass(frozen=True)
class FindInput:
    """find 工具的输入参数。

    参数:
        pattern: 要匹配的 glob 模式
        path: 搜索根路径（默认 "."）
        max_results: 最多返回的文件路径数（默认 500）
    """
    pattern: str
    path: str = "."
    max_results: int = 500


@dataclass(frozen=True)
class SearchOutput:
    """搜索工具的统一输出类型。

    参数:
        text: 给 LLM 看的搜索结果文本
        details: 结构化的详细数据（匹配数、扫描文件数）
        metadata: 元数据（是否截断、是否扫描受限）
    """
    text: str
    details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


_OUTPUT_SCHEMA = {
    "$schema": _DRAFT,
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "details": {"type": "object"},
        "metadata": {"type": "object"},
    },
    "required": ["text", "details", "metadata"],
    "additionalProperties": False,
}


# ── 注册创建函数 ──────────────────────────────────────────────────────────────


def create_search_registrations(
    sandbox: WorkspaceSandbox,
    *,
    allow: Callable[[str], bool],
) -> list[ToolRegistration]:
    """创建搜索工具注册（grep 和 find）。

    参数:
        sandbox: 工作区沙箱
        allow: 工具启用过滤器

    返回:
        ToolRegistration 列表
    """
    configs = (
        (
            "grep",
            GrepInput,
            _grep_schema(),
            "Search UTF-8 workspace file contents with a regular expression and return bounded file:line matches. Use path and glob to narrow known code areas; use find for path-name discovery. Repository metadata, environments, dependency directories, and oversized scans are skipped. Check truncation and scan-limit metadata before concluding a symbol is absent.",
        ),
        (
            "find",
            FindInput,
            _find_schema(),
            "Find workspace files whose relative paths match one glob pattern. Use this for file-name or extension discovery, then read or grep the relevant results; do not use it for content search. Metadata, environments, dependency directories, and oversized scans are skipped, and results are bounded.",
        ),
    )
    result: list[ToolRegistration] = []
    for name, input_type, schema, description in configs:
        if not allow(name):
            continue
        result.append(
            ToolRegistration(
                version="1.0.0",
                implementation_version="2",
                spec=ToolSpec(name, description, schema, _OUTPUT_SCHEMA),
                category="search",
                source="builtin",
                owner="codepilot.builtin",
                policy=_policy(),
                input_codec=DataclassCodec(input_type, schema),
                output_codec=DataclassCodec(SearchOutput, _OUTPUT_SCHEMA),
                handler=_SearchHandler(sandbox, name),
                renderer=_Renderer(),
                access_resolver=_Resolver(sandbox, name),
            )
        )
    return result


# ── 访问解析器 ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Resolver:
    """搜索工具访问解析器 —— 将搜索路径解析为权限请求。"""
    sandbox: WorkspaceSandbox
    name: str

    def resolve(self, input, request):
        """把搜索输入解析为工作区内只读资源访问请求。"""
        _ = request
        root = self.sandbox.ensure_readable_path(self.sandbox.resolve_path(input.path))
        return ToolAccessResolution(
            input=input,
            access=ToolAccessRequest(
                actions=(self.name,),
                resources=(ToolResource("workspace:///" + self.sandbox.relative_path(root)),),
                effects=frozenset({"filesystem_read"}),
                risk="low",
                reason=f"Search workspace path {input.path}",
                safe_preview={"path": input.path, "pattern": input.pattern},
            ),
        )


# ── 处理器 ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _SearchHandler:
    """搜索工具处理器 —— 执行 grep 或 find 操作。

    使用 asyncio.to_thread 将 CPU 密集的文件扫描放到线程池执行，
    避免阻塞事件循环。

    参数:
        sandbox: 工作区沙箱
        name: 工具名称（"grep" 或 "find"）
    """
    sandbox: WorkspaceSandbox
    name: str

    async def __call__(self, input, context: ToolExecutionContext) -> SearchOutput:
        """入口方法 —— 在开始前检查取消信号，然后分发到具体逻辑。

        参数:
            input: GrepInput 或 FindInput
            context: 执行上下文

        返回:
            SearchOutput 搜索结果
        """
        context.cancellation.raise_if_cancelled()
        root = self.sandbox.ensure_readable_path(self.sandbox.resolve_path(input.path))
        if not root.exists():
            raise ToolHandlerError(f"{self.name}.path_missing", f"Search path not found: {input.path}")
        operation = self._grep if self.name == "grep" else self._find
        # 在后台线程执行文件扫描（避免阻塞事件循环）
        output = await asyncio.to_thread(operation, root, input, context.cancellation)
        context.effects.report(
            ToolEffect(
                kind="filesystem_read",
                resource=ToolResource("workspace:///" + self.sandbox.relative_path(root)),
                operation=self.name,
                status="completed",
                certainty="observed",
            )
        )
        return output

    def _grep(self, root: Path, input: GrepInput, cancellation) -> SearchOutput:
        """在文件内容中搜索正则表达式匹配。

        处理流程:
        1. 编译正则表达式（区分大小写/不区分大小写）
        2. 遍历所有文件（跳过排除目录）
        3. 对匹配 glob 模式的文件，读取内容搜索
        4. 扫描限制：最多 20,000 文件 + 64MB 数据
        5. 结果限制：最多 max_results 个匹配行

        结果格式: "相对路径:行号:匹配行内容"

        参数:
            root: 搜索根目录
            input: GrepInput
            cancellation: 取消令牌（定期检查）

        返回:
            SearchOutput 搜索结果
        """
        try:
            regex = re.compile(input.pattern, 0 if input.case_sensitive else re.IGNORECASE)
        except re.error as exc:
            raise ToolHandlerError("grep.invalid_pattern", str(exc)) from exc
        matches: list[str] = []
        scanned = 0
        scanned_bytes = 0
        scan_limited = False
        for path in _files(root, cancellation):
            cancellation.raise_if_cancelled()
            relative = self.sandbox.relative_path(path)
            if not _glob_matches(relative, input.glob):
                continue
            try:
                safe = self.sandbox.ensure_readable_path(path)
                size = safe.stat().st_size
                if scanned >= _MAX_SCANNED_FILES or scanned_bytes + size > _MAX_SCANNED_BYTES:
                    scan_limited = True
                    break
                lines = safe.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError, ValueError):
                continue
            scanned += 1
            scanned_bytes += size
            for number, line in enumerate(lines, start=1):
                if number % 256 == 0:
                    cancellation.raise_if_cancelled()
                if regex.search(line):
                    matches.append(f"{relative}:{number}:{line}")
                    if len(matches) >= input.max_results:
                        return SearchOutput(
                            text="\n".join(matches),
                            details={"match_count": len(matches), "scanned_files": scanned},
                            metadata={"truncated": True, "scan_limited": False},
                        )
        return SearchOutput(
            text="\n".join(matches),
            details={"match_count": len(matches), "scanned_files": scanned},
            metadata={"truncated": scan_limited, "scan_limited": scan_limited},
        )

    def _find(self, root: Path, input: FindInput, cancellation) -> SearchOutput:
        """按 glob 模式匹配文件路径。

        处理流程:
        1. 遍历所有文件（跳过排除目录）
        2. 对每个文件，用 glob 模式匹配其相对路径
        3. 扫描限制：最多 20,000 文件
        4. 结果限制：最多 max_results 个匹配

        参数:
            root: 搜索根目录
            input: FindInput
            cancellation: 取消令牌

        返回:
            SearchOutput 搜索结果
        """
        matches: list[str] = []
        scanned = 0
        scan_limited = False
        for path in _files(root, cancellation):
            cancellation.raise_if_cancelled()
            scanned += 1
            if scanned > _MAX_SCANNED_FILES:
                scan_limited = True
                break
            try:
                self.sandbox.ensure_readable_path(path)
            except ValueError:
                continue
            relative = self.sandbox.relative_path(path)
            scoped = path.relative_to(root).as_posix() if root.is_dir() else path.name
            if _glob_matches(scoped, input.pattern):
                matches.append(relative)
                if len(matches) >= input.max_results:
                    return SearchOutput(
                        text="\n".join(matches),
                        details={"match_count": len(matches)},
                        metadata={"truncated": True, "scan_limited": False},
                    )
        return SearchOutput(
            text="\n".join(matches),
            details={"match_count": len(matches)},
            metadata={"truncated": scan_limited, "scan_limited": scan_limited},
        )


# ── 渲染器 ────────────────────────────────────────────────────────────────────


class _Renderer:
    """将搜索结果文本转换为模型可消费的文本内容块。"""

    def render(self, data):
        """渲染搜索工具的文本结果。"""
        return (TextContent(text=str(data["text"])),)


# ── 辅助函数 ──────────────────────────────────────────────────────────────────


def _files(root: Path, cancellation):
    """遍历目录树，逐个 yield 文件路径。

    跳过 _EXCLUDED_DIRS 中的目录。
    在每个检查点调用 cancellation.raise_if_cancelled() 响应取消。

    参数:
        root: 遍历根目录
        cancellation: 取消令牌

    生成:
        Path 文件路径
    """
    if root.is_file():
        yield root
        return
    for current, dirs, files in os.walk(root):
        cancellation.raise_if_cancelled()
        # 原地修改 dirs 列表来过滤遍历的目录（os.walk 特性）
        dirs[:] = sorted(name for name in dirs if name not in _EXCLUDED_DIRS)
        for name in sorted(files):
            cancellation.raise_if_cancelled()
            yield Path(current) / name


def _glob_matches(path: str, pattern: str) -> bool:
    """检查路径是否匹配 glob 模式。

    支持 **/ 前缀的递归匹配，尝试两种模式匹配：
    1. 原始模式
    2. 如果模式以 **/ 开头，尝试去掉前缀后的模式

    参数:
        path: 文件路径字符串
        pattern: glob 模式

    返回:
        True 表示匹配
    """
    patterns = [pattern]
    if pattern.startswith("**/"):
        patterns.append(pattern[3:])
    return any(fnmatch.fnmatch(path, item) or Path(path).match(item) for item in patterns)


# ── 策略 ──────────────────────────────────────────────────────────────────────


def _policy() -> ToolPolicy:
    """搜索工具的安全策略 —— 只读、低风险、可并行。"""
    return ToolPolicy(
        allowed_modes=frozenset({"plan", "execute"}),
        declared_effects=frozenset({"filesystem_read"}),
        required_permissions=frozenset({"workspace.read"}),
        base_risk="low",
        approval="never",
        timeout=TimeoutPolicy(15_000, 30_000),
        concurrency=ConcurrencyPolicy(mode="parallel"),
        output_limits=OutputLimits(),
        output_trust=OutputTrustPolicy(),
    )


# ── JSON Schema 定义 ──────────────────────────────────────────────────────────


def _grep_schema() -> dict[str, Any]:
    return {"$schema": _DRAFT, "type": "object", "properties": {"pattern": {"type": "string", "minLength": 1}, "path": {"type": "string", "default": "."}, "glob": {"type": "string", "default": "**/*"}, "max_results": {"type": "integer", "minimum": 1, "maximum": 10_000, "default": 200}, "case_sensitive": {"type": "boolean", "default": True}}, "required": ["pattern"], "additionalProperties": False}


def _find_schema() -> dict[str, Any]:
    return {"$schema": _DRAFT, "type": "object", "properties": {"pattern": {"type": "string", "minLength": 1}, "path": {"type": "string", "default": "."}, "max_results": {"type": "integer", "minimum": 1, "maximum": 10_000, "default": 500}}, "required": ["pattern"], "additionalProperties": False}


__all__ = ["create_search_registrations"]
