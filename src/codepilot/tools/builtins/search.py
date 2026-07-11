from __future__ import annotations

"""
内置搜索工具：grep 和 find。

本模块为 Codepilot 提供两个核心的代码库搜索工具：
  - grep_tool:  基于正则表达式的文件内容搜索工具，支持在指定目录下按 glob 模式筛选文件并进行正则匹配。
  - find_tool:  基于 glob 模式的文件/目录路径搜索工具，用于按文件名模式快速定位文件或目录。

这两个工具通过 create_search_tools() 工厂函数统一创建，返回 ToolDefinition 列表，
由工具注册中心注册后供 AI 代理调用。

设计要点：
  - 所有路径操作受 WorkspaceSandbox 沙箱约束，防止路径越界访问。
  - 对二进制文件和超大文件做了自动跳过（参见 _MAX_FILE_BYTES 和 _is_binary）。
  - 对常见无关目录（如 .git、node_modules）做了自动忽略（参见 _IGNORED_DIRS）。
  - 搜索结果有上限保护（max_matches / max_results），防止输出爆炸影响上下文窗口。
"""

import re
from typing import Any, Callable

from codepilot.protocols import TextContent
from codepilot.tools.contracts import ToolCallRequest, ToolDefinition, ToolResult
from codepilot.tools.registry import get_builtin_tool_metadata
from codepilot.tools.sandbox import WorkspaceSandbox

# =============================================================================
# 目录忽略规则
# =============================================================================

# _IGNORED_DIRS: 在搜索文件时自动排除的目录名集合。
# 这些目录通常包含版本控制数据、构建产物、依赖缓存或 IDE 配置，
# 搜索它们既无意义又会严重拖慢扫描速度，因此一律跳过。
_IGNORED_DIRS = {
    ".git",           # Git 版本控制元数据目录
    ".codepilot",     # Codepilot 自身的工作数据目录
    ".pytest_cache",  # pytest 测试缓存目录
    "__pycache__",    # Python 字节码缓存目录
    "node_modules",   # Node.js 第三方依赖目录（文件数量极大）
    ".venv",          # Python 虚拟环境目录
    "venv",           # Python 虚拟环境目录（另一种常见命名）
    "dist",           # 构建/分发产物目录
    "build",          # 构建产物目录
}

# =============================================================================
# 扫描限制常量
# =============================================================================

# _MAX_SCAN_FILES: 单次搜索最多扫描的文件数量上限（5000 个文件）。
# 原因：
#   1. 防止在大型仓库（如 monorepo）中因 glob 匹配到过多文件而导致工具调用超时。
#   2. 限制单次工具调用的 CPU 和 I/O 开销，保证其他并发请求的响应性。
#   3. 避免向 AI 模型返回过多上下文，超出 token 限制或淹没关键信息。
# 超出上限的文件会被静默截断，同时在返回的 metadata.truncated 中标记。
_MAX_SCAN_FILES = 5000

# _MAX_FILE_BYTES: 单文件内容搜索时允许的最大文件大小（2 MB）。
# 原因：
#   1. 超大文件（如压缩包、日志文件、数据库 dump）通常不是代码搜索的目标，
#      读入内存会浪费资源并拖慢整体搜索速度。
#   2. 大于此阈值的文件即使包含匹配行，其上下文价值也有限，
#      且全量加载可能触发内存压力。
#   3. 与 _is_binary() 互补：既有二进制检测（null 字节），也有大小兜底。
# 注意：此限制仅适用于 grep_tool（内容搜索），find_tool 不受影响。
_MAX_FILE_BYTES = 2 * 1024 * 1024


# =============================================================================
# 工具工厂函数
# =============================================================================

def create_search_tools(
    sandbox: WorkspaceSandbox,
    *,
    allow: Callable[[str], bool],
) -> list[ToolDefinition]:
    """
    创建搜索工具的工厂函数。

    根据 allow 回调的权限判断，按需创建 grep_tool 和 find_tool 两个内置搜索工具，
    返回对应的 ToolDefinition 列表供外部注册使用。

    参数:
        sandbox: 工作区沙箱实例，用于解析和校验所有文件路径，防止路径越界。
        allow:   权限检查回调，接收工具名称（"grep" 或 "find"），返回是否允许创建该工具。
                 允许外部根据用户配置或安全策略动态控制工具可用性。

    返回:
        ToolDefinition 列表，包含被允许的搜索工具定义。
    """
    workspace = sandbox.root
    tools: list[ToolDefinition] = []

    # =========================================================================
    # grep_tool: 基于正则表达式的文件内容搜索
    # =========================================================================

    async def grep_tool(request: ToolCallRequest, signal=None, on_update=None) -> ToolResult:
        """
        在文件内容中按正则表达式进行搜索。

        参数（通过 request.arguments 传入）:
            pattern:       必填，正则表达式模式字符串。
            path:          搜索起始路径或具体文件，相对于工作区根目录，默认为 "."（整个工作区）。
            glob:          path 为目录时使用的文件筛选 glob 模式，默认为 "**/*"。
            max_matches:   返回的最大匹配行数，默认 200。达到上限后提前终止扫描。
            case_sensitive: 是否区分大小写，默认 True。

        返回:
            ToolResult，其中 content 包含匹配行（格式: "相对路径:行号:内容前220字符"），
            metadata 中包含匹配数、扫描文件数、跳过二进制文件数及截断标记。

        内部流程:
            1. 参数解析与校验（pattern 必填、路径合法性、正则编译）。
            2. 按 glob 模式收集候选文件列表（自动排除忽略目录和二进制文件）。
            3. 逐文件逐行进行正则匹配，达到 max_matches 上限后提前终止。
            4. 组装 ToolResult，包含结果文本和元数据。
        """
        _ = signal, on_update  # 信号和更新回调由工具运行器注入，当前搜索工具不使用
        params = request.arguments
        pattern = str(params.get("pattern", ""))
        start_path = str(params.get("path", "."))
        glob_pattern = str(params.get("glob", "**/*"))
        max_matches = int(params.get("max_matches", 200))
        case_sensitive = bool(params.get("case_sensitive", True))

        # ----- 参数校验 -----
        if not pattern:
            return _error_result("Missing pattern", "missing_pattern")

        # 路径解析：通过沙箱将用户输入的相对路径解析为工作区内的绝对路径
        root, error = _resolve_path(sandbox, start_path)
        if error is not None:
            return error
        if not root.exists():
            return _error_result(f"Path not found: {start_path}", "path_not_found")

        # 编译正则表达式，捕获并报告不合法的正则语法
        try:
            regex = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
        except re.error as exc:
            return _error_result(f"Invalid regex: {exc}", "invalid_regex")

        # ----- 收集候选文件列表 -----
        # path 可以是具体文件；目录路径才使用 glob 展开。
        if root.is_file():
            files = [root] if not _is_ignored(root, workspace) else []
        else:
            files = sorted(
                (
                    path
                    for path in root.glob(glob_pattern)
                    if path.is_file() and not _is_ignored(path, workspace)
                ),
                key=lambda path: path.as_posix(),
            )

        # ----- 逐文件搜索 -----
        matches: list[str] = []
        scanned = 0          # 实际扫描的文件计数
        skipped_binary = 0   # 因二进制或超大而跳过的文件计数
        for file_path in files[:_MAX_SCAN_FILES]:
            scanned += 1
            try:
                # 跳过超大文件（> _MAX_FILE_BYTES）和二进制文件（含 null 字节）
                if file_path.stat().st_size > _MAX_FILE_BYTES or _is_binary(file_path):
                    skipped_binary += 1
                    continue
                text = file_path.read_text(encoding="utf-8")
            except Exception:
                # 文件读取失败（权限、编码等问题），静默跳过，继续处理下一个文件
                continue
            # 逐行匹配正则，格式化为 "相对路径:行号:内容（截断至220字符）"
            for line_no, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    rel = file_path.relative_to(workspace).as_posix()
                    matches.append(f"{rel}:{line_no}:{line[:220]}")
                    if len(matches) >= max_matches:
                        break
            if len(matches) >= max_matches:
                break

        # 判断结果是否被截断：
        #   1. 匹配行数达到 max_matches 上限
        #   2. 候选文件总数超过 _MAX_SCAN_FILES 上限（可能有未扫描的文件）
        truncated = len(matches) >= max_matches or len(files) > _MAX_SCAN_FILES

        return ToolResult(
            content=[TextContent(text="\n".join(matches) if matches else "(no matches)")],
            details={
                "matches": len(matches),
                "scanned_files": scanned,
                "skipped_binary_files": skipped_binary,
            },
            metadata={
                "search_paths": [root.relative_to(workspace).as_posix()],
                "matches": len(matches),
                "truncated": truncated,
                "output_quality": {"truncated": truncated},
            },
        )

    # =========================================================================
    # find_tool: 基于 glob 模式的文件/目录路径搜索
    # =========================================================================

    async def find_tool(request: ToolCallRequest, signal=None, on_update=None) -> ToolResult:
        """
        按 glob 模式搜索文件或目录路径。

        参数（通过 request.arguments 传入）:
            path:       搜索起始路径，相对于工作区根目录，默认为 "."。
            pattern:    glob 匹配模式，默认为 "**/*"（匹配所有文件和目录）。
            max_results: 返回的最大结果数，默认 200。

        返回:
            ToolResult，其中 content 每行是一个相对路径（目录以 "/" 结尾），
            metadata 中包含结果数量和截断标记。

        注意：
            find_tool 不检查文件大小和二进制属性——它只匹配路径名，不读取文件内容。
            但它仍然会通过 _is_ignored() 排除 _IGNORED_DIRS 中的目录。
        """
        _ = signal, on_update
        params = request.arguments
        start_path = str(params.get("path", "."))
        pattern = str(params.get("pattern", "**/*"))
        max_results = int(params.get("max_results", 200))

        # 路径解析与校验
        root, error = _resolve_path(sandbox, start_path)
        if error is not None:
            return error
        if not root.exists():
            return _error_result(f"Path not found: {start_path}", "path_not_found")

        # 收集候选路径：按 glob 匹配并排除忽略目录，
        # 按 POSIX 路径排序以保证结果的可重复性。
        candidates = sorted(
            (path for path in root.glob(pattern) if not _is_ignored(path, workspace)),
            key=lambda path: path.as_posix(),
        )

        # 遍历候选列表，转换为相对路径，目录追加 "/" 以区别于文件
        results = []
        for path in candidates[:_MAX_SCAN_FILES]:
            rel = path.relative_to(workspace).as_posix()
            results.append(rel + ("/" if path.is_dir() else ""))
            if len(results) >= max_results:
                break

        truncated = len(results) >= max_results or len(candidates) > _MAX_SCAN_FILES

        return ToolResult(
            content=[TextContent(text="\n".join(results) if results else "(no files)")],
            details={"count": len(results)},
            metadata={
                "search_paths": [root.relative_to(workspace).as_posix()],
                "truncated": truncated,
                "output_quality": {"truncated": truncated},
            },
        )

    # =========================================================================
    # 权限检查与工具注册
    # =========================================================================

    # 仅当 allow 回调返回 True 时才创建对应的工具定义
    if allow("grep"):
        tools.append(_tool("grep", "Search Content", "在文件或目录内容里按正则搜索。", _grep_schema(), grep_tool))
    if allow("find"):
        tools.append(_tool("find", "Find Files", "按 glob 查找文件/目录路径。", _find_schema(), find_tool))
    return tools


# =============================================================================
# 内部辅助函数
# =============================================================================

def _tool(name: str, label: str, description: str, parameters: dict[str, Any], execute) -> ToolDefinition:
    """
    构造单个 ToolDefinition 实例的辅助函数。

    从工具注册中心获取该工具名称对应的预定义元数据（如分类、图标等），
    与传入的参数、执行函数组合成完整的 ToolDefinition。

    参数:
        name:        工具名称（如 "grep"、"find"），与注册中心元数据中的名称对应。
        label:       工具的短标签，用于 UI 展示。
        description: 工具的描述文本（中文），供 AI 模型理解工具用途。
        parameters:  JSON Schema 格式的参数定义，描述工具接受的参数结构。
        execute:      异步执行函数，签名为 async (ToolCallRequest) -> ToolResult。

    返回:
        完整填充的 ToolDefinition 实例。

    异常:
        ValueError: 当在注册中心找不到对应工具的元数据时抛出。
    """
    metadata = get_builtin_tool_metadata(name)
    if metadata is None:
        raise ValueError(f"Missing builtin metadata for {name}")
    return ToolDefinition(
        name=name,
        label=label,
        description=description,
        parameters=parameters,
        metadata=metadata,
        execute=execute,
    )


def _resolve_path(sandbox: WorkspaceSandbox, path_text: str) -> tuple[Any | None, ToolResult | None]:
    """
    通过沙箱解析用户输入的路径字符串。

    参数:
        sandbox:   WorkspaceSandbox 实例，封装了工作区根目录和路径安全校验逻辑。
        path_text: 用户输入的路径字符串（相对路径或受限的绝对路径）。

    返回:
        二元组 (resolved_path, error_result):
          - 成功: (Path 对象, None)
          - 失败: (None, ToolResult 错误对象)  —— 当路径尝试越界访问工作区外部时。

    安全性:
        WorkspaceSandbox.resolve_path() 会校验路径是否在允许的工作区范围内，
        如果检测到路径试图访问工作区外部（如 "../" 逃逸或符号链接越界），
        会抛出 ValueError，由本函数捕获并转换为友好的错误 ToolResult。
    """
    try:
        return sandbox.resolve_path(path_text), None
    except ValueError:
        return None, _error_result(f"Path escapes workspace boundary: {path_text}", "path_escapes_workspace")


def _error_result(message: str, error_code: str) -> ToolResult:
    """
    构造标准的错误 ToolResult。

    参数:
        message:    用户可读的错误描述信息。
        error_code: 机器可读的错误码（如 "missing_pattern"、"path_not_found"、"path_escapes_workspace"）。

    返回:
        标记为错误状态的 ToolResult，包含 recovery_hint 供 AI 模型参考如何进行恢复。
    """
    return ToolResult(
        content=[TextContent(text=message)],
        status="error",
        is_error=True,
        error_code=error_code,
        metadata={"recovery_hint": {"message": "Adjust the search path or pattern and retry."}},
    )


def _is_ignored(path, root) -> bool:
    """
    判断给定路径是否属于应被忽略的目录。

    逻辑：
        1. 计算 path 相对于 root 的相对路径。
        2. 如果 path 不在 root 子树内（relative_to 失败），视为应忽略（返回 True）。
        3. 遍历相对路径的每一级目录组件，若任一级出现在 _IGNORED_DIRS 集合中，
           则该路径应被忽略。

    参数:
        path: 待检查的文件或目录 Path 对象。
        root: 搜索的根目录 Path 对象。

    返回:
        True  表示该路径应被忽略（位于排除目录中或不在根目录子树内）。
        False 表示该路径可以正常搜索。

    示例:
        假设 root = /workspace，_IGNORED_DIRS 包含 "node_modules"：
          - /workspace/src/main.py             -> False（正常搜索）
          - /workspace/node_modules/lodash/... -> True（被忽略）
          - /workspace/app/.git/config         -> True（被忽略）
    """
    try:
        relative = path.relative_to(root)
    except ValueError:
        # path 不在 root 的子树内（例如符号链接指向外部），视为应忽略
        return True
    # 检查相对路径的每一级是否命中忽略目录列表
    return any(part in _IGNORED_DIRS for part in relative.parts)


def _is_binary(path) -> bool:
    """
    通过检测 null 字节来判断文件是否为二进制文件。

    原理：
        文本文件（UTF-8、ASCII 等）的字节流中不会出现 null 字节（\x00）。
        而几乎所有二进制格式（可执行文件、压缩包、图片、对象文件等）的前几个字节中
        通常会包含 null 字节。因此读取文件的前 4096 字节并检查是否包含 \x00
        是一种高效且准确的二进制文件检测方法。

    参数:
        path: 待检测文件的 Path 对象。

    返回:
        True  表示文件是二进制文件（或无法读取，保守视为二进制跳过）。
        False 表示文件可能是文本文件，可以进行内容搜索。

    注意:
        此检测方法对 UTF-8/UTF-16 文本文件有效，但对某些不含 null 字节的
        二进制格式（如纯 ASCII PGM 图像文件）可能产生假阴性。
        不过在实际代码仓库中，这种假阴性极少发生且影响有限
        （最多导致一次无效的正则匹配尝试）。
    """
    try:
        with path.open("rb") as handle:
            chunk = handle.read(4096)
    except OSError:
        # 文件无法打开（权限不足、已删除等），保守视为二进制并跳过
        return True
    return b"\x00" in chunk


# =============================================================================
# JSON Schema 参数定义
# =============================================================================

def _grep_schema() -> dict[str, Any]:
    """
    grep_tool 的 JSON Schema 参数定义。

    定义 grep 工具接受的参数结构，用于：
      1. 向 AI 模型声明工具的参数格式（Function Calling / Tool Use）。
      2. 在工具调用时进行参数校验。

    参数说明:
        pattern:       正则表达式模式（必填）。
        path:          搜索起始路径或具体文件（可选，默认 "."）。
        glob:          path 为目录时的文件筛选 glob 模式（可选，默认 "**/*"）。
        max_matches:   最大匹配行数（可选，默认 200）。
        case_sensitive: 是否区分大小写（可选，默认 true）。

    additionalProperties 设为 False，拒绝未定义的额外参数。
    """
    return {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "搜索起始目录或具体文件路径。"},
            "glob": {"type": "string", "description": "path 为目录时使用的文件筛选 glob。"},
            "max_matches": {"type": "integer"},
            "case_sensitive": {"type": "boolean"},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }


def _find_schema() -> dict[str, Any]:
    """
    find_tool 的 JSON Schema 参数定义。

    定义 find 工具接受的参数结构，用于：
      1. 向 AI 模型声明工具的参数格式（Function Calling / Tool Use）。
      2. 在工具调用时进行参数校验。

    参数说明:
        path:       搜索起始路径（可选，默认 "."）。
        pattern:    glob 匹配模式（可选，默认 "**/*"）。
        max_results: 最大结果数（可选，默认 200）。

    注意：find_tool 没有必填参数——不传任何参数时返回工作区根目录下的所有文件和目录。
    additionalProperties 设为 False，拒绝未定义的额外参数。
    """
    return {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "pattern": {"type": "string"},
            "max_results": {"type": "integer"},
        },
        "required": [],
        "additionalProperties": False,
    }


__all__ = ["create_search_tools"]
