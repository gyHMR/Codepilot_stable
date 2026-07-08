from __future__ import annotations

"""
内置文件系统工具模块：ls、read、write、edit、apply_patch。

本模块是 Codepilot 工具系统中的核心文件操作层，在 WorkspaceSandbox（工作区沙箱）
的约束下提供安全、可追溯的文件 CRUD 操作。所有工具通过 create_file_tools() 工厂
函数统一创建，返回 ToolDefinition 列表供工具注册表使用。

核心设计原则：
1. 路径安全：所有路径操作需经过沙箱解析，禁止逃逸到工作区外部。
2. 可追溯性：每次修改都记录变更证据（change_evidence），包含前后哈希值。
3. UTF-8 约束：仅处理有效的 UTF-8 文本文件，遇到二进制或非 UTF-8 文件返回友好错误。
4. 唯一性约束：编辑工具默认要求 old_text 在文件中唯一匹配，防止意外修改多处。

提供的工具：
- ls：列出目录内容，显示文件名和文件大小。
- read：分页读取文本文件，支持 offset/limit 分页和 max_chars 截断。
- write：写入文本文件，支持覆盖控制和变更检测（内容未变则不标记为已修改）。
- edit：基于 old_text -> new_text 的单文件替换编辑，支持匹配计数和文件哈希校验。
- apply_patch：批量编辑工具，对多个文件执行唯一匹配替换，整体作为原子操作提交。
"""

from dataclasses import dataclass
from typing import Any, Callable

from codepilot.protocols import TextContent
from codepilot.tools.contracts import ToolCallRequest, ToolDefinition, ToolResult
from codepilot.tools.registry import get_builtin_tool_metadata
from codepilot.tools.sandbox import WorkspaceSandbox, file_state_for_path


@dataclass(frozen=True)
class _PatchEdit:
    """
    内部编辑操作的数据载体（不可变数据类）。

    表示一次文本替换操作的核心三元组：
    - path:   目标文件在工作区中的相对路径。
    - old_text: 要查找并替换的原始文本片段。
    - new_text: 用于替换的新文本片段。

    同时被 edit_tool（单次编辑）和 apply_patch_tool（批量编辑）复用。
    """
    path: str
    old_text: str
    new_text: str


def create_file_tools(
    sandbox: WorkspaceSandbox,
    *,
    allow: Callable[[str], bool],
    edit_require_unique_match: bool = True,
) -> list[ToolDefinition]:
    """
    文件工具工厂函数 —— 创建所有内置文件系统工具。

    这是本模块唯一的公开入口。调用方传入工作区沙箱实例和一个 allow 谓词函数，
    由工厂函数根据 allow 判断结果有条件地向工具列表中添加工具。

    参数：
        sandbox: WorkspaceSandbox 实例，封装了工作区根目录和路径解析/边界检查逻辑。
                 所有路径操作都以该沙箱为基准进行解析。
        allow: 单参数谓词函数，接收工具名称字符串（如 "ls", "read", "write", "edit",
               "apply_patch"），返回 bool 值表示是否允许该工具。通过此机制可实现
               细粒度的工具权限控制。
        edit_require_unique_match: 控制 edit_tool 的匹配唯一性策略。默认为 True，
               即当 old_text 在文件中出现多次且未指定 occurrence_index 时，
               edit_tool 会拒绝执行并返回错误，要求调用方提供更精确的 old_text。

    返回值：
        list[ToolDefinition] —— 根据 allow 谓词过滤后的工具定义列表。
        每个 ToolDefinition 包含工具名称、标签、中文描述、参数 JSON Schema、
        元数据和异步执行函数。

    内部流程：
        1. 定义 max_write_chars = 1_000_000 限制单次写入的最大字符数。
        2. 依次定义五个闭包工具函数（ls_tool, read_tool, write_tool, edit_tool,
           apply_patch_tool），每个闭包捕获 sandbox 和工厂参数。
        3. 调用 allow("tool_name") 判断每个工具，允许则调用 _tool() 包装并追加。
        4. _tool() 内部通过 get_builtin_tool_metadata() 获取工具的预注册元数据，
           丢失则抛 ValueError。
    """
    tools: list[ToolDefinition] = []
    # 单次写入操作的最大字符数上限（约 1MB 文本）
    max_write_chars = 1_000_000

    # ──────────────────────────────────────────────────────────────────
    # ls_tool — 目录列表工具
    # ──────────────────────────────────────────────────────────────────
    async def ls_tool(request: ToolCallRequest, signal=None, on_update=None) -> ToolResult:
        """
        列出指定目录的内容，返回文件名和文件大小信息。

        参数（来自 request.arguments）：
            path: 要列出的目录路径，默认为 "."（当前目录）。
            max_entries: 最大返回条目数，默认 100。超过该数量会被截断，
                        并在 output_quality.truncated 中标记。

        输出格式：
            每行一条记录，格式为 "filename\t<size_or_dash>"：
            - 目录：名称后带 "/" 后缀，大小列显示 "-"。
            - 文件：直接显示名称，大小列显示字节数。

        返回值：
            ToolResult，其 content 中包含以换行分隔的目录列表文本。
            metadata 中包含 output_quality.truncated 标记是否因 max_entries 截断。
        """
        _ = signal, on_update  # 显式忽略未使用的参数
        params = request.arguments
        path_text = str(params.get("path", "."))
        max_entries = int(params.get("max_entries", 100))

        # 通过沙箱解析路径，确保不逃逸工作区
        target, error = _resolve_read_path(sandbox, path_text)
        if error is not None:
            return error

        # 路径不存在
        if not target.exists():
            return _error_result(f"Path not found: {path_text}", "path_not_found")

        # 路径不是目录
        if not target.is_dir():
            return _error_result(f"Not a directory: {path_text}", "not_a_directory")

        # 列出目录条目，按名称排序后截取前 max_entries 条
        items = sorted(target.iterdir(), key=lambda path: path.name)[:max_entries]

        # 构建每一行的输出文本：名称[后缀]\t大小
        lines = []
        for item in items:
            # 目录添加 "/" 后缀，大小显示 "-"
            suffix = "/" if item.is_dir() else ""
            # 目录不显示字节大小，显示 "-"
            size = "-" if item.is_dir() else str(item.stat().st_size)
            lines.append(f"{item.name}{suffix}\t{size}")

        # 计算相对于工作区根目录的路径
        rel = target.relative_to(sandbox.root).as_posix()

        return ToolResult(
            content=[TextContent(text="\n".join(lines) if lines else "(empty)")],
            metadata={
                "read_paths": [rel],
                "output_quality": {"truncated": len(lines) >= max_entries},
            },
        )

    # ──────────────────────────────────────────────────────────────────
    # read_tool — 文件读取工具
    # ──────────────────────────────────────────────────────────────────
    async def read_tool(request: ToolCallRequest, signal=None, on_update=None) -> ToolResult:
        """
        分页读取文本文件内容，支持 offset/limit 行分页和 max_chars 字符截断。

        参数（来自 request.arguments）：
            path:      要读取的文件路径（必填）。
            max_chars: 返回内容的最大字符数，默认 20000。当行内容累计超过该值时，
                       后续行被截断，并在输出中标注截断原因。
            offset:    起始行号（1-based），默认从第 1 行开始。
            limit:     最大读取行数，默认 200 行。

        分页与截断机制：
            先按分页（offset + limit）选定候选行范围，再按 max_chars 进行字符级截断。
            两者任一触发都会在输出中注明截断原因（"max_chars" 或 "line_limit"）。
            当有更多内容可读时，输出末尾会给出下一次 read 调用的推荐参数。

        输出格式：
            lines <start>-<end> of <total>
            <line_no>\t<line_content>
            ...
            ...<truncated: <reason>>...   （如有截断）
            next: read(path="...", offset=N, limit=M)   （如有更多内容）

        返回值：
            ToolResult，其 content 中包含格式化的文件片段文本。
            metadata 中包含丰富的状态信息：
            - file_state: 当前文件状态（路径、大小、SHA256 哈希等）。
            - actual_start_line / actual_end_line: 实际返回的起止行号。
            - returned_lines / total_lines: 返回行数 / 总行数。
            - has_more: 是否还有更多内容可读。
            - next_offset: 下次读取的推荐 offset 值。
            - truncated / truncated_reason: 截断标记及原因。
            - output_quality: 输出质量元数据（编码状态、截断、可靠性评估等）。
        """
        _ = signal, on_update
        params = request.arguments
        path_text = str(params.get("path", ""))

        # 路径为必填参数
        if not path_text:
            return _error_result("Missing path", "missing_path")

        max_chars = int(params.get("max_chars", 20000))
        offset = int(params.get("offset", 1))
        # limit 可为 None（不传），此时默认 200 行
        limit = params.get("limit")
        limit = int(limit) if limit is not None else 200

        # 路径安全解析
        target, error = _resolve_read_path(sandbox, path_text)
        if error is not None:
            return error

        # 路径存在性检查
        if not target.exists():
            return _error_result(f"Path not found: {path_text}", "path_not_found")

        # 类型检查：必须是文件，不能是目录
        if not target.is_file():
            return _error_result(f"Not a file: {path_text}", "not_a_file")

        # 读取原始文本内容
        try:
            raw = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # 非 UTF-8 文本文件（可能是二进制文件）
            return _error_result(
                f"File is not valid UTF-8 text: {path_text}",
                "invalid_utf8",
                metadata={"output_quality": _output_quality(decode_status="invalid_utf8", may_be_binary=True)},
            )

        # 按行拆分
        lines = raw.splitlines()

        # 计算分页窗口：
        # start_index: 将 1-based offset 转为 0-based 索引，并限制在有效范围内
        start_index = min(max(offset, 1) - 1, len(lines))
        # requested_end_index: 按 limit 截取行数，不超过文件总行数
        requested_end_index = min(len(lines), start_index + max(limit, 1))

        # 构建候选行列表，格式为 "行号\t行内容"
        candidates = [
            f"{line_no}\t{line}"
            for line_no, line in enumerate(lines[start_index:requested_end_index], start=start_index + 1)
        ]

        # ── 按 max_chars 进行字符级截断 ──
        selected: list[str] = []     # 最终选中的行列表
        body_chars = 0                # 已累计的字符数（含换行分隔符开销）
        char_truncated = False        # 是否因 max_chars 触发了截断

        for rendered_line in candidates:
            # extra: 当前行在输出中的字符开销（行内容 + 与前行的换行分隔符）
            extra = len(rendered_line) + (1 if selected else 0)

            # 已有内容时，累加后会超过 max_chars，截断
            if selected and body_chars + extra > max_chars:
                char_truncated = True
                break

            # 首行就超过 max_chars：不返回任何行，直接标记截断
            if not selected and len(rendered_line) > max_chars:
                char_truncated = True
                break

            selected.append(rendered_line)
            body_chars += extra

        # 实际返回的起止行号（1-based），没有选中行则为 None
        actual_start_line = start_index + 1 if selected else None
        actual_end_line = start_index + len(selected) if selected else None

        # 是否还有未读取的行
        has_more = (actual_end_line or start_index) < len(lines)

        # 下一次读取的推荐 offset（一行之后开始）
        next_offset = (actual_end_line + 1) if actual_end_line and has_more else None

        # 因行数限制（limit）导致的截断标记
        line_limit_truncated = requested_end_index < len(lines) and not char_truncated

        # 统一的截断原因描述
        truncated_reason = "max_chars" if char_truncated else ("line_limit" if line_limit_truncated else None)
        truncated = bool(truncated_reason)

        # 组装输出文本
        header_end = actual_end_line or start_index  # 实际返回的最后行号（用于页眉）
        rendered_parts = [f"lines {start_index + 1}-{header_end} of {len(lines)}"]  # 页眉

        if selected:
            rendered_parts.append("\n".join(selected))  # 文件内容体

        if has_more:
            if truncated_reason:
                rendered_parts.append(f"...<truncated: {truncated_reason}>...")  # 截断提示
            # 下一页的推荐调用方式
            rendered_parts.append(
                f'next: read(path="{path_text}", offset={next_offset}, limit={max(limit, 1)})'
            )

        rendered = "\n".join(rendered_parts)

        # 文件路径和状态信息
        relative_path = target.relative_to(sandbox.root).as_posix()
        state = file_state_for_path(sandbox.root, relative_path)

        return ToolResult(
            content=[TextContent(text=rendered or "(empty)")],
            details={"file_state": state},
            metadata={
                "file_state": state,
                "read_paths": [relative_path],
                "path": relative_path,
                "start_line": actual_start_line,
                "end_line": actual_end_line,
                "actual_start_line": actual_start_line,
                "actual_end_line": actual_end_line,
                "returned_lines": len(selected),
                "total_lines": len(lines),
                "has_more": has_more,
                "next_offset": next_offset,
                "truncated": truncated,
                "char_truncated": char_truncated,
                "truncated_reason": truncated_reason,
                "output_quality": _output_quality(
                    truncated=truncated,
                    original_chars=len(raw),
                    returned_chars=len(rendered),
                ),
            },
        )

    # ──────────────────────────────────────────────────────────────────
    # write_tool — 文件写入工具
    # ──────────────────────────────────────────────────────────────────
    async def write_tool(request: ToolCallRequest, signal=None, on_update=None) -> ToolResult:
        """
        将文本内容写入文件，支持覆盖控制和变更检测。

        参数（来自 request.arguments）：
            path:      目标文件路径（必填）。
            content:   要写入的文本内容（必填）。
            overwrite: 是否允许覆盖已存在的文件，默认 True。
                       设为 False 时，若目标文件已存在则返回错误。

        智能变更检测：
            写入前先读取目标文件的当前内容（如果存在），与待写入内容进行比较。
            若内容完全相同，则跳过实际写入操作，返回 "File unchanged" 结果，
            且 workspace_changed=False，避免触发不必要的下游变更传播。

        原子写入保证：
            - 先创建父目录（exist_ok=True）。
            - 使用 utf-8 编码写入，统一换行符为 "\n"。

        尺寸限制：
            单次写入内容不能超过 max_write_chars（1,000,000 字符）。

        返回值：
            ToolResult，其 content 中包含操作结果描述。
            metadata 中包含 change_evidence（变更证据），记录操作类型
            （create/update/unchanged）、前后文件哈希值等。
        """
        _ = signal, on_update
        params = request.arguments
        path_text = str(params.get("path", ""))
        content = str(params.get("content", ""))
        overwrite = bool(params.get("overwrite", True))

        # 路径必填检查
        if not path_text:
            return _error_result("Missing path", "missing_path")

        # 内容大小限制
        if len(content) > max_write_chars:
            return _error_result("Content is too large", "content_too_large")

        # 写入路径解析（带可变性校验，确保目标在可写区域内）
        target, error = _resolve_write_path(sandbox, path_text)
        if error is not None:
            return error

        # 目标存在但非文件（如目录）
        if target.exists() and not target.is_file():
            return _error_result(f"Target is not a file: {path_text}", "target_not_file")

        # overwrite=False 且文件已存在
        if target.exists() and not overwrite:
            return _error_result(f"File exists: {path_text}", "file_exists")

        # 读取原始内容（用于变更检测）
        try:
            original = target.read_text(encoding="utf-8") if target.exists() else None
        except UnicodeDecodeError:
            return _error_result(f"Existing file is not valid UTF-8: {path_text}", "invalid_utf8")

        relative_path = target.relative_to(sandbox.root).as_posix()
        before_hash = _state_hash(sandbox.root, relative_path)

        # ── 内容未变更检测 ──
        # 如果新内容与原始内容完全相同，跳过写入，避免无意义的文件修改
        if original == content:
            state = file_state_for_path(sandbox.root, relative_path)
            return ToolResult(
                content=[TextContent(text=f"File unchanged: {relative_path}")],
                affected_paths=[relative_path],
                workspace_changed=False,  # 工作区未发生实际变化
                diff_summary="No content change",
                details={"changed": False, "file_state": state},
                metadata={
                    "file_state": state,
                    "change_evidence": _change_evidence(
                        "unchanged", relative_path, before_hash, _state_hash(sandbox.root, relative_path)
                    ),
                },
            )

        # ── 执行写入 ──
        # 确保父目录存在
        target.parent.mkdir(parents=True, exist_ok=True)
        # 写入文件，统一换行符为 LF
        target.write_text(content, encoding="utf-8", newline="\n")

        state = file_state_for_path(sandbox.root, relative_path)

        # 区分"创建"与"更新"操作
        action = "created" if original is None else "updated"

        return ToolResult(
            content=[TextContent(text=f"Wrote file: {relative_path}")],
            affected_paths=[relative_path],
            workspace_changed=True,
            diff_summary=f"{action} {relative_path}: {len(original or '')} -> {len(content)} characters",
            details={"changed": True, "action": action, "file_state": state},
            metadata={
                "file_state": state,
                "change_evidence": _change_evidence(
                    "create" if original is None else "update",
                    relative_path,
                    before_hash,
                    str(state.get("sha256", "<missing>")),
                ),
            },
        )

    # ──────────────────────────────────────────────────────────────────
    # edit_tool — 单文件文本替换编辑工具
    # ──────────────────────────────────────────────────────────────────
    async def edit_tool(request: ToolCallRequest, signal=None, on_update=None) -> ToolResult:
        """
        按 old_text -> new_text 替换文件内容，支持精确匹配控制。

        参数（来自 request.arguments）：
            path:                 目标文件路径（必填）。
            old_text:             要替换的原始文本（必填，不可为空）。
            new_text:             替换后的新文本（必填，可为空字符串）。
            replace_all:          是否替换所有匹配项，默认 False。
            occurrence_index:     指定替换第几次出现（1-based），优先级高于 replace_all。
            expected_occurrences: 期望的 old_text 出现次数，用于校验匹配数量。
            expected_file_hash:   期望的文件 SHA256 哈希值，用于检测并发修改（stale file）。

        唯一性约束（受 edit_require_unique_match 控制）：
            当 replace_all=False、未指定 occurrence_index 且 old_text 出现多次时，
            若 edit_require_unique_match=True，则返回 multiple_matches 错误，
            要求调用方提供更精确的 old_text 或使用 occurrence_index 明确指定。

        实现：
            委托给 _apply_single_edit() 核心编辑函数执行实际替换逻辑。
        """
        _ = signal, on_update
        params = request.arguments

        # 从参数构建编辑数据载体
        edit = _PatchEdit(
            path=str(params.get("path", "")),
            old_text=str(params.get("old_text", "")),
            new_text=str(params.get("new_text", "")),
        )

        # 解析可选的控制参数
        replace_all = bool(params.get("replace_all", False))
        occurrence_index = params.get("occurrence_index")
        occurrence_index = int(occurrence_index) if occurrence_index is not None else None
        expected_occurrences = params.get("expected_occurrences")
        expected_occurrences = int(expected_occurrences) if expected_occurrences is not None else None
        expected_hash = params.get("expected_file_hash")

        # 委托核心编辑逻辑
        return _apply_single_edit(
            sandbox,
            edit,
            replace_all=replace_all,
            occurrence_index=occurrence_index,
            expected_occurrences=expected_occurrences,
            expected_file_hash=str(expected_hash) if expected_hash is not None else None,
            require_unique_match=edit_require_unique_match,
            max_chars=max_write_chars,
            label="Edited file",
        )

    # ──────────────────────────────────────────────────────────────────
    # apply_patch_tool — 批量补丁应用工具
    # ──────────────────────────────────────────────────────────────────
    async def apply_patch_tool(request: ToolCallRequest, signal=None, on_update=None) -> ToolResult:
        """
        对多个文件执行批量结构化编辑（补丁）。

        参数（来自 request.arguments）：
            edits: 编辑操作的列表，每个元素是一个包含 path/old_text/new_text 的对象。
                   列表长度限制为 1~20 个编辑。

        与 edit_tool 的区别：
            - edit_tool 针对单个文件提供灵活的匹配控制（replace_all、occurrence_index 等）。
            - apply_patch_tool 面向批量场景，每个编辑要求 exactly one match（唯一匹配），
              且所有编辑的校验在写入前完成（原子性：要么全部通过校验，要么全部拒绝）。

        执行流程：
            1. 校验 edits 参数（类型、数量上限）。
            2. 逐条构建 _PatchEdit 对象并验证字段完整性。
            3. 预检（pre-flight check）阶段：
               a. 解析每个文件路径并验证文件存在且可读。
               b. 读取原始内容并统计 old_text 的出现次数。
               c. 必须 exactly one match，否则立即返回 patch_match_count 错误。
               d. 预先计算替换后的内容，记录变更前哈希值。
            4. 预检全部通过后，一次性写入所有文件。
            5. 收集变更证据并返回统一结果。

        返回值：
            ToolResult，content 显示成功应用的编辑数量。
            metadata.change_evidence 中包含每条编辑的变更证据列表。
        """
        _ = signal, on_update
        params = request.arguments
        raw_edits = params.get("edits")

        # edits 必须是至少包含一个元素的列表
        if not isinstance(raw_edits, list) or not raw_edits:
            return _error_result("edits must contain at least one edit", "invalid_patch")

        # 批量编辑上限：最多 20 个编辑
        if len(raw_edits) > 20:
            return _error_result("apply_patch supports at most 20 edits", "patch_too_large")

        # 第一遍：构建 _PatchEdit 对象列表
        edits: list[_PatchEdit] = []
        for index, item in enumerate(raw_edits):
            if not isinstance(item, dict):
                return _error_result(f"edits[{index}] must be an object", "invalid_patch")
            edit = _PatchEdit(
                path=str(item.get("path", "")),
                old_text=str(item.get("old_text", "")),
                new_text=str(item.get("new_text", "")),
            )
            if not edit.path or edit.old_text == "":
                return _error_result(f"edits[{index}] requires path and old_text", "invalid_patch")
            edits.append(edit)

        # 第二遍：预检阶段 — 读取文件、统计匹配数、预先计算替换结果
        # prepared 列表元素：(edit, target_path, relative_path, updated_content, before_hash)
        prepared: list[tuple[_PatchEdit, Any, str, str, str]] = []
        for edit in edits:
            target, error = _resolve_write_path(sandbox, edit.path)
            if error is not None:
                return error
            if not target.exists() or not target.is_file():
                return _error_result(f"Path not found or not file: {edit.path}", "path_not_file")
            try:
                original = target.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return _error_result(f"File is not valid UTF-8: {edit.path}", "invalid_utf8")

            # 计算匹配次数：apply_patch 要求恰好匹配一次
            count = original.count(edit.old_text)
            if count != 1:
                return _error_result(
                    f"Patch for {edit.path} expected exactly one match, found {count}",
                    "patch_match_count",
                    metadata={"matches": count},
                )

            # 预先计算替换后的文本（只替换第一个出现的匹配）
            updated = original.replace(edit.old_text, edit.new_text, 1)
            rel = target.relative_to(sandbox.root).as_posix()
            before_hash = _state_hash(sandbox.root, rel)
            prepared.append((edit, target, rel, updated, before_hash))

        # 第三遍：预检全部通过，执行实际写入
        affected: list[str] = []
        evidences = []
        for _edit, target, rel, updated, before_hash in prepared:
            target.write_text(updated, encoding="utf-8", newline="\n")
            affected.append(rel)
            evidences.append(
                _change_evidence("update", rel, before_hash, _state_hash(sandbox.root, rel))
            )

        return ToolResult(
            content=[TextContent(text=f"Applied patch edits: {len(prepared)}")],
            affected_paths=affected,
            workspace_changed=True,
            diff_summary=f"applied {len(prepared)} patch edit(s)",
            details={"edits": len(prepared)},
            metadata={"change_evidence": evidences},
        )

    # ── 根据 allow 谓词有条件地注册工具 ──
    # 每个工具的中文描述用于向中文用户提供友好的工具说明
    if allow("ls"):
        tools.append(_tool("ls", "List Directory", "列出目录内容，返回文件名和大小。", _ls_schema(), ls_tool))
    if allow("read"):
        tools.append(_tool("read", "Read File", "读取文本文件内容。", _read_schema(), read_tool))
    if allow("write"):
        tools.append(_tool("write", "Write File", "写入文本文件。", _write_schema(), write_tool))
    if allow("edit"):
        tools.append(_tool("edit", "Edit File", "按 old_text -> new_text 替换文件内容。", _edit_schema(), edit_tool))
    if allow("apply_patch"):
        tools.append(_tool("apply_patch", "Apply Patch", "按结构化 edits 对一个或多个文件执行唯一匹配替换，适合多处小改。", _apply_patch_schema(), apply_patch_tool))

    return tools


def _apply_single_edit(
    sandbox: WorkspaceSandbox,
    edit: _PatchEdit,
    *,
    replace_all: bool,
    occurrence_index: int | None,
    expected_occurrences: int | None,
    expected_file_hash: str | None,
    require_unique_match: bool,
    max_chars: int,
    label: str,
) -> ToolResult:
    """
    核心编辑逻辑 —— 执行单次文本替换操作。

    这是 edit_tool 和 apply_patch_tool 共用的底层实现。edit_tool 通过丰富的
    控制参数提供灵活的编辑能力，apply_patch_tool 则通过固定参数（replace_all=False,
    require_unique_match 等）实现批量唯一匹配编辑。

    执行流程（按顺序的防御性校验链）：

    1. 参数校验
       - edit.path 不能为空。
       - edit.old_text 不能为空（空字符串无法匹配任何内容）。
       - old_text + new_text 的总长度不能超过 max_chars 限制。

    2. 路径与文件校验
       - 通过 _resolve_write_path 解析并验证路径在工作区可写范围内。
       - 目标文件必须存在且为普通文件。

    3. UTF-8 编码校验
       - 读取整个文件内容，非 UTF-8 则返回错误。

    4. 文件哈希校验（Stale File Detection，过期文件检测）
       - 若调用方提供了 expected_file_hash，则与当前文件的 SHA256 哈希比对。
       - 不匹配说明文件在最后一次读取后已被修改，返回 stale_file 错误。
       - 此机制防止基于过期文件内容进行的编辑覆盖他人的并发修改。

    5. 匹配次数校验
       - 使用 str.count() 统计 old_text 的出现次数。
       - 若 expected_occurrences 被指定，校验实际匹配数是否等于期望值。
       - 若匹配数为 0，返回 no_match 错误。
       - 若匹配数 > 1、未指定 replace_all、未指定 occurrence_index、且
         require_unique_match=True，则返回 multiple_matches 错误。

    6. 执行替换（三种模式，按优先级排序）
       - replace_all=True: 替换所有出现的 old_text。
       - occurrence_index 指定: 仅替换第 N 次出现（通过 _replace_nth 实现）。
       - 默认: 仅替换第一次出现。

    7. 写入与结果组装
       - 将替换后的内容写回文件。
       - 若实际内容未变化（updated == original），workspace_changed=False。
       - 返回包含变更证据的 ToolResult。

    参数：
        sandbox: 工作区沙箱实例。
        edit: 编辑操作的三元组（path, old_text, new_text）。
        replace_all: 是否替换所有匹配项。
        occurrence_index: 指定替换第几次出现的匹配（1-based）。
        expected_occurrences: 期望的匹配次数，用于校验。
        expected_file_hash: 期望的文件哈希，用于过期检测。
        require_unique_match: 是否要求唯一匹配。
        max_chars: 编辑载荷大小上限。
        label: 结果描述中的操作标签（如 "Edited file"）。

    返回值：
        ToolResult，包含替换次数、文件状态、变更证据等信息。
    """
    # ── 1. 参数校验 ──
    if not edit.path:
        return _error_result("Missing path", "missing_path")
    if edit.old_text == "":
        return _error_result("old_text cannot be empty", "empty_old_text")
    if len(edit.old_text) + len(edit.new_text) > max_chars:
        return _error_result("Edit payload is too large", "content_too_large")

    # ── 2. 路径解析与文件存在性校验 ──
    target, error = _resolve_write_path(sandbox, edit.path)
    if error is not None:
        return error
    if not target.exists() or not target.is_file():
        return _error_result(f"Path not found or not file: {edit.path}", "path_not_file")

    # ── 3. 读取原始文件内容 ──
    try:
        original = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return _error_result(f"File is not valid UTF-8: {edit.path}", "invalid_utf8")

    # ── 4. 文件哈希校验（过期文件检测）──
    relative_path = target.relative_to(sandbox.root).as_posix()
    state = file_state_for_path(sandbox.root, relative_path)
    before_hash = str(state.get("sha256", "<missing>"))

    if expected_file_hash is not None and before_hash != expected_file_hash:
        # 文件自上次读取后已被修改，拒绝编辑以避免覆盖他人变更
        return _error_result(
            "File changed since it was read; read it again before editing",
            "stale_file",
            metadata={"file_state": state},
        )

    # ── 5. 匹配次数校验 ──
    count = original.count(edit.old_text)

    # 期望匹配次数的显式校验
    if expected_occurrences is not None and count != expected_occurrences:
        return _error_result(
            f"Expected {expected_occurrences} matches, found {count}",
            "unexpected_match_count",
        )

    # 无匹配
    if count == 0:
        return _error_result("No match found", "no_match")

    # 多匹配且未指定唯一选择策略时的拒绝逻辑
    if not replace_all and count > 1 and occurrence_index is None and require_unique_match:
        return _error_result(
            "Multiple matches found; refine old_text or use occurrence_index",
            "multiple_matches",
        )

    # ── 6. 执行替换（三种模式）──
    if replace_all:
        # 模式 A: 替换所有匹配项
        updated = original.replace(edit.old_text, edit.new_text)
        replacements = count
    elif occurrence_index is not None:
        # 模式 B: 替换第 N 次出现
        if occurrence_index < 1 or occurrence_index > count:
            return _error_result("occurrence_index is out of range", "occurrence_out_of_range")
        # _replace_nth 精确替换第 occurrence_index 次出现
        updated = _replace_nth(original, edit.old_text, edit.new_text, occurrence_index)
        replacements = 1
    else:
        # 模式 C: 默认替换第一次出现
        updated = original.replace(edit.old_text, edit.new_text, 1)
        replacements = 1

    # ── 7. 写入文件 ──
    target.write_text(updated, encoding="utf-8", newline="\n")

    # 获取写入后的新状态
    new_state = file_state_for_path(sandbox.root, relative_path)

    return ToolResult(
        content=[TextContent(text=f"{label}: {relative_path} (replacements={replacements})")],
        affected_paths=[relative_path],
        # 若替换后内容未实际变化，则工作区未修改
        workspace_changed=updated != original,
        diff_summary=f"edited {relative_path}: {replacements} replacement(s)",
        details={"replacements": replacements, "file_state": new_state},
        metadata={
            "file_state": new_state,
            "change_evidence": _change_evidence(
                "update" if updated != original else "unchanged",
                relative_path,
                before_hash,
                str(new_state.get("sha256", "<missing>")),
            ),
        },
    )


def _tool(name: str, label: str, description: str, parameters: dict[str, Any], execute) -> ToolDefinition:
    """
    工具包装工厂 —— 将执行函数和元数据组装为 ToolDefinition。

    参数：
        name:        工具的唯一名称标识（如 "ls", "read", "write", "edit", "apply_patch"）。
        label:       工具的英文人类可读标签。
        description: 工具的中文描述，用于向用户说明工具功能。
        parameters:  JSON Schema 格式的参数定义。
        execute:     工具的异步执行函数。

    实现：
        从工具注册表中获取预定义的元数据（get_builtin_tool_metadata），
        元数据包括工具的版本、分类、权限等信息。若元数据不存在则抛出异常，
        这是防御性编程 —— 确保所有内置工具都有已注册的元数据。

    返回值：
        ToolDefinition 实例，可直接添加到工具定义列表中。
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


def _resolve_read_path(sandbox: WorkspaceSandbox, path_text: str) -> tuple[Any | None, ToolResult | None]:
    """
    只读取路径解析 —— 通过沙箱解析路径并检查工作区边界。

    与其他路径解析函数的区别：
        _resolve_read_path 仅用于读取操作（ls_tool、read_tool），
        不执行可变性检查（mutable check），因为读取操作不需要写入权限。

    参数：
        sandbox:   工作区沙箱实例。
        path_text: 待解析的路径字符串（相对于工作区根目录或绝对路径）。

    返回值：
        二元组 (resolved_path, error_result)：
        - 成功时返回 (Path对象, None)。
        - 路径逃逸工作区边界时返回 (None, ToolResult错误)。
    """
    try:
        return sandbox.resolve_path(path_text), None
    except ValueError:
        # 路径逃逸工作区边界被沙箱拒绝
        return None, _error_result(f"Path escapes workspace boundary: {path_text}", "path_escapes_workspace")


def _resolve_write_path(sandbox: WorkspaceSandbox, path_text: str) -> tuple[Any | None, ToolResult | None]:
    """
    可写路径解析 —— 通过沙箱解析路径并校验可写性。

    与 _resolve_read_path 的区别：
        _resolve_write_path 额外调用 ensure_mutable_path()，确保目标路径
        在工作区的可写区域内（而非只读区域，如系统文件或依赖包目录）。

    参数：
        sandbox:   工作区沙箱实例。
        path_text: 待解析的路径字符串。

    返回值：
        二元组 (resolved_path, error_result)：
        - 成功时返回 (Path对象, None)。
        - 路径逃逸或不可写时返回 (None, ToolResult错误)。
    """
    try:
        return sandbox.ensure_mutable_path(sandbox.resolve_path(path_text)), None
    except ValueError as exc:
        return None, _error_result(str(exc), "path_escapes_workspace")


def _error_result(
    message: str,
    error_code: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> ToolResult:
    """
    构建标准化的错误 ToolResult。

    统一的错误返回工厂，确保所有工具返回的错误具有一致的结构：
    - status="error" 标记工具执行失败。
    - is_error=True 供上层错误处理逻辑快速判断。
    - error_code 作为可编程的错误分类标识。
    - metadata 中自动附加恢复提示（recovery_hint），帮助 AI 或用户自行纠错。

    参数：
        message:    用户可读的错误描述文本。
        error_code: 错误码（如 "path_not_found", "multiple_matches", "stale_file" 等）。
        metadata:   额外的元数据字典，会与错误默认元数据合并。

    返回值：
        ToolResult 错误实例。
    """
    return ToolResult(
        content=[TextContent(text=message)],
        status="error",
        is_error=True,
        error_code=error_code,
        metadata={**_metadata_for_error(error_code), **(metadata or {})},
    )


def _metadata_for_error(error_code: str) -> dict[str, Any]:
    """
    根据错误码生成恢复提示（Recovery Hint）元数据。

    恢复提示是嵌入在错误 metadata 中的自然语言建议，引导 AI 或用户
    在遇到特定错误时采取正确的修正操作。支持的常见错误码及其建议：

    - path_not_found / path_not_file: 建议列表或搜索工作区确认路径。
    - path_escapes_workspace: 建议使用工作区内的路径。
    - stale_file: 建议重新读取文件后再编辑。
    - multiple_matches: 建议读取更大范围的上下文以提供唯一的 old_text。
    - no_match: 建议先读取当前文件内容。
    - invalid_patch: 建议按正确的格式传递参数。

    若错误码不在已知列表中，返回空字典（不附加恢复提示）。
    """
    hints = {
        "path_not_found": "List or search the workspace to confirm the path.",
        "path_not_file": "List or search the workspace to confirm the path.",
        "path_escapes_workspace": "Use a path inside the current workspace.",
        "stale_file": "Read the file again before editing.",
        "multiple_matches": "Read a larger target region and provide unique old_text.",
        "no_match": "Read the current file content before retrying.",
        "invalid_patch": "Pass edits as [{path, old_text, new_text}].",
    }
    if error_code not in hints:
        return {}
    return {"recovery_hint": {"message": hints[error_code]}}


def _output_quality(
    *,
    decode_status: str = "ok",
    truncated: bool = False,
    original_chars: int | None = None,
    returned_chars: int | None = None,
    may_be_binary: bool = False,
) -> dict[str, Any]:
    """
    生成输出质量元数据 —— 记录读取操作返回内容的质量特征。

    此函数为 read_tool 的 output_quality 字段提供标准化结构，
    帮助下游消费者（AI 模型、日志系统、监控面板）评估返回内容的可靠性。

    字段说明：
        encoding:     编码格式，正常为 "utf-8"，解码失败时为 "unknown"。
        decode_status: 解码状态，"ok" 或 "invalid_utf8"。
        truncated:    是否因 max_chars 或 line_limit 发生了截断。
        original_chars: 原始文件的字符总数。
        returned_chars: 实际返回的字符数。
        may_be_binary: 是否可能为二进制文件（基于解码失败推断）。
        reliable_for_reasoning: 综合可靠性评估 —— 当编码正常、未截断、
                                且非二进制时为 True，表示内容完整可用于推理。
    """
    return {
        "encoding": "utf-8" if decode_status != "invalid_utf8" else "unknown",
        "decode_status": decode_status,
        "truncated": truncated,
        "original_chars": original_chars,
        "returned_chars": returned_chars,
        "may_be_binary": may_be_binary,
        "reliable_for_reasoning": decode_status == "ok" and not truncated and not may_be_binary,
    }


def _change_evidence(change_kind: str, path: str, before_hash: str, after_hash: str) -> dict[str, Any]:
    """
    生成变更证据（Change Evidence）—— 文件操作的可审计记录。

    变更证据记录了文件修改操作的前后状态，供审计、回滚和变更追溯使用。

    字段说明：
        change_kind:  变更类型（"create" / "update" / "unchanged"）。
        before_hashes: 变更前的文件哈希映射（{路径: 哈希值}）。
        after_hashes:  变更后的文件哈希映射（{路径: 哈希值}）。
        affected_paths: 受影响文件的相对路径列表。
        effect_detection: 效应检测方式，固定为 "direct"（直接文件写入）。
        effect_detection_confidence: 检测置信度，固定为 "high"。
        safe_revert_available: 是否有安全的回滚路径，当前固定为 False。
    """
    return {
        "change_kind": change_kind,
        "before_hashes": {path: before_hash},
        "after_hashes": {path: after_hash},
        "affected_paths": [path],
        "effect_detection": "direct",
        "effect_detection_confidence": "high",
        "safe_revert_available": False,
    }


def _state_hash(workspace: Any, path: str) -> str:
    """
    计算指定文件的当前 SHA256 哈希值（辅助函数）。

    内部通过 file_state_for_path 获取文件的完整状态信息，然后提取
    SHA256 字段。若文件不存在或哈希不可用，返回字符串 "<missing>"。

    参数：
        workspace: 工作区根目录路径。
        path:      相对于工作区根目录的文件路径。

    返回值：
        文件的 SHA256 哈希值字符串，或 "<missing>"。
    """
    return str(file_state_for_path(workspace, path).get("sha256", "<missing>"))


def _replace_nth(text: str, old: str, new: str, nth: int) -> str:
    """
    替换文本中第 N 次出现的子串（1-based）。

    与 str.replace(old, new) 的区别：
        Python 内置的 str.replace 接受 count 参数表示替换"前 N 次出现"，
        但不支持精确指定"仅替换第 N 次出现"。本函数填补了这一空白。

    实现方式：
        通过循环使用 str.find() 逐次定位 old 的出现位置，定位到第 nth 次时
        执行字符串拼接替换（text[:found] + new + text[found + len(old):]）。

    参数：
        text: 原始文本。
        old:  要查找的子串。
        new:  替换用的新子串。
        nth:  要替换的序号（1-based，即第 1 次、第 2 次……）。

    返回值：
        替换后的文本。

    异常：
        ValueError: 若 old 在 text 中的出现次数不足 nth 次。
    """
    start = 0
    for index in range(nth):
        found = text.find(old, start)
        if found < 0:
            raise ValueError("nth occurrence not found")
        # 当找到第 nth 次出现时，执行替换
        if index == nth - 1:
            return text[:found] + new + text[found + len(old) :]
        # 非目标出现，跳过并继续向后搜索
        start = found + len(old)
    return text


# ══════════════════════════════════════════════════════════════════════
# JSON Schema 定义 —— 各工具的参数定义
# ══════════════════════════════════════════════════════════════════════


def _ls_schema() -> dict[str, Any]:
    """
    ls 工具的参数 JSON Schema。

    参数：
        path (可选):       目标目录路径，字符串类型。
        max_entries (可选): 最大条目数，整数类型。
    """
    return {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "max_entries": {"type": "integer"},
        },
        "required": [],
        "additionalProperties": False,
    }


def _read_schema() -> dict[str, Any]:
    """
    read 工具的参数 JSON Schema。

    参数：
        path (必填):     目标文件路径。
        max_chars (可选): 最大返回字符数，整数类型。
        offset (可选):    起始行号（1-based），整数类型。
        limit (可选):     最大返回行数，整数类型。
    """
    return {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "max_chars": {"type": "integer"},
            "offset": {"type": "integer"},
            "limit": {"type": "integer"},
        },
        "required": ["path"],
        "additionalProperties": False,
    }


def _write_schema() -> dict[str, Any]:
    """
    write 工具的参数 JSON Schema。

    参数：
        path (必填):      目标文件路径。
        content (必填):   要写入的文本内容。
        overwrite (可选): 是否允许覆盖已存在文件，布尔类型，默认 true。
    """
    return {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "overwrite": {"type": "boolean"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }


def _edit_schema() -> dict[str, Any]:
    """
    edit 工具的参数 JSON Schema。

    参数：
        path (必填):                  目标文件路径。
        old_text (必填):              要被替换的原始文本。
        new_text (必填):              替换后的新文本。
        replace_all (可选):           是否替换所有匹配项，布尔类型。
        occurrence_index (可选):      指定替换第几次出现，整数类型（1-based）。
        expected_occurrences (可选):  期望的匹配次数，整数类型。
        expected_file_hash (可选):    期望的文件 SHA256 哈希值，字符串类型。
    """
    return {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
            "replace_all": {"type": "boolean"},
            "occurrence_index": {"type": "integer"},
            "expected_occurrences": {"type": "integer"},
            "expected_file_hash": {"type": "string"},
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    }


def _apply_patch_schema() -> dict[str, Any]:
    """
    apply_patch 工具的参数 JSON Schema。

    参数：
        edits (必填): 编辑列表，数组类型，长度 1~20。
            每个元素为对象，包含：
            - path (必填):     目标文件路径。
            - old_text (必填): 要被替换的原始文本。
            - new_text (必填): 替换后的新文本。
    """
    return {
        "type": "object",
        "properties": {
            "edits": {
                "type": "array",
                "minItems": 1,
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                    },
                    "required": ["path", "old_text", "new_text"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["edits"],
        "additionalProperties": False,
    }


__all__ = ["create_file_tools"]
