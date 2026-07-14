"""规范的工作区文件工具 —— ls / read / write / edit / apply_patch。

本文件实现了所有文件系统操作的内置工具，包括目录列表、文件读取、
文件写入、精确文本替换和批量补丁应用。

每个工具通过 create_file_registrations() 注册，包含完整的六个组件：
- Spec（模型可见的定义）
- Codec（编解码器）
- Handler（业务逻辑）
- Resolver（访问权限解析）
- Renderer（结果渲染）
- Policy（安全策略）

安全设计：
- 所有路径操作经过 WorkspaceSandbox 进行沙箱隔离
- 写操作需要 execute 模式 + 权限审批
- 读操作在 plan 和 execute 模式下均可
- 写操作串行执行（serial），读操作可并行（parallel）
- 批量写操作（超过 20 文件或 500KB）触发额外的批量审批限制
- 原子化写入（apply_patch）：先写临时文件，再 os.replace 替换，失败回滚
"""

import asyncio
import hashlib
import os
import uuid
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

# 批量操作自动审批阈值
# 当文件数 > 20 或估计写入字节 > 500KB 时，提升风险等级并限制审批范围
_AUTO_APPROVE_MAX_FILES = 20
_AUTO_APPROVE_MAX_BYTES = 500_000


# ── 输入类型（dataclass） ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class LsInput:
    """ls 工具的输入参数。

    参数:
        path: 要列出的目录路径（默认当前目录 "."）
        max_entries: 最多返回的条目数（默认 100）
    """
    path: str = "."
    max_entries: int = 100


@dataclass(frozen=True)
class ReadInput:
    """read 工具的输入参数。

    参数:
        path: 要读取的文件路径
        max_chars: 最多读取的字符数（默认 20,000）
        offset: 开始读取的行号（从 1 开始，默认 1）
        limit: 最多读取的行数（默认 200）
    """
    path: str
    max_chars: int = 20_000
    offset: int = 1
    limit: int = 200


@dataclass(frozen=True)
class WriteInput:
    """write 工具的输入参数。

    参数:
        path: 要写入的文件路径
        content: 文件内容
        overwrite: 是否覆盖已存在的文件（默认 True）
    """
    path: str
    content: str
    overwrite: bool = True


@dataclass(frozen=True)
class EditInput:
    """edit 工具的输入参数。

    参数:
        path: 要编辑的文件路径
        old_text: 要被替换的旧文本（必须精确匹配）
        new_text: 替换后的新文本
        replace_all: 是否替换所有匹配（默认 False）
        occurrence_index: 替换第几次出现的匹配（可选，从 1 开始）
        expected_occurrences: 预期的匹配次数（可选，用于验证）
        expected_file_hash: 期望的文件 SHA256 哈希（可选，用于验证文件未变）
    """
    path: str
    old_text: str
    new_text: str
    replace_all: bool = False
    occurrence_index: int | None = None
    expected_occurrences: int | None = None
    expected_file_hash: str | None = None


@dataclass(frozen=True)
class ApplyPatchInput:
    """apply_patch 工具的输入参数（批量编辑）。

    参数:
        edits: 编辑操作列表，每个元素包含 path/old_text/new_text
    """
    edits: list[dict[str, Any]]


# ── 输出类型 ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FileOutput:
    """文件操作的统一输出类型。

    参数:
        text: 给 LLM 看的文本描述
        details: 结构化的详细数据
        metadata: 元数据（如是否截断）
        affected_paths: 受影响的文件路径列表（仅写操作）
        workspace_changed: 工作区是否发生了变更
        diff_summary: 变更摘要文本
    """
    text: str
    details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    affected_paths: list[str] = field(default_factory=list)
    workspace_changed: bool = False
    diff_summary: str | None = None


# 输出 Schema（所有文件工具共享）
_OUTPUT_SCHEMA = {
    "$schema": _DRAFT,
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "details": {"type": "object"},
        "metadata": {"type": "object"},
        "affected_paths": {"type": "array", "items": {"type": "string"}},
        "workspace_changed": {"type": "boolean"},
        "diff_summary": {"type": ["string", "null"]},
    },
    "required": [
        "text",
        "details",
        "metadata",
        "affected_paths",
        "workspace_changed",
        "diff_summary",
    ],
    "additionalProperties": False,
}


# ── 注册创建函数 ──────────────────────────────────────────────────────────────


def create_file_registrations(
    sandbox: WorkspaceSandbox,
    *,
    allow: Callable[[str], bool],
    edit_require_unique_match: bool = True,
) -> list[ToolRegistration]:
    """创建所有启用的文件操作工具注册。

    注册 5 个文件工具：ls / read / write / edit / apply_patch
    每个工具都绑定 input_codec、output_codec、handler、resolver、renderer、policy。

    处理流程:
    1. 遍历工具配置元组
    2. 对每个工具：创建 codec、构建 ToolRegistration
    3. 所有工具共享 _FileHandler 和 _Resolver（通过 name 区分）
    4. 未启用的工具（allow() 返回 False 的）被跳过

    参数:
        sandbox: 工作区沙箱实例
        allow: 工具启用过滤器（接收工具名称，返回是否启用）
        edit_require_unique_match: edit 工具是否要求唯一匹配（默认 True）

    返回:
        ToolRegistration 列表（只包含被启用的工具）
    """
    configs = (
        ("ls", LsInput, _ls_schema(), False, "List one workspace directory with names and file sizes."),
        ("read", ReadInput, _read_schema(), False, "Read a UTF-8 workspace file with line pagination and truncation metadata."),
        ("write", WriteInput, _write_schema(), True, "Create or replace one UTF-8 workspace file and report whether content changed."),
        ("edit", EditInput, _edit_schema(), True, "Replace exact text in one UTF-8 workspace file with occurrence and hash guards."),
        ("apply_patch", ApplyPatchInput, _patch_schema(), True, "Validate and atomically apply one to twenty exact text replacements."),
    )
    registrations: list[ToolRegistration] = []
    for name, input_type, input_schema, mutating, description in configs:
        if not allow(name):
            continue
        input_codec = DataclassCodec(input_type, input_schema)
        output_codec = DataclassCodec(FileOutput, _OUTPUT_SCHEMA)
        registrations.append(
            ToolRegistration(
                version="1.0.0",
                implementation_version="2",
                spec=ToolSpec(name, description, input_schema, _OUTPUT_SCHEMA),
                category="filesystem",
                source="builtin",
                owner="codepilot.builtin",
                policy=_policy(mutating),
                input_codec=input_codec,
                output_codec=output_codec,
                handler=_FileHandler(
                    sandbox=sandbox,
                    name=name,
                    unique_edit=edit_require_unique_match,
                ),
                renderer=_Renderer(),
                access_resolver=_Resolver(sandbox, name, mutating),
            )
        )
    return registrations


# ── 访问解析器 ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _ResolvedFileInput:
    value: object
    targets: tuple[Path, ...]


@dataclass(frozen=True)
class _Resolver:
    """文件工具访问解析器 —— 将路径参数解析为权限请求。

    对每个文件操作，解析目标路径：
    - 通过 sandbox 解析路径（防止逃逸）
    - 根据是否 mutating 选择读/写沙箱检查
    - 构造 ToolAccessRequest（包含资源、效果、风险等级）
    - 对批量写入操作，估算文件数和字节数，决定是否提升风险等级

    批量操作甄别：
    - 超过 _AUTO_APPROVE_MAX_FILES（20）或 _AUTO_APPROVE_MAX_BYTES（500KB）
      的操作会被标记为 "bulk"，风险提升为 medium，审批范围限制为 once
    - 防止 LLM 通过单次调用大量写入文件来绕过审批

    参数:
        sandbox: 工作区沙箱
        name: 工具名称
        mutating: 是否是变更操作
    """
    sandbox: WorkspaceSandbox
    name: str
    mutating: bool

    def resolve(self, input, request):
        """解析输入中的路径为访问权限请求。

        对于 apply_patch，遍历所有 edits 中的 path。
        对于其他工具，取 input.path。

        参数:
            input: 解码后的输入对象
            request: 原始执行请求

        返回:
            ToolAccessResolution 包含访问权限信息
        """
        _ = request
        paths = (
            [str(item.get("path", "")) for item in input.edits]
            if self.name == "apply_patch"
            else [str(input.path)]
        )
        resources: list[ToolResource] = []
        targets: list[Path] = []
        for raw in paths:
            target = self.sandbox.resolve_path(raw)
            target = (
                self.sandbox.ensure_mutable_path(target)
                if self.mutating
                else self.sandbox.ensure_readable_path(target)
            )
            targets.append(target)
            resources.append(_resource(self.sandbox, target))
        effects = (
            frozenset({"filesystem_read", "filesystem_write"})
            if self.mutating
            else frozenset({"filesystem_read"})
        )
        # 估算批量写入大小，超过阈值则提升风险
        file_count, estimated_bytes = _mutation_size(self.name, input)
        bulk = self.mutating and (
            file_count > _AUTO_APPROVE_MAX_FILES
            or estimated_bytes > _AUTO_APPROVE_MAX_BYTES
        )
        return ToolAccessResolution(
            input=_ResolvedFileInput(input, tuple(targets)),
            access=ToolAccessRequest(
                actions=(f"{self.name}.bulk" if bulk else self.name,),
                resources=tuple(resources),
                effects=effects,
                risk="medium" if bulk else "low",
                reason=f"{self.name} workspace path(s)",
                safe_preview={
                    "paths": paths,
                    "file_count": file_count,
                    "estimated_bytes": estimated_bytes,
                    "operation_profile": (
                        "bulk_write"
                        if bulk
                        else "workspace_write" if self.mutating else "workspace_read"
                    ),
                },
                approval_scopes=(
                    frozenset({"once"})
                    if bulk
                    else frozenset({"once", "session", "project"})
                ),
            ),
        )


# ── 处理器 ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _FileHandler:
    """文件工具处理器 —— 实际的文件操作逻辑。

    根据 name 分发到不同的处理方法：
    - ls: 列出目录内容
    - read: 读取文件内容（带行分页和截断）
    - write: 创建或替换文件
    - edit: 精确文本替换（支持哈希验证、匹配次数验证）
    - apply_patch: 批量精确文本替换（原子化写入 + 失败回滚）

    参数:
        sandbox: 工作区沙箱
        name: 工具名称
        unique_edit: edit 是否要求唯一匹配
    """

    sandbox: WorkspaceSandbox
    name: str
    unique_edit: bool

    async def __call__(self, resolved: _ResolvedFileInput, context: ToolExecutionContext) -> FileOutput:
        """入口方法 —— 根据工具名称分发到具体处理逻辑。

        在开始前检查取消信号（cancellation.raise_if_cancelled）。

        参数:
            input: 解码后的输入（LsInput / ReadInput / WriteInput 等）
            context: 工具执行上下文

        返回:
            FileOutput 统一输出
        """
        context.cancellation.raise_if_cancelled()
        input = resolved.value
        if self.name == "ls":
            return self._ls(input, resolved.targets[0], context)
        if self.name == "read":
            return self._read(input, resolved.targets[0], context)
        if self.name == "write":
            return self._write(input, resolved.targets[0], context)
        if self.name == "edit":
            return self._edit(input, resolved.targets[0], context)
        return self._patch(input, resolved.targets, context)

    def _ls(self, input: LsInput, target: Path, context: ToolExecutionContext) -> FileOutput:
        """列出目录内容。

        处理流程:
        1. 解析路径并验证可读性
        2. 检查路径是否是一个目录
        3. 排序条目（目录在前，其余按字母序）
        4. 截断到 max_entries 限制
        5. 报告副作用

        参数:
            input: LsInput（path, max_entries）
            context: 执行上下文

        返回:
            FileOutput 包含目录列表文本和元数据
        """
        target = self.sandbox.ensure_readable_path(target)
        if not target.is_dir():
            raise ToolHandlerError("ls.not_directory", f"Directory not found: {input.path}")
        entries = sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        shown = entries[: input.max_entries]
        lines = [
            f"{item.name}/" if item.is_dir() else f"{item.name}\t{item.stat().st_size} bytes"
            for item in shown
        ]
        truncated = len(entries) > len(shown)
        if truncated:
            lines.append(f"... {len(entries) - len(shown)} more entries")
        self._effect(context, target, "filesystem_read", "list directory")
        relative = self.sandbox.relative_path(target) or "."
        return FileOutput(
            text="\n".join(lines),
            details={"path": relative, "entry_count": len(entries)},
            metadata={"truncated": truncated},
        )

    def _read(self, input: ReadInput, target: Path, context: ToolExecutionContext) -> FileOutput:
        """读取文件内容。

        处理流程:
        1. 解析路径并验证可读性
        2. 检查文件是否存在
        3. 读取 UTF-8 编码的文件
        4. 按 offset/limit 行号分页
        5. 按 max_chars 截断（在最近的换行符处截断，避免截断行）
        6. 报告副作用

        输出元数据包含 output_quality 信息，LLM 可以据此判断
        截断是否影响推理可靠性。

        参数:
            input: ReadInput（path, max_chars, offset, limit）
            context: 执行上下文

        返回:
            FileOutput 包含文件内容和分页/截断元数据
        """
        target = self.sandbox.ensure_readable_path(target)
        if not target.is_file():
            raise ToolHandlerError("read.not_file", f"File not found: {input.path}")
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolHandlerError("read.not_utf8", f"File is not valid UTF-8: {input.path}") from exc
        lines = text.splitlines(keepends=True)
        start = min(len(lines), input.offset - 1)
        selected = lines[start : start + input.limit]
        rendered = "".join(selected)
        char_truncated = len(rendered) > input.max_chars
        if char_truncated:
            rendered = rendered[: input.max_chars]
            if "\n" in rendered:
                rendered = rendered[: rendered.rfind("\n") + 1]
        line_truncated = start + len(selected) < len(lines)
        truncated = char_truncated or line_truncated
        self._effect(context, target, "filesystem_read", "read file")
        relative = self.sandbox.relative_path(target)
        return FileOutput(
            text=rendered,
            details={
                "path": relative,
                "offset": input.offset,
                "line_count": len(lines),
                "returned_lines": len(rendered.splitlines()),
            },
            metadata={
                "truncated": truncated,
                "output_quality": {
                    "truncated": truncated,
                    "original_chars": len(text),
                    "returned_chars": len(rendered),
                    "reliable_for_reasoning": not truncated,
                },
            },
        )

    def _write(self, input: WriteInput, target: Path, context: ToolExecutionContext) -> FileOutput:
        """创建或替换文件。

        处理流程:
        1. 解析路径并验证可修改性
        2. 检查是否应阻止覆盖（overwrite=False 且文件已存在）
        3. 读取原内容做变更检测（content unchanged = no-op）
        4. 内容变更时：创建父目录 → 写入新内容（强制 \n 换行符）
        5. 返回是否变更的信息

        变更检测确保如果写入内容与原内容相同，不会实际写盘，
        避免不必要的文件修改时间戳更新。

        参数:
            input: WriteInput（path, content, overwrite）
            context: 执行上下文

        返回:
            FileOutput 包含写入结果（"Updated" 或 "Unchanged"）
        """
        target = self.sandbox.ensure_mutable_path(target)
        if target.exists() and not input.overwrite:
            raise ToolHandlerError("write.exists", f"File already exists: {input.path}")
        previous = target.read_text(encoding="utf-8") if target.is_file() else None
        changed = previous != input.content
        if changed:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(input.content, encoding="utf-8", newline="\n")
        return self._mutation_output(target, changed, context, "write file")

    def _edit(self, input: EditInput, target: Path, context: ToolExecutionContext) -> FileOutput:
        """对文件执行精确的文本替换。

        支持四种替换模式:
        1. occurrence_index: 替换第 N 次匹配（从 1 开始）
        2. replace_all: 替换所有匹配
        3. 唯一匹配（默认）: 如果 count!=1 则报错（由 unique_edit 控制）
        4. replace_all=False 且无 occurrence_index: 仅替换第一次

        安全保护:
        - expected_file_hash: 如果提供，先验证文件哈希是否匹配
          （防止并发修改导致替换位置错误）
        - expected_occurrences: 如果提供，先验证匹配次数
          （防止误判，确认 LLM 对替换目标数量的理解正确）

        参数:
            input: EditInput（path, old_text, new_text, ...）
            context: 执行上下文

        返回:
            FileOutput 包含编辑结果（含 replacements 计数）
        """
        target = self.sandbox.ensure_mutable_path(target)
        if not target.is_file():
            raise ToolHandlerError("edit.not_file", f"File not found: {input.path}")
        text = target.read_text(encoding="utf-8")
        if input.expected_file_hash and _hash_text(text) != input.expected_file_hash:
            raise ToolHandlerError("edit.hash_mismatch", "File content hash no longer matches")
        count = text.count(input.old_text)
        if input.expected_occurrences is not None and count != input.expected_occurrences:
            raise ToolHandlerError("edit.occurrence_mismatch", f"Expected {input.expected_occurrences} matches, found {count}")
        if count == 0:
            raise ToolHandlerError("edit.no_match", "old_text was not found")
        if input.occurrence_index is not None:
            updated = _replace_occurrence(text, input.old_text, input.new_text, input.occurrence_index)
            replacements = 1
        elif input.replace_all:
            updated = text.replace(input.old_text, input.new_text)
            replacements = count
        else:
            if self.unique_edit and count != 1:
                raise ToolHandlerError("edit.match_not_unique", f"old_text matched {count} times")
            updated = text.replace(input.old_text, input.new_text, 1)
            replacements = 1
        changed = updated != text
        if changed:
            target.write_text(updated, encoding="utf-8", newline="\n")
        output = self._mutation_output(target, changed, context, "edit file")
        return FileOutput(**{**output.__dict__, "details": {"replacements": replacements}})

    def _patch(self, input: ApplyPatchInput, targets: tuple[Path, ...], context: ToolExecutionContext) -> FileOutput:
        """批量应用文本替换补丁（原子化写入 + 失败回滚）。

        原子化保证：
        1. 先验证所有编辑的匹配（在内存中模拟修改）
        2. 对所有变更文件执行原子化写入（先写临时文件，再 os.replace）
        3. 如果任何一个写入失败，回滚所有已替换的文件
        4. 如果操作被取消（CancelledError），也执行回滚

        支持 1-20 个编辑操作，每个操作包含 path/old_text/new_text。
        每个编辑操作必须唯一匹配（count=1），防止歧义。

        参数:
            input: ApplyPatchInput（edits 列表）
            context: 执行上下文

        返回:
            FileOutput 包含补丁应用结果
        """
        if not 1 <= len(input.edits) <= 20:
            raise ToolHandlerError("apply_patch.invalid_count", "edits must contain between 1 and 20 items")
        staged: dict[Path, str] = {}
        originals: dict[Path, str] = {}
        changed_paths: list[Path] = []
        for index, (edit, resolved_target) in enumerate(zip(input.edits, targets, strict=True)):
            path = str(edit.get("path", ""))
            old_text = edit.get("old_text")
            new_text = edit.get("new_text")
            if not isinstance(old_text, str) or not isinstance(new_text, str):
                raise ToolHandlerError("apply_patch.invalid_edit", f"edits[{index}] requires string old_text/new_text")
            target = self.sandbox.ensure_mutable_path(resolved_target)
            if not target.is_file():
                raise ToolHandlerError("apply_patch.not_file", f"File not found: {path}")
            current = staged.get(target)
            if current is None:
                current = target.read_text(encoding="utf-8")
                originals[target] = current
            count = current.count(old_text)
            if count != 1:
                raise ToolHandlerError("apply_patch.match_not_unique", f"edits[{index}] matched {count} times")
            staged[target] = current.replace(old_text, new_text, 1)
            if target not in changed_paths:
                changed_paths.append(target)
        # 原子化写入所有变更
        _replace_files_atomically(
            {target: staged[target] for target in changed_paths},
            originals,
            context,
        )
        for target in changed_paths:
            self._effect(context, target, "filesystem_read", "validate patch")
            self._effect(context, target, "filesystem_write", "apply patch")
        relatives = [self.sandbox.relative_path(path) for path in changed_paths]
        return FileOutput(
            text=f"Applied {len(input.edits)} edit(s) across {len(changed_paths)} file(s).",
            details={"edit_count": len(input.edits)},
            affected_paths=relatives,
            workspace_changed=True,
            diff_summary=f"updated {len(changed_paths)} file(s)",
        )

    def _mutation_output(
        self,
        target: Path,
        changed: bool,
        context: ToolExecutionContext,
        operation: str,
    ) -> FileOutput:
        """构建变更操作的通用输出。

        无论 write/edit，变更后的输出结构一致：
        - 报告读取副作用（变更前检查）
        - 如果实际变更，报告写入副作用
        - 返回 "Updated path" 或 "Unchanged path"

        参数:
            target: 目标文件路径
            changed: 内容是否实际变更
            context: 执行上下文
            operation: 操作描述（"write file" / "edit file"）

        返回:
            FileOutput 对象
        """
        self._effect(context, target, "filesystem_read", f"{operation} precondition")
        if changed:
            self._effect(context, target, "filesystem_write", operation)
        relative = self.sandbox.relative_path(target)
        return FileOutput(
            text=f"{'Updated' if changed else 'Unchanged'} {relative}",
            affected_paths=[relative] if changed else [],
            workspace_changed=changed,
            diff_summary=f"updated {relative}" if changed else None,
        )

    def _effect(
        self,
        context: ToolExecutionContext,
        target: Path,
        kind: str,
        operation: str,
    ) -> None:
        """报告一个文件操作副作用。

        参数:
            context: 执行上下文（包含 effects reporter）
            target: 文件路径
            kind: 副作用类型（filesystem_read / filesystem_write）
            operation: 操作描述
        """
        context.effects.report(
            ToolEffect(
                kind=kind,
                resource=_resource(self.sandbox, target),
                operation=operation,
                status="completed",
                certainty="observed",
            )
        )


# ── 渲染器 ────────────────────────────────────────────────────────────────────


class _Renderer:
    """渲染器 —— 将 FileOutput 转为 LLM 可消费的 TextContent。"""
    def render(self, data):
        """渲染文件工具输出为模型可消费的文本内容块。"""
        return (TextContent(text=str(data["text"])),)


# ── 策略 ──────────────────────────────────────────────────────────────────────


def _policy(mutating: bool) -> ToolPolicy:
    """构建文件工具的安全策略。

    读/写策略的关键区别：
    - allowed_modes: 写操作只允许在 execute 模式（plan 模式下不可写）
    - declared_effects: 写操作包含 filesystem_write
    - required_permissions: 写操作需要 workspace.write
    - base_risk: 写操作为 medium，读操作为 low
    - approval: 写操作为 on_risk（按风险审批），读操作为 never
    - concurrency: 写操作串行执行（防止文件竞争），读操作可并行

    参数:
        mutating: 是否是变更操作

    返回:
        ToolPolicy 安全策略
    """
    return ToolPolicy(
        allowed_modes=frozenset({"execute"} if mutating else {"plan", "execute"}),
        declared_effects=(
            frozenset({"filesystem_read", "filesystem_write"})
            if mutating
            else frozenset({"filesystem_read"})
        ),
        required_permissions=(
            frozenset({"workspace.read", "workspace.write"})
            if mutating
            else frozenset({"workspace.read"})
        ),
        base_risk="medium" if mutating else "low",
        approval="on_risk" if mutating else "never",
        timeout=TimeoutPolicy(15_000, 60_000),
        concurrency=(
            ConcurrencyPolicy(mode="serial", group="workspace_mutation")
            if mutating
            else ConcurrencyPolicy(mode="parallel")
        ),
        output_limits=OutputLimits(),
        output_trust=OutputTrustPolicy(),
    )


# ── 辅助函数 ──────────────────────────────────────────────────────────────────


def _resource(sandbox: WorkspaceSandbox, target: Path) -> ToolResource:
    """构建工作区资源的 URI。

    格式: "workspace:///相对路径"
    相对路径通过 sandbox.relative_path() 计算（确保在工作区内）。
    """
    return ToolResource("workspace:///" + sandbox.relative_path(target))


def _mutation_size(name: str, input: object) -> tuple[int, int]:
    """估算写入操作的规模（文件数和字节数）。

    用于 Resolver 判断是否触发批量操作限制（bulk）。
    - write: 1 个文件，content 字节数
    - edit: 1 个文件，old_text + new_text 字节数
    - apply_patch: 所有 edits 的 old_text + new_text 字节数总和

    参数:
        name: 工具名称
        input: 解码后的输入对象

    返回:
        (file_count, estimated_bytes) 文件数和估计字节数
    """
    if name == "write":
        return 1, len(input.content.encode("utf-8"))
    if name == "edit":
        return 1, len(input.old_text.encode("utf-8")) + len(input.new_text.encode("utf-8"))
    if name == "apply_patch":
        return len(input.edits), sum(
            len(str(item.get("old_text", "")).encode("utf-8"))
            + len(str(item.get("new_text", "")).encode("utf-8"))
            for item in input.edits
        )
    return 1, 0


def _hash_text(text: str) -> str:
    """计算文本的 SHA256 哈希。

    用于 edit 工具的 expected_file_hash 验证。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _replace_occurrence(text: str, old: str, new: str, index: int) -> str:
    """替换文本中的第 N 次匹配。

    参数:
        text: 原始文本
        old: 要替换的旧文本
        new: 新文本
        index: 第几次匹配（从 1 开始）

    返回:
        替换后的文本

    抛出:
        ToolHandlerError: index 无效或匹配不存在
    """
    if index < 1:
        raise ToolHandlerError("edit.invalid_occurrence", "occurrence_index must be positive")
    start = -1
    cursor = 0
    for _ in range(index):
        start = text.find(old, cursor)
        if start < 0:
            raise ToolHandlerError("edit.occurrence_missing", f"Occurrence {index} was not found")
        cursor = start + len(old)
    return text[:start] + new + text[start + len(old) :]


def _replace_files_atomically(
    staged: dict[Path, str],
    originals: dict[Path, str],
    context: ToolExecutionContext,
) -> None:
    """原子化替换多个文件。

    策略：
    1. 对所有目标文件，先写入临时文件（同目录，UUID 文件名）
    2. 所有临时文件写入完毕后，用 os.replace 替换原文件
    3. 如果任何一步失败（异常或取消），回滚已替换的文件

    回滚：
    - 对已被替换的文件，恢复原始内容
    - 清理所有临时文件
    - 如果是 CancelledError，重新抛出（上层处理）
    - 如果是其他异常，抛出 ToolHandlerError

    参数:
        staged: 目标文件 → 新内容 的映射
        originals: 目标文件 → 原始内容 的映射（用于回滚）
        context: 执行上下文（用于检查取消）

    抛出:
        ToolHandlerError: 原子写入失败，可能包含回滚失败信息
    """
    temporary: dict[Path, Path] = {}
    replaced: list[Path] = []
    try:
        for target, content in staged.items():
            context.cancellation.raise_if_cancelled()
            temporary[target] = _write_replacement_file(target, content)
        for target, temp in temporary.items():
            context.cancellation.raise_if_cancelled()
            os.replace(temp, target)
            replaced.append(target)
    except (Exception, asyncio.CancelledError) as exc:
        rollback_errors: list[str] = []
        for target in reversed(replaced):
            try:
                restore = _write_replacement_file(target, originals[target])
                os.replace(restore, target)
            except OSError as rollback_exc:
                rollback_errors.append(f"{target.name}: {rollback_exc}")
        for temp in temporary.values():
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass
        if isinstance(exc, asyncio.CancelledError):
            raise
        message = f"Atomic patch failed: {exc}"
        if rollback_errors:
            message += "; rollback failed for " + ", ".join(rollback_errors)
        raise ToolHandlerError("apply_patch.atomic_write_failed", message) from exc


def _write_replacement_file(target: Path, content: str) -> Path:
    """写入替换临时文件（原子化写入的辅助函数）。

    在同目录下创建临时文件（格式: .{target}.{uuid}.tmp），
    写入内容后调用 fsync 确保数据落盘，然后复制原文件权限。

    参数:
        target: 目标文件路径
        content: 要写入的内容

    返回:
        临时文件的路径

    抛出:
        OSError: 写入失败时，自动清理临时文件后重新抛出
    """
    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp, target.stat().st_mode)
        return temp
    except Exception:
        temp.unlink(missing_ok=True)
        raise


# ── JSON Schema 定义 ──────────────────────────────────────────────────────────


def _ls_schema() -> dict[str, Any]:
    return {"$schema": _DRAFT, "type": "object", "properties": {"path": {"type": "string", "default": "."}, "max_entries": {"type": "integer", "minimum": 1, "maximum": 10_000, "default": 100}}, "additionalProperties": False}


def _read_schema() -> dict[str, Any]:
    return {"$schema": _DRAFT, "type": "object", "properties": {"path": {"type": "string"}, "max_chars": {"type": "integer", "minimum": 1, "default": 20_000}, "offset": {"type": "integer", "minimum": 1, "default": 1}, "limit": {"type": "integer", "minimum": 1, "default": 200}}, "required": ["path"], "additionalProperties": False}


def _write_schema() -> dict[str, Any]:
    return {"$schema": _DRAFT, "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string", "maxLength": 1_000_000}, "overwrite": {"type": "boolean", "default": True}}, "required": ["path", "content"], "additionalProperties": False}


def _edit_schema() -> dict[str, Any]:
    return {"$schema": _DRAFT, "type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string", "minLength": 1}, "new_text": {"type": "string"}, "replace_all": {"type": "boolean", "default": False}, "occurrence_index": {"type": ["integer", "null"], "minimum": 1}, "expected_occurrences": {"type": ["integer", "null"], "minimum": 0}, "expected_file_hash": {"type": ["string", "null"]}}, "required": ["path", "old_text", "new_text"], "additionalProperties": False}


def _patch_schema() -> dict[str, Any]:
    return {"$schema": _DRAFT, "type": "object", "properties": {"edits": {"type": "array", "minItems": 1, "maxItems": 20, "items": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string", "minLength": 1}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"], "additionalProperties": False}}}, "required": ["edits"], "additionalProperties": False}


__all__ = ["create_file_registrations"]
