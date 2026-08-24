"""规范的工作区状态工具 —— workspace_status。

本文件实现了一个简单的工作区状态查询工具，用于向 LLM 提供
工作区根目录的基本信息：目录中的条目数、是否为 Git 仓库等。

该工具是只读的，在 plan 和 execute 模式下均可使用，
不需要审批，可以并行执行。
"""

from dataclasses import dataclass
from typing import Any

from ..codecs import DataclassCodec
from ..contracts import ToolExecutionContext, ToolRegistration, ToolSpec
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


# ── 输入输出类型 ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WorkspaceStatusInput:
    """workspace_status 工具的输入参数。

    参数:
        include_hidden: 是否包含隐藏文件（以 "." 开头的文件），默认 False
    """
    include_hidden: bool = False


@dataclass(frozen=True)
class WorkspaceStatusOutput:
    """workspace_status 工具的输出类型。

    参数:
        text: 给 LLM 看的文本摘要
        details: 结构化详情（工作区路径、条目数、条目列表、Git 状态）
        metadata: 元数据
    """
    text: str
    details: dict[str, Any]
    metadata: dict[str, Any]


# ── 注册创建函数 ──────────────────────────────────────────────────────────────


def create_workspace_status_registration(sandbox: WorkspaceSandbox) -> ToolRegistration:
    """创建工作区状态工具注册。

    该工具用于查询工作区根目录的基本信息：
    - 工作区路径
    - 可见条目数（可包含隐藏文件）
    - 前 100 个条目名称
    - 是否为 Git 仓库

    参数:
        sandbox: 工作区沙箱

    返回:
        workspace_status 工具的 ToolRegistration
    """
    input_schema = {
        "$schema": _DRAFT,
        "type": "object",
        "properties": {"include_hidden": {"type": "boolean", "default": False}},
        "additionalProperties": False,
    }
    output_schema = {
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

    class Resolver:
        """工作区状态工具的访问解析器。"""
        def resolve(self, input, request):
            _ = request
            return ToolAccessResolution(
                input=input,
                access=ToolAccessRequest(
                    actions=("workspace_status",),
                    resources=(ToolResource("workspace:///"),),
                    effects=frozenset({"filesystem_read"}),
                    risk="low",
                    reason="Inspect workspace root status",
                ),
            )

    async def handler(input: WorkspaceStatusInput, context: ToolExecutionContext):
        """workspace_status 处理器 —— 列出工作区根目录内容。

        处理流程:
        1. 检查取消信号
        2. 列出工作区根目录的所有条目
        3. 根据 include_hidden 过滤隐藏文件
        4. 报告副作用
        5. 返回文本摘要和结构化详情

        参数:
            input: WorkspaceStatusInput
            context: 执行上下文

        返回:
            WorkspaceStatusOutput 包含工作区状态信息
        """
        context.cancellation.raise_if_cancelled()
        entries = sorted(
            item.name
            for item in sandbox.root.iterdir()
            if input.include_hidden or not item.name.startswith(".")
        )
        details = {
            "workspace": str(sandbox.root),
            "entry_count": len(entries),
            "entries": entries[:100],
            "is_git_repository": (sandbox.root / ".git").exists(),
        }
        context.effects.report(
            ToolEffect(
                kind="filesystem_read",
                resource=ToolResource("workspace:///"),
                operation="inspect workspace status",
                status="completed",
                certainty="observed",
            )
        )
        return WorkspaceStatusOutput(
            text=(
                f"Workspace: {sandbox.root}\n"
                f"Entries: {len(entries)}\n"
                f"Git repository: {details['is_git_repository']}"
            ),
            details=details,
            metadata={},
        )

    class Renderer:
        def render(self, data):
            return (TextContent(text=data["text"]),)

    return ToolRegistration(
        version="1.0.0",
        implementation_version="2",
        spec=ToolSpec(
            "workspace_status",
            "Return a bounded orientation snapshot of the workspace root, visible top-level entries, and Git repository presence. Use once when the repository shape is unknown; use ls, find, grep, and read for targeted follow-up rather than repeatedly requesting the same snapshot.",
            input_schema,
            output_schema,
        ),
        category="filesystem",
        source="builtin",
        owner="codepilot.builtin",
        policy=ToolPolicy(
            allowed_modes=frozenset({"plan", "execute"}),
            declared_effects=frozenset({"filesystem_read"}),
            required_permissions=frozenset({"workspace.read"}),
            base_risk="low",
            approval="never",
            timeout=TimeoutPolicy(5_000, 10_000),
            concurrency=ConcurrencyPolicy(mode="parallel"),
            output_limits=OutputLimits(),
            output_trust=OutputTrustPolicy(),
        ),
        input_codec=DataclassCodec(WorkspaceStatusInput, input_schema),
        output_codec=DataclassCodec(WorkspaceStatusOutput, output_schema),
        handler=handler,
        renderer=Renderer(),
        access_resolver=Resolver(),
    )


__all__ = ["create_workspace_status_registration"]
