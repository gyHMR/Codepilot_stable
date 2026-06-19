from __future__ import annotations

"""
系统提示词渲染模块。

负责构建最终发送给 LLM 的系统提示词。

阶段B改进：
- 使用 PromptPlan 进行结构化段落组装
- 每个段落有明确的名称、来源和优先级
- 保持最终传入 Agent 的仍是字符串
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from codepilot.tools import AgentTool

from .context import RuntimeContext


@dataclass(frozen=True)
class PromptSection:
    """系统提示词中的一个有序段落。"""

    name: str
    content: str
    source: str
    priority: int


@dataclass
class PromptPlan:
    """结构化系统提示词计划。"""

    sections: list[PromptSection] = field(default_factory=list)

    def render(self) -> str:
        sections = sorted(self.sections, key=lambda section: section.priority)
        return "\n\n".join(
            section.content.strip()
            for section in sections
            if section.content.strip()
        )

    def get_sources(self) -> dict[str, str]:
        return {section.name: section.source for section in self.sections}


# ── 段落优先级常量 ────────────────────────────────────────────────

PRIORITY_IDENTITY = 10       # 身份定义
PRIORITY_RULES = 20          # 核心规则
PRIORITY_SAFETY = 30         # 安全策略
PRIORITY_WORKSPACE = 40      # 工作区指令
PRIORITY_REPOSITORY = 50     # 仓库上下文
PRIORITY_MEMORY = 60         # 长期记忆
PRIORITY_CAPABILITY = 70     # 能力说明
PRIORITY_EXTENSIONS = 80     # 扩展内容
PRIORITY_RUNTIME = 90        # 运行时事实


def build_runtime_system_prompt(
    *,
    base_system_prompt: str,
    tools: list[AgentTool],
    runtime_context: RuntimeContext,
    workspace: Path,
) -> str:
    """构建运行时系统提示词（工厂调用入口）。

    使用 PromptPlan 进行结构化段落组装。

    Args:
        base_system_prompt: 基础系统提示词（来自配置或会话恢复）。
        tools: 当前会话可用的工具列表。
        runtime_context: 运行时上下文（仓库信息、准则、记忆等）。
        workspace: 工作区目录路径。

    Returns:
        完整的系统提示词字符串。
    """
    plan = PromptPlan()

    # 1. 身份和核心规则
    if base_system_prompt:
        plan.sections.append(PromptSection(
            name="identity",
            content=base_system_prompt,
            source="config",
            priority=PRIORITY_IDENTITY,
        ))
    else:
        plan.sections.append(PromptSection(
            name="identity",
            content=_build_default_identity(tools, runtime_context),
            source="default",
            priority=PRIORITY_IDENTITY,
        ))

    # 2. 安全策略
    plan.sections.append(PromptSection(
        name="safety",
        content=_build_safety_section(),
        source="default",
        priority=PRIORITY_SAFETY,
    ))

    # 3. 仓库上下文
    if runtime_context.repository_context:
        plan.sections.append(PromptSection(
            name="repository",
            content=runtime_context.repository_context,
            source="workspace",
            priority=PRIORITY_REPOSITORY,
        ))

    # 4. 长期记忆
    if runtime_context.memory_text:
        plan.sections.append(PromptSection(
            name="memory",
            content=f"长期记忆（MEMORY）：\n{runtime_context.memory_text}",
            source="memory",
            priority=PRIORITY_MEMORY,
        ))

    # 5. 扩展内容
    for i, section in enumerate(runtime_context.append_sections or []):
        if section.strip():
            plan.sections.append(PromptSection(
                name=f"extension_{i}",
                content=section.strip(),
                source="extension",
                priority=PRIORITY_EXTENSIONS,
            ))

    # 6. 运行时事实
    date = datetime.now().strftime("%Y-%m-%d")
    cwd_text = str(workspace.resolve()).replace("\\", "/")
    plan.sections.append(PromptSection(
        name="runtime_facts",
        content=f"当前日期：{date}\n当前工作目录：{cwd_text}",
        source="runtime",
        priority=PRIORITY_RUNTIME,
    ))

    return plan.render()


def _build_default_identity(
    tools: list[AgentTool],
    runtime_context: RuntimeContext,
    *,
    tool_names: list[str] | None = None,
) -> str:
    """构建默认的身份和规则段落。"""
    tool_names = tool_names if tool_names is not None else _canonical_tool_names(tools)
    snippets = {**_default_tool_snippets(), **(runtime_context.tool_snippets or {})}
    visible_tools = [name for name in tool_names if name in snippets]
    tools_list = "\n".join([f"- {name}: {snippets[name]}" for name in visible_tools]) if visible_tools else "- （由运行时提供）"
    tools_text = "、".join(tool_names) if tool_names else "（由运行时提供）"

    # 合并准则
    guidelines = [
        "先理解目标与约束，再开始操作；需求不清时只提最小必要问题。",
        "对代码与文件系统的判断，优先基于工具结果，不凭空猜测。",
        "变更应小步、可验证、可回滚，优先修复根因而不是症状。",
        "涉及风险操作时先提示影响范围，再执行更安全替代方案。",
        "输出要简洁直接：先结论，再关键证据，再下一步。",
    ]
    if runtime_context.prompt_guidelines:
        guidelines.extend([g.strip() for g in runtime_context.prompt_guidelines if g.strip()])
    guidelines_text = "\n".join([f"{i + 1}. {g}" for i, g in enumerate(guidelines)])

    return f"""你是一个专业、可靠的编程助手。

工作原则（必须遵守）：
{guidelines_text}

可用工具（当前会话）：
- 工具名：{tools_text}
- 工具说明：
{tools_list}

工具使用规范：
1. 查目录优先 ls/find，查内容优先 read/grep；不要用 bash 代替常规读写工具。
2. 修改前先读文件并定位上下文，确认修改点后再 edit/write。
3. edit 只做精确替换；需要大段重构或新文件时再用 write。
4. 执行 bash 前先检查副作用，禁止与目标无关的破坏性命令。
5. 若可先做只读验证，就先只读验证，再执行写操作。

短任务探索规则：
1. 任务缺少具体文件、符号或错误信息时，先获取最小事实，不立即修改。
2. "修一下测试"：先识别测试入口并运行相关测试，取得真实失败。
3. "加个接口"：先搜索现有路由或相邻接口，读取项目已有写法。
4. "为什么挂了"：优先使用用户提供的错误、最近失败结果和会话历史；若"这里"没有明确指代，再询问最小必要信息。
5. 已有明确文件、符号或堆栈时，直接进行针对性搜索和读取。
6. 每次探索应逐步收窄范围，避免一次读取大量无关文件。

代码质量要求：
1. 保持现有风格与命名习惯；
2. 优先修复根因，不只绕过症状；
3. 对关键行为变更，补充最小测试或验证步骤；
4. 若执行失败，明确错误原因、影响范围与修复建议；
5. 变更完成后给出"做了什么 / 为什么这样做 / 如何验证"。"""


def _build_safety_section() -> str:
    """构建安全策略段落。"""
    return """安全边界：
1. 不输出或泄露敏感密钥；
2. 不执行明显危险、不可逆且与目标无关的命令；
3. 涉及潜在破坏操作时，先说明影响范围并给出替代方案。"""


def build_default_system_prompt(tool_names: list[str] | None = None) -> str:
    """构建默认系统提示词（便捷入口）。

    Args:
        tool_names: 可选的工具名称列表。

    Returns:
        默认系统提示词字符串。
    """
    plan = PromptPlan()
    plan.sections.append(PromptSection(
        name="identity",
        content=_build_default_identity(
            [],
            RuntimeContext(
                repository_context="",
                prompt_guidelines=[],
                append_sections=[],
                tool_snippets={},
                memory_text="",
            ),
            tool_names=tool_names or [],
        ),
        source="default",
        priority=PRIORITY_IDENTITY,
    ))
    plan.sections.append(PromptSection(
        name="safety",
        content=_build_safety_section(),
        source="default",
        priority=PRIORITY_SAFETY,
    ))
    date = datetime.now().strftime("%Y-%m-%d")
    cwd_text = str(Path.cwd().resolve()).replace("\\", "/")
    plan.sections.append(PromptSection(
        name="runtime_facts",
        content=f"当前日期：{date}\n当前工作目录：{cwd_text}",
        source="runtime",
        priority=PRIORITY_RUNTIME,
    ))
    return plan.render()


def _default_tool_snippets() -> dict[str, str]:
    """默认的工具说明片段（用于系统提示词中的工具说明部分）。"""
    return {
        "ls": "列出目录内容（文件名、目录、大小）。",
        "find": "按 glob 查找文件路径。",
        "read": "读取文本文件内容。",
        "grep": "按正则在文件中搜索内容。",
        "edit": "对文件做精确文本替换。",
        "write": "写入新文件或重写文件。",
        "bash": "执行命令行命令（需注意风险）。",
    }


def _canonical_tool_names(tools: list[AgentTool]) -> list[str]:
    """提取工具名称列表（去重并保持顺序）。"""
    names: list[str] = []
    seen: set[str] = set()
    for tool in tools:
        if tool.name not in seen:
            seen.add(tool.name)
            names.append(tool.name)
    return names
