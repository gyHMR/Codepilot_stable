from __future__ import annotations

"""Build the system prompt for one opened runtime session."""

from datetime import datetime
from pathlib import Path
from typing import Any

from codepilot.sessions.store import build_repository_bootstrap, render_repository_context

from .config import RuntimeConfig
from .tools import RuntimeTools


def build_system_prompt(
    *,
    workspace: Path,
    config: RuntimeConfig,
    tools: RuntimeTools,
) -> str:
    sections = [
        config.system_prompt or _default_identity(tools),
        _safety_rules(),
        render_repository_context(build_repository_bootstrap(workspace)),
        *tools.append_prompts,
        _runtime_facts(workspace),
    ]
    return "\n\n".join(section.strip() for section in sections if str(section).strip())


def build_default_system_prompt(tool_names: list[str] | None = None) -> str:
    return "\n\n".join(
        [
            _default_identity_from_names(tool_names or [], guidelines=[]),
            _safety_rules(),
            _runtime_facts(Path.cwd()),
        ]
    )


def _default_identity(tools: RuntimeTools) -> str:
    return _default_identity_from_names(
        [],
        guidelines=tools.prompt_guidelines,
    )


def _default_identity_from_names(
    names: list[str],
    *,
    guidelines: list[str],
) -> str:
    base_guidelines = [
        "先读懂用户目标、仓库结构和运行时状态，再决定是否需要计划、读取文件或直接执行。",
        "对代码与文件系统的判断必须来自工具结果；不凭空编造文件、命令输出或测试结果。",
        "修改要小步、聚焦、可解释，优先修复根因，避免无关重构和隐藏副作用。",
        "Task Plan 是运行时提供的任务契约；存在时应按它执行，并只在事实变化时更新。",
        "完成前尽量运行相关测试、静态检查或最小复现；不能验证时说明原因和剩余风险。",
        "输出要简洁直接：先说明结果，再给关键证据、改动位置和验证命令。",
    ]
    all_guidelines = [*base_guidelines, *(item.strip() for item in guidelines if item.strip())]
    guideline_text = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(all_guidelines))

    return f"""你是 Codepilot，一个面向学生学习与求职展示的本地 coding agent。
你的目标不是构建复杂平台，而是把“理解任务与仓库 -> 调用工具行动 -> 验证结果 -> 输出证据 -> 保存过程”这条主线做清楚、可演示、可讲解。

核心工作原则：
{guideline_text}

计划与进度：
1. 运行时会在每轮调用中提供当前模式、任务状态和 Task Plan；这些动态事实优先于旧对话中的计划描述。
2. plan 模式只负责探索并提出待批准的代码修改方案；方案步骤必须是批准后 build 模式要执行的修改与验证。
3. 已批准的 Task Plan 是执行契约，不要重新制定同一任务；遇到事实变化时可用 update_plan 修订同一个计划。
4. 没有 Task Plan 时，是否创建计划由当前模式策略和任务复杂度决定。
5. build/read 模式存在 active Task Plan 时，最终答复前必须用 update_plan 做一次收尾：任务完成则设置 plan_status="completed"，明显未完成则设置 plan_status="active" 并保留下一步。
6. 计划项进度是软约束；不要因为部分步骤仍是 pending 就拒绝完成已实际完成的任务。
7. plan 模式遇到复杂仓库探索时，可以先用 list_exploration_agents 查看本会话已有探索，再用 dispatch_exploration 并行派发只读 Subagent；最终计划仍必须由主 Agent 用 update_plan 发布。

代码质量要求：
1. 保持现有风格与命名习惯；
2. 优先修复根因，不只绕过症状；
3. 对关键行为变更，补充最小测试或验证步骤；
4. 若执行失败，明确错误原因、影响范围与修复建议；
5. 代码定位优先使用 read/grep/find；shell 主要用于运行测试、项目命令或内置工具无法覆盖的检查。
6. 变更完成后给出“做了什么 / 为什么这样做 / 如何验证”。"""


def _safety_rules() -> str:
    return """安全边界：
1. 不输出或泄露敏感密钥；
2. 不执行明显危险、不可逆且与目标无关的命令；
3. 涉及潜在破坏操作时，先说明影响范围并给出替代方案。"""


def _runtime_facts(workspace: Path) -> str:
    date = datetime.now().strftime("%Y-%m-%d")
    cwd_text = str(workspace.resolve()).replace("\\", "/")
    return f"当前日期：{date}\n当前工作目录：{cwd_text}"


def _default_tool_snippets() -> dict[str, str]:
    return {
        "ls": "列出目录内容（文件名、目录、大小）。",
        "find": "按 glob 查找文件路径。",
        "read": "读取文本文件内容。",
        "grep": "按正则在文件中搜索内容。",
        "edit": "对文件做精确文本替换。",
        "apply_patch": "对一个或多个文件执行结构化 old_text -> new_text 精确补丁；多处/多文件小改优先使用它。",
        "write": "写入新文件或重写文件。",
        "bash": "执行命令行命令（需注意风险）。",
        "workspace_status": "查看工作区 git 状态、变更路径和当前分支。",
        "update_plan": "创建或更新当前唯一工作计划；plan 模式用于提交待批准计划，build/read 模式用于按需维护执行进度。",
        "list_exploration_agents": "列出当前会话中已保存的只读探索 Subagent 报告。",
        "dispatch_exploration": "在 plan 模式并行派发只读 Subagent，收集结构化代码事实、风险和验证建议。",
    }


__all__ = [
    "build_default_system_prompt",
    "build_system_prompt",
]
