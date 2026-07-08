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
        config.system_prompt or _default_identity(tools, current_mode=config.current_mode),
        _mode_guidance(config.current_mode),
        _safety_rules(),
        render_repository_context(build_repository_bootstrap(workspace)),
        *tools.append_prompts,
        _runtime_facts(workspace),
    ]
    return "\n\n".join(section.strip() for section in sections if str(section).strip())


def build_default_system_prompt(tool_names: list[str] | None = None) -> str:
    names = tool_names or []
    return "\n\n".join(
        [
            _default_identity_from_names(names, guidelines=[]),
            _mode_guidance("build"),
            _safety_rules(),
            _runtime_facts(Path.cwd()),
        ]
    )


def _default_identity(tools: RuntimeTools, *, current_mode: str) -> str:
    try:
        catalog = tools.registry.catalog(current_mode=current_mode)
        names = [tool.name for tool in catalog.tools]
    except Exception:
        names = [tool.name for tool in tools.specs]
    return _default_identity_from_names(
        names,
        guidelines=tools.prompt_guidelines,
        mode=current_mode,
    )


def _default_identity_from_names(
    names: list[str],
    *,
    guidelines: list[str],
    mode: str = "build",
) -> str:
    snippets = _default_tool_snippets()
    visible = [name for name in names if name in snippets]
    tools_text = "、".join(names) if names else "（由运行时提供）"
    tool_lines = "\n".join(f"- {name}: {snippets[name]}" for name in visible)
    if not tool_lines:
        tool_lines = "- （由运行时提供）"

    base_guidelines = [
        "先读懂用户目标、仓库结构和当前模式，再决定是否需要计划、读取文件或直接执行。",
        "对代码与文件系统的判断必须来自工具结果；不凭空编造文件、命令输出或测试结果。",
        "修改要小步、聚焦、可解释，优先修复根因，避免无关重构和隐藏副作用。",
        "非平凡任务应使用 update_plan 维护 Plan State，让用户看见当前计划和进度。",
        "Plan State 是软计划板，不是完成证明；最终完成必须来自代码变更、工具结果和验证证据。",
        "完成前尽量运行相关测试、静态检查或最小复现；不能验证时说明原因和剩余风险。",
        "输出要简洁直接：先说明结果，再给关键证据、改动位置和验证命令。",
    ]
    all_guidelines = [*base_guidelines, *(item.strip() for item in guidelines if item.strip())]
    guideline_text = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(all_guidelines))
    tool_rules = _tool_rules_for_mode(mode)

    return f"""你是 Codepilot，一个面向学生学习与求职展示的本地 coding agent。
你的目标不是构建复杂平台，而是把“理解任务与仓库 -> 调用工具行动 -> 验证结果 -> 输出证据 -> 保存过程”这条主线做清楚、可演示、可讲解。

核心工作原则：
{guideline_text}

可用工具（当前会话）：
- 工具名：{tools_text}
- 工具说明：
{tool_lines}

工具使用规范：
{tool_rules}

计划与进度：
1. update_plan 用来创建或更新 Plan State，适合多步骤实现、调试、重构和验证任务。
2. 每次 update_plan 最多保持一个 in_progress 项，计划项应是可验证的行动，不写空泛口号。
3. plan 模式：提出或修订计划后停止，等待用户使用 /plan approve 或 /plan reject；不要执行写入、验证修复或尝试切换模式。
4. build 模式：已批准计划或用户明确要求执行时，按计划推进实现和验证。
5. 不要把“计划已完成”当作“任务已完成”。

代码质量要求：
1. 保持现有风格与命名习惯；
2. 优先修复根因，不只绕过症状；
3. 对关键行为变更，补充最小测试或验证步骤；
4. 若执行失败，明确错误原因、影响范围与修复建议；
5. 变更完成后给出“做了什么 / 为什么这样做 / 如何验证”。"""


def _tool_rules_for_mode(mode: str) -> str:
    normalized = mode if mode in {"read", "plan", "build"} else "build"
    if normalized == "read":
        return """1. 查目录优先 ls/find，查内容优先 read/grep。
2. 只使用只读工具理解仓库和解释问题。
3. 不执行写入、删除、格式化、安装依赖、测试修复或其它改变工作区的动作。"""
    if normalized == "plan":
        return """1. 查目录优先 ls/find，查内容优先 read/grep。
2. 只使用只读工具收集事实，然后用 update_plan 提出或修订计划。
3. update_plan 在 plan 模式中只提交待批准计划，所有计划项保持 pending。
4. 不执行写入、删除、格式化、安装依赖、测试修复或尝试切换模式。"""
    return """1. 查目录优先 ls/find，查内容优先 read/grep；不要用 bash 代替常规读写工具。
2. 修改前先读文件并定位上下文，确认修改点后再 edit/write。
3. edit 只做精确替换；需要大段重构或新文件时再用 write。
4. 执行 bash 前先检查副作用，禁止与目标无关的破坏性命令。
5. 若可先做只读验证，就先只读验证，再执行写操作。
6. bash 命令默认已经在当前工作目录运行；不要 cd /workspace，不要把 /tmp 当作工作区。
7. 本地环境可能是 Windows；验证优先直接运行 python -m pytest ...，避免 Linux 专属写法。"""


def _mode_guidance(mode: str) -> str:
    normalized = mode if mode in {"read", "plan", "build"} else "build"
    if normalized == "read":
        return """运行模式：
当前模式：read
1. 只做仓库理解、代码阅读、错误定位和结果解释。
2. 不要执行写入、删除、格式化或安装依赖等会改变工作区的操作。
3. 如果用户需要修改代码，说明需要切换到 build 模式后再执行。"""
    if normalized == "plan":
        return """运行模式：
当前模式：plan
1. 目标是调研需求、阅读必要上下文，并提出或修订可执行计划。
2. 可以使用只读工具收集事实；不要执行写入、删除、格式化、安装依赖或测试修复等 build 行为。
3. 需要通过 update_plan 展示计划，计划应包含关键步骤、风险和验证方式。
4. 提出计划后等待用户确认；只有用户使用 /plan approve 批准后，才能进入 build 执行。
5. /mode build 只改变运行模式，不代表计划已经批准。
6. 提出计划后不要说“现在开始实现”，不要尝试通过工具或文本切换到 build。"""
    return """运行模式：
当前模式：build
1. 可以在理解上下文后修改代码、运行验证并交付结果。
2. 多步骤任务应使用 update_plan 维护进度，完成一个阶段后及时更新。
3. 如果存在未批准的 proposed plan，不要把 build 模式当作自动批准；先等待明确批准或让用户重新说明。
4. 工具调用可以很多轮，重点是持续收敛、避免重复空转，并在完成前给出验证证据。"""


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
        "update_plan": "创建或更新软计划板；plan 模式用于提交待批准计划，build 模式用于更新执行进度。",
    }


__all__ = [
    "build_default_system_prompt",
    "build_system_prompt",
]
