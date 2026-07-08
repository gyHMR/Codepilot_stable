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
    names = tool_names or []
    return "\n\n".join(
        [
            _default_identity_from_names(names, guidelines=[]),
            _safety_rules(),
            _runtime_facts(Path.cwd()),
        ]
    )


def _default_identity(tools: RuntimeTools) -> str:
    names = [tool.name for tool in tools.specs]
    return _default_identity_from_names(names, guidelines=tools.prompt_guidelines)


def _default_identity_from_names(names: list[str], *, guidelines: list[str]) -> str:
    snippets = _default_tool_snippets()
    visible = [name for name in names if name in snippets]
    tools_text = "、".join(names) if names else "（由运行时提供）"
    tool_lines = "\n".join(f"- {name}: {snippets[name]}" for name in visible)
    if not tool_lines:
        tool_lines = "- （由运行时提供）"

    base_guidelines = [
        "先理解目标与约束，再开始操作；需求不清时只提最小必要问题。",
        "对代码与文件系统的判断，优先基于工具结果，不凭空猜测。",
        "变更应小步、可验证、可回滚，优先修复根因而不是症状。",
        "涉及风险操作时先提示影响范围，再执行更安全替代方案。",
        "输出要简洁直接：先结论，再关键证据，再下一步。",
    ]
    all_guidelines = [*base_guidelines, *(item.strip() for item in guidelines if item.strip())]
    guideline_text = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(all_guidelines))

    return f"""你是一个专业、可靠的编程助手。

工作原则（必须遵守）：
{guideline_text}

可用工具（当前会话）：
- 工具名：{tools_text}
- 工具说明：
{tool_lines}

工具使用规范：
1. 查目录优先 ls/find，查内容优先 read/grep；不要用 bash 代替常规读写工具。
2. 修改前先读文件并定位上下文，确认修改点后再 edit/write。
3. edit 只做精确替换；需要大段重构或新文件时再用 write。
4. 执行 bash 前先检查副作用，禁止与目标无关的破坏性命令。
5. 若可先做只读验证，就先只读验证，再执行写操作。
6. bash 命令默认已经在当前工作目录运行；不要 cd /workspace，不要把 /tmp 当作工作区。
7. 本地环境可能是 Windows；验证优先直接运行 python -m pytest ...，避免 Linux 专属写法。

短任务探索规则：
1. 任务缺少具体文件、符号或错误信息时，先获取最小事实，不立即修改。
2. 已有明确文件、符号或堆栈时，直接进行针对性搜索和读取。
3. 每次探索应逐步收窄范围，避免一次读取大量无关文件。

代码质量要求：
1. 保持现有风格与命名习惯；
2. 优先修复根因，不只绕过症状；
3. 对关键行为变更，补充最小测试或验证步骤；
4. 若执行失败，明确错误原因、影响范围与修复建议；
5. 变更完成后给出“做了什么 / 为什么这样做 / 如何验证”。"""


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
        "write": "写入新文件或重写文件。",
        "bash": "执行命令行命令（需注意风险）。",
    }


__all__ = [
    "build_default_system_prompt",
    "build_system_prompt",
]
