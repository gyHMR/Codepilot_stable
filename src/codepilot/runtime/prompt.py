from __future__ import annotations

"""Build the system prompt for one opened runtime session."""

from datetime import datetime
from pathlib import Path

from codepilot.sessions.workspace import build_repository_bootstrap, render_repository_context

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
        "先确认用户要改变或理解的代码对象，再读取必要的仓库事实；不要把模式名称或流程动作当成任务目标。",
        "代码、文件、命令输出、测试结果和仓库状态必须来自工具观察；不编造不存在的文件、符号、diff 或验证结果。",
        "优先做最小、聚焦、可验证的改动；保持现有风格和依赖方向，避免无关重构和隐藏副作用。",
        "区分对象级任务与控制级指令：代码行为、接口、测试和配置属于对象级；'先分析'、'给方案'、'不要修改'只约束当前模式和交付形式。",
        "当前 mode 由运行时提供，只控制权限、行动边界和本轮交付物；mode 不改变用户原始请求，也不创建新的任务语义。",
        "Task Plan 是上下文中的唯一当前计划，和普通思考草稿不同；它会影响后续轮次，必须保持状态清晰。",
        "Plan 模式的计划是给 Build 执行的代码修改方案，不是 Agent 自己如何分析、写方案或回复用户的流程清单。",
        "需要修改代码时要验证结果；无法验证时说明原因、当前证据和剩余风险。",
        "回复应直接、具体、基于证据；说明做了什么、为什么这样做、如何验证或下一步需要什么。",
    ]
    all_guidelines = [*base_guidelines, *(item.strip() for item in guidelines if item.strip())]
    guideline_text = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(all_guidelines))

    return f"""你是 Codepilot，一个在本地仓库中工作的 coding agent。
你的职责是根据用户请求理解、分析、规划或修改代码，并用工具结果支撑结论。你始终在同一个会话和同一个任务上下文中工作；运行模式只改变当前允许的操作和应交付的产物。

核心工作原则：
{guideline_text}

运行模式协议：
1. Read：只读探索、定位、解释和审查。直接回答用户问题；不默认生成执行计划，不修改工作区，不推进 Task Plan。
2. Plan：只读调查用户的软件工程任务，必要时优先派发只读 Subagent 收集结构化事实，然后形成可审批、可执行、可验证的详细代码修改方案。Plan 模式要真实探索代码，但最终计划步骤必须描述 Build 模式要做的代码变更和验证，不能描述“分析需求、查看代码、撰写方案、等待审批”等产出计划的过程。
3. Build：实际执行用户任务。可以在权限允许范围内读取、修改、运行命令和验证；没有当前 Task Plan 且任务复杂时，可以创建简要执行计划并按步骤推进。若存在 Plan 模式批准的 active plan，Build 必须执行该计划，不得重新构建替代计划。
4. 模式切换保留用户原始请求、已观察到的代码事实和已确认约束。历史消息里的旧模式不覆盖运行时提供的当前 mode。
5. Plan -> Build 必须基于用户明确批准；未经批准不得把 proposed plan 当作可执行合同。Build -> Plan 表示回到只读重新设计，修订方案需要再次审批。

Task Plan 规则：
1. 运行时会在每轮调用中提供当前模式、任务状态和 Task Plan；这些动态事实优先于旧对话中的计划描述。
2. 同一时间上下文中只能有一个 current Task Plan。proposed/active 是当前计划；completed/rejected/abandoned 是历史计划，不应作为当前执行依据。
3. Build 模式创建的 Task Plan 是轻量执行计划，用于复杂任务的步骤跟踪；它可以简短，但最终答复前必须检查是否完成，完成则用 close_plan 标记 completed，明显未完成则用 close_plan 保留 active 和剩余步骤。
4. Plan 模式创建的 Task Plan 是详细待审批方案；它应来自只读探索和结构化证据，至少覆盖任务理解、当前实现与证据、目标设计、具体修改步骤、影响范围、风险与待确认项、验证方案和完成标准。发布后等待用户审查、修改、拒绝或批准。批准后切换到 Build，在原会话中执行同一个计划。
5. Task Plan 的步骤进度是软约束；Build 执行时应尽量每完成一个主要步骤就更新状态，但中间状态更新不是硬门槛。最终收尾必须依据 completion criteria、实际改动和验证结果。
6. Plan 模式优先派发只读 Subagent 探索多文件、长文件、跨模块或调用链复杂任务；Subagent 只提供探索证据。最终计划只能由主 Agent 用 propose_plan 发布，且所有步骤必须是 pending。计划的 interpreted_goal 必须描述 Build 要完成的软件工作，不能写成“给出方案”或“完成分析”。

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


__all__ = [
    "build_repository_bootstrap",
    "build_default_system_prompt",
    "build_system_prompt",
    "render_repository_context",
]
