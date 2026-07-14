"""按配置装配 Provider、Tools、MCP、Skill 与系统提示词。"""

from __future__ import annotations

"""Open one runtime session by wiring config, model, tools, prompt, and ports."""

from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from codepilot.core.tool_adapters import (
    create_interaction_registration,
    create_plan_registrations,
)
from codepilot.llm.adapter import ProviderModelPort
from codepilot.llm.registry import builtin_api_provider_registry
from codepilot.extensions import load_extensions, load_skills
from codepilot.extensions.mcp import MCPManager, create_mcp_manager
from codepilot.protocols.commands import RegisteredCommand
from codepilot.sessions.contracts import SessionOptions
from codepilot.sessions.workspace import build_repository_bootstrap, render_repository_context
from codepilot.tools import (
    CheckpointToolStateStore,
    FileToolGrantStore,
    PermissionEngine,
    PermissionRule,
    ToolRuntime,
)
from codepilot.tools.builtins import create_builtin_registrations
from codepilot.tools.contracts import ToolRegistration
from codepilot.tools.registry import ToolRegistry
from codepilot.tools.sandbox import ShellExecutionPolicy

from .actions import SessionOpenIntent
from .config import load_runtime_config
from .coordinator import create_session_controller
from .model import convert_to_llm, resolve_runtime_model
from .registry import RuntimeSession, RuntimeStatusInfo
from .subagents.tools import create_subagent_registrations


@dataclass
class RuntimeTools:
    """Runtime 装配后的工具注册表、执行器和扩展能力集合。"""
    registry: ToolRegistry
    mcp_manager: MCPManager | None = None
    commands: dict[str, RegisteredCommand] = field(default_factory=dict)
    before_prompt_hooks: list[Any] = field(default_factory=list)
    after_prompt_hooks: list[Any] = field(default_factory=list)
    prompt_guidelines: list[str] = field(default_factory=list)
    append_prompts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def build_runtime_tools(
    workspace: Path,
    intent: SessionOpenIntent,
    config: Any,
) -> RuntimeTools:
    warnings: list[str] = []
    loaded_extensions = load_extensions(workspace, configured_paths=config.extension_paths)
    loaded_skills = load_skills(workspace, configured_paths=config.skill_paths)
    warnings.extend(f"extension: {error}" for error in loaded_extensions.errors)
    warnings.extend(f"skill: {error}" for error in loaded_skills.errors)
    warnings.extend(f"skill: {item}" for item in loaded_skills.diagnostics)

    mcp_manager = create_mcp_manager(
        config.mcp_servers,
        transport_factory=intent.mcp_transport_factory,
    )
    builtin_tools = create_builtin_registrations(
        workspace,
        enabled_names=config.enabled_builtin_tools,
        edit_require_unique_match=config.edit_require_unique_match,
        shell_policy=ShellExecutionPolicy(
            timeout_seconds=config.shell_timeout_seconds,
            max_timeout_seconds=config.shell_max_timeout_seconds,
            stdout_limit=config.shell_stdout_limit,
            stderr_limit=config.shell_stderr_limit,
            allowed_env=tuple(config.shell_allowed_env or ()),
        ),
    )

    registry = ToolRegistry()
    registry.reserve(
        {
            "propose_plan",
            "create_build_plan",
            "update_plan_progress",
            "close_plan",
            "request_user_input",
            "list_exploration_agents",
            "dispatch_exploration",
        }
    )
    for registrations, source in (
        (builtin_tools, "builtin"),
        (list(intent.tools), "caller"),
        (loaded_skills.tools, "skill"),
        (loaded_extensions.tools, "extension"),
    ):
        _register_registration_groups(
            registry,
            registrations,
            source=source,
            warnings=warnings,
        )
    _validate_skill_requirements(
        loaded_skills.skills,
        registry=registry,
        configured_mcp_servers={
            str(item.get("name") or "").strip()
            for item in config.mcp_servers or []
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        },
        warnings=warnings,
    )
    return RuntimeTools(
        registry=registry,
        mcp_manager=mcp_manager if mcp_manager.configured else None,
        commands={**loaded_skills.commands, **loaded_extensions.commands},
        before_prompt_hooks=[*loaded_extensions.before_prompt_hooks, *loaded_skills.before_prompt_hooks],
        after_prompt_hooks=[*loaded_extensions.after_prompt_hooks, *loaded_skills.after_prompt_hooks],
        prompt_guidelines=[
            *(config.prompt_guidelines or []),
            *loaded_extensions.prompt_guidelines,
            *loaded_skills.prompt_guidelines,
        ],
        append_prompts=[
            *([config.append_system_prompt] if config.append_system_prompt else []),
            *loaded_extensions.append_prompts,
            *loaded_skills.append_prompts,
            *(
                _debug_prompt_sources(loaded_extensions, loaded_skills)
                if config.prompt_debug_sources
                else []
            ),
        ],
        warnings=warnings,
    )


def build_runtime_session(intent: SessionOpenIntent) -> RuntimeSession:
    """Build a runnable session from the public open-session intent."""

    provider_registry = builtin_api_provider_registry()

    config = load_runtime_config(intent)
    model = resolve_runtime_model(intent, config)
    runtime_model = model.model
    if intent.model_context_window is not None or intent.model_max_output_tokens is not None:
        runtime_model = replace(
            runtime_model,
            context_window=int(intent.model_context_window or runtime_model.context_window),
            max_tokens=int(intent.model_max_output_tokens or runtime_model.max_tokens),
        )
    tools = build_runtime_tools(config.workspace, intent, config)

    system_prompt = build_system_prompt(
        workspace=config.workspace,
        config=config,
        tools=tools,
    )

    def system_prompt_for(_mode: str) -> str:
        return system_prompt

    before_prompt_hooks = [*intent.before_prompt_hooks, *tools.before_prompt_hooks]
    after_prompt_hooks = [*intent.after_prompt_hooks, *tools.after_prompt_hooks]

    session_options = SessionOptions(
        model=runtime_model,
        workspace_dir=config.workspace,
        system_prompt=system_prompt,
        system_prompt_builder=system_prompt_for,
        session_id=intent.session_id,
        messages=list(intent.messages),
        thinking_level=config.thinking_level,
        max_tool_calls_per_turn=config.max_tool_calls_per_turn,
        memory_enabled=intent.memory_enabled,
        current_mode=config.current_mode,
        planning_budget_profile=config.planning_budget_profile,
        convert_to_llm=convert_to_llm,
        get_api_key=model.get_api_key,
        retry_enabled=config.retry_enabled,
        max_retries=config.max_retries,
        retry_base_delay_ms=config.retry_base_delay_ms,
        run_timeout_seconds=int(intent.run_timeout_seconds)
        if intent.run_timeout_seconds is not None
        else None,
        extension_commands={
            **intent.extension_commands,
            **tools.commands,
        },
        before_prompt_hooks=before_prompt_hooks,
        after_prompt_hooks=after_prompt_hooks,
        stream_fn=intent.stream_fn,
    )

    controller = create_session_controller(session_options)
    effective_options = replace(session_options, session_id=controller.session_id)
    model_port = ProviderModelPort(
        model=effective_options.model,
        stream_fn=effective_options.stream_fn,
        convert_messages=effective_options.convert_to_llm,
        get_api_key=effective_options.get_api_key,
        registry=provider_registry,
    )
    tool_state_store = CheckpointToolStateStore(
        session_id=controller.session_id,
        grant_store=FileToolGrantStore(config.workspace / ".codepilot" / "tool_grants.json"),
    )
    tool_port = ToolRuntime(
        registry=tools.registry,
        permission_engine=_permission_engine(config.tool_permission_mode),
        state_store=tool_state_store,
    )
    tool_checkpoint = controller.component_checkpoint_state("tools")
    if tool_checkpoint is not None:
        tool_port.restore_checkpoint_state(tool_checkpoint)

    runtime_session = RuntimeSession(
        controller=controller,
        model_port=model_port,
        tool_port=tool_port,
        status=RuntimeStatusInfo(
            session_id=controller.session_id,
            model_id=model.display_name,
            workspace=str(config.workspace),
            permission_mode=config.tool_permission_mode,
            credential_source=model.credential_source,
            warnings=tuple(tools.warnings),
        ),
        commands={
            **intent.extension_commands,
            **tools.commands,
        },
        mcp_manager=tools.mcp_manager,
    )
    tools.registry.extend(create_plan_registrations())
    tools.registry.register(create_interaction_registration())
    tools.registry.extend(
        create_subagent_registrations(
            workspace=config.workspace,
            session_provider=lambda: runtime_session,
        )
    )
    return runtime_session


def build_system_prompt(
    *,
    workspace: Path,
    config: Any,
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
    return _default_identity_from_names([], guidelines=tools.prompt_guidelines)


def _default_identity_from_names(
    names: list[str],
    *,
    guidelines: list[str],
) -> str:
    del names
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
    guideline_text = "\n".join(
        f"{index + 1}. {item}" for index, item in enumerate(all_guidelines)
    )
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


def _register_registration_groups(
    registry: ToolRegistry,
    registrations: list[ToolRegistration],
    *,
    source: str,
    warnings: list[str],
) -> None:
    groups: dict[str, list[ToolRegistration]] = {}
    for registration in registrations:
        if not isinstance(registration, ToolRegistration):
            raise TypeError(
                f"{source} tools must be ToolRegistration values, got "
                f"{type(registration).__name__}"
            )
        if registration.source != source:
            raise ValueError(
                f"{source} tool '{registration.spec.name}' declares source "
                f"'{registration.source}'"
            )
        groups.setdefault(registration.owner, []).append(registration)

    for owner, items in groups.items():
        if source == "builtin":
            registry.register_batch(items, owner=owner)
            continue
        for item in items:
            try:
                registry.register(item)
            except Exception as exc:
                warnings.append(
                    f"{source} owner '{owner}' tool '{item.spec.name}' registration failed: {exc}"
                )


def _debug_prompt_sources(extensions: Any, skills: Any) -> list[str]:
    lines: list[str] = ["## Prompt Sources", "### extensions"]
    lines.extend([f"- {path}" for path in extensions.loaded_paths] or ["- (none)"])
    lines.append("### skills")
    lines.extend([f"- {path}" for path in skills.loaded_paths] or ["- (none)"])
    errors = [*extensions.errors, *skills.errors]
    if errors:
        lines.append("### errors")
        lines.extend(f"- {error}" for error in errors)
    if skills.diagnostics:
        lines.append("### diagnostics")
        lines.extend(f"- {item}" for item in skills.diagnostics)
    return ["\n".join(lines)]


def _validate_skill_requirements(
    skills: list[Any],
    *,
    registry: ToolRegistry,
    configured_mcp_servers: set[str],
    warnings: list[str],
) -> None:
    available_tools = {entry.spec.name for entry in registry.catalog_snapshot().entries}
    for package in skills:
        manifest = package.manifest
        missing_tools = sorted(set(manifest.required_tools) - available_tools)
        missing_mcp = sorted(set(manifest.required_mcp) - configured_mcp_servers)
        if missing_tools:
            warnings.append(
                f"skill '{manifest.name}' requires unavailable tools: "
                + ", ".join(missing_tools)
            )
        if missing_mcp:
            warnings.append(
                f"skill '{manifest.name}' requires unconfigured MCP servers: "
                + ", ".join(missing_mcp)
            )


def _permission_engine(mode: str) -> PermissionEngine:
    mutating_effects = frozenset(
        {
            "filesystem_write",
            "filesystem_delete",
            "process_spawn",
            "external_state_write",
        }
    )
    sensitive_read_effects = frozenset(
        {"network_access", "credential_access", "external_state_read"}
    )
    if mode == "read-only":
        return PermissionEngine(
            denied_effects=mutating_effects | frozenset({"credential_access"}),
            approval_effects=frozenset({"network_access", "external_state_read"}),
            granted_permissions=frozenset(
                {"workspace.read", "session.*", "mcp.call"}
            ),
        )
    if mode == "ask":
        return PermissionEngine(
            approval_effects=mutating_effects | sensitive_read_effects,
            granted_permissions=frozenset({"*"}),
        )
    workspace_capabilities = (
        "write",
        "edit",
        "apply_patch",
        "command.inspection",
        "command.bounded_mutation",
    )
    return PermissionEngine(
        approval_effects=sensitive_read_effects,
        granted_permissions=frozenset({"*"}),
        rules=tuple(
            PermissionRule(
                action_pattern=action,
                resource_pattern="workspace:///*",
                effect="allow",
                modes=frozenset({"execute"}),
                source="runtime",
                priority=100,
            )
            for action in workspace_capabilities
        )
    )


__all__ = [
    "build_default_system_prompt",
    "build_runtime_session",
    "build_runtime_tools",
    "build_system_prompt",
    "RuntimeTools",
]
