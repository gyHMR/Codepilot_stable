from __future__ import annotations

# 新手导读：runtime 组装主入口：把配置、模型、工具、扩展、prompt 和 session 选项串成 RuntimeAssembly。
# 关注点：想理解项目启动流程，先从 create_agent_session() 的步骤读起。

"""
Runtime 装配模块。

负责把上层友好的 CreateAgentSessionOptions 装配成一次会话创建所需的
RuntimeAssembly 和 AgentSession。

装配流程：
1) 加载运行时输入（工作区资源、会话恢复元数据）
2) 解析模型配置（从 options / 会话元数据 / 工作区配置中确定模型）
3) 解析运行时配置（多源合并）
4) 组装工具列表（内置工具 + 扩展工具 + MCP 工具）
5) 构建运行时上下文（仓库信息、提示词准则等）
6) 构建系统提示词
7) 组装钩子（before/after tool call、生命周期钩子）
8) 构造最终的 AgentSessionOptions 并创建 AgentSession
9) 保存装配产物（RuntimeAssembly）
"""

from dataclasses import replace
import os

from codepilot.core.message_conversion import convert_to_llm
from codepilot.llm.env_api_keys import get_env_api_key_name
from codepilot.sessions.session import AgentSession

from .bootstrap.config import RuntimeInputs, load_runtime_inputs, resolve_runtime_config
from .bootstrap.context import build_runtime_context, build_repository_bootstrap
from .bootstrap.hook_pipeline import compose_after_tool_call, compose_before_tool_call, compose_lifecycle_hooks
from .bootstrap.model_resolver import ResolvedModel, resolve_model
from .bootstrap.prompt import build_runtime_system_prompt
from .bootstrap.tool_assembler import assemble_tools
from .contracts import (
    AgentSessionOptions,
    CapabilityCatalog,
    ConfigValueSource,
    CreateAgentSessionOptions,
    ResolvedConfigValue,
    ResolvedRuntimeProfile,
    RuntimeAssembly,
    RuntimeDiagnostic,
)


class UnknownRuntimeConfigKeyError(KeyError):
    """请求解释的配置项不属于 Runtime 配置。"""


def create_agent_session(options: CreateAgentSessionOptions) -> AgentSession:
    """创建一个完整装配的 AgentSession。"""

    session, _ = assemble_runtime(options)
    return session


def assemble_runtime(options: CreateAgentSessionOptions) -> tuple[AgentSession, RuntimeAssembly]:
    """完整装配流程，返回 AgentSession 和 RuntimeAssembly。

    装配流程：
    1. load_runtime_inputs: 加载工作区资源和会话恢复元数据。
    2. resolve_model: 解析模型配置。
    3. resolve_runtime_config: 多源合并运行时配置。
    4. assemble_tools: 组装工具列表（内置 + 扩展 + MCP）。
    5. build_runtime_context: 构建运行时上下文。
    6. build_runtime_system_prompt: 构建系统提示词。
    7. compose_*: 组装钩子函数。
    8. 构建 RuntimeAssembly。

    Args:
        options: 友好的创建会话选项。

    Returns:
        (AgentSession, RuntimeAssembly) 元组。
    """
    diagnostics: list[RuntimeDiagnostic] = []

    # 步骤 1：加载运行时输入数据
    inputs = load_runtime_inputs(options)

    # 步骤 2：解析模型
    resolved_model = resolve_model(options, inputs)

    # 确定凭证来源
    credential_source, credential_location = _resolve_credential_source(
        options,
        inputs,
        resolved_model,
    )

    # 步骤 3：解析运行时配置
    config = resolve_runtime_config(options, inputs)

    # 构建配置来源
    sources = _build_config_sources(config.sources)
    sources.update({
        "model": resolved_model.source,
        "provider": resolved_model.source,
        "model_id": resolved_model.source,
    })

    # 构建 ResolvedRuntimeProfile
    profile = ResolvedRuntimeProfile(
        model=resolved_model.model,
        credential_source=credential_source,
        credential_location=credential_location,
        permission_mode=config.tool_permission_mode,  # type: ignore[arg-type]
        task_mode=config.task_mode,
        planning_budget_profile=config.planning_budget_profile,
        sources=sources,
    )

    # 步骤 4：组装工具
    assembled_tools = assemble_tools(inputs.workspace, options, config)

    # 收集工具装配诊断
    for diag in assembled_tools.diagnostics:
        diagnostics.append(diag)

    # 步骤 5：构建运行时上下文
    runtime_context = build_runtime_context(
        inputs.workspace,
        config,
        assembled_tools.loaded_extensions,
        assembled_tools.loaded_skills,
    )

    # 步骤 6：构建系统提示词
    system_prompt = build_runtime_system_prompt(
        base_system_prompt=config.system_prompt,
        tools=assembled_tools.tools,
        runtime_context=runtime_context,
        workspace=inputs.workspace,
    )

    # 步骤 7：组装钩子函数（合并调用方钩子和扩展钩子）
    before_tool_call = compose_before_tool_call(
        options.before_tool_call,
        assembled_tools.loaded_extensions.before_tool_hooks,
    )
    after_tool_call = compose_after_tool_call(
        options.after_tool_call,
        assembled_tools.loaded_extensions.after_tool_hooks,
    )
    before_prompt_hooks = compose_lifecycle_hooks(
        options.before_prompt_hooks,
        assembled_tools.loaded_extensions.before_prompt_hooks,
    )
    after_prompt_hooks = compose_lifecycle_hooks(
        options.after_prompt_hooks,
        assembled_tools.loaded_extensions.after_prompt_hooks,
    )

    # 步骤 8：构造最终的 AgentSessionOptions
    session_options = AgentSessionOptions(
        model=resolved_model.model,
        workspace_dir=inputs.workspace,
        system_prompt=system_prompt,
        tools=assembled_tools.tools,
        session_id=options.session_id,
        messages=options.messages,
        thinking_level=config.thinking_level,
        tool_execution=config.tool_execution,
        max_tool_calls_per_turn=config.max_tool_calls_per_turn,
        memory_enabled=options.memory_enabled,
        task_control_enabled=options.task_control_enabled,
        task_mode=config.task_mode,
        planning_budget_profile=config.planning_budget_profile,
        max_task_replans_per_run=options.max_task_replans_per_run or 2,
        convert_to_llm=convert_to_llm,
        get_api_key=resolved_model.get_api_key,
        retry_enabled=config.retry_enabled,
        max_retries=config.max_retries,
        retry_base_delay_ms=config.retry_base_delay_ms,
        extension_commands={
            **options.extension_commands,
            **assembled_tools.loaded_skills.commands,
            **assembled_tools.loaded_extensions.commands,
        },
        before_prompt_hooks=before_prompt_hooks,
        after_prompt_hooks=after_prompt_hooks,
        before_tool_call=before_tool_call,
        after_tool_call=after_tool_call,
        stream_fn=options.stream_fn,
        prepare_context=None,
    )

    # 步骤 9：构建能力目录和装配产物
    capability_catalog = CapabilityCatalog(
        tools=assembled_tools.registered_tools,
        commands={
            **assembled_tools.loaded_skills.commands,
            **assembled_tools.loaded_extensions.commands,
        },
    )
    session = AgentSession(session_options)
    effective_session_options = replace(
        session_options,
        session_id=session.session_id,
    )
    assembly = RuntimeAssembly(
        session_options=effective_session_options,
        profile=profile,
        repository=build_repository_bootstrap(inputs.workspace),
        capabilities=capability_catalog,
        tool_runtime=assembled_tools.tool_runtime,
        diagnostics=diagnostics,
    )

    return session, assembly


def _resolve_credential_source(
    options: CreateAgentSessionOptions,
    inputs: RuntimeInputs,
    resolved_model: ResolvedModel,
) -> tuple[str, str | None]:
    """解析凭证来源。

    Returns:
        (凭证来源, 凭证位置) 元组。
    """
    resources = inputs.resources

    if options.get_api_key:
        return "caller", "get_api_key function"

    if (
        resources
        and resources.model
        and resolved_model.source.location == ".codepilot/model.local.json"
    ):
        if resources.model.api_key_env:
            if os.getenv(resources.model.api_key_env):
                return "env", resources.model.api_key_env
        if resources.model.api_key:
            return "local-file", ".codepilot/model.local.json"

    env_name = get_env_api_key_name(resolved_model.model.provider)
    if env_name and os.getenv(env_name):
        return "env", env_name
    if resolved_model.model.api == "openai-compatible":
        fallback_env_name = get_env_api_key_name("openai")
        if fallback_env_name and os.getenv(fallback_env_name):
            return "env", fallback_env_name

    return "missing", None


def explain_runtime_config(
    options: CreateAgentSessionOptions,
    key: str,
) -> ResolvedConfigValue:
    """解析单个配置项的最终值和来源，供 CLI 等接口展示。"""

    inputs = load_runtime_inputs(options)
    if key in {"model", "provider", "model_id"}:
        resolved_model = resolve_model(options, inputs)
        model = resolved_model.model
        values = {
            "model": f"{model.provider}/{model.id}" if model.provider else model.id,
            "provider": model.provider,
            "model_id": model.id,
        }
        return ResolvedConfigValue(
            key=key,
            value=values[key],
            source=resolved_model.source,
        )

    config = resolve_runtime_config(options, inputs)
    if key not in config.sources:
        raise UnknownRuntimeConfigKeyError(key)
    return ResolvedConfigValue(
        key=key,
        value=getattr(config, key),
        source=_build_config_sources(config.sources)[key],
    )


def _build_config_sources(raw_sources: dict[str, str]) -> dict[str, ConfigValueSource]:
    """构建配置来源映射。"""
    source_map = {
        "options": ConfigValueSource(kind="cli"),
        "restored_session": ConfigValueSource(kind="session"),
        "workspace": ConfigValueSource(kind="project", location=".codepilot/settings.json"),
        "default": ConfigValueSource(kind="default"),
    }
    sources: dict[str, ConfigValueSource] = {}
    for key, value in raw_sources.items():
        try:
            sources[key] = source_map[value]
        except KeyError as exc:
            raise ValueError(f"Unknown runtime config source: {value}") from exc
    return sources
