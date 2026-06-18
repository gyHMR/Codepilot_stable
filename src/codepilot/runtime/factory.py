from __future__ import annotations

"""
coding_agent 工厂入口。
"""

from codepilot.sessions.session import AgentSession

from .config_loader import load_workspace_resources, read_restored_session_meta, resolve_runtime_config
from .context_builder import build_runtime_context_sources
from .convert_to_llm import convert_to_llm
from .hook_pipeline import compose_after_tool_call, compose_before_tool_call, compose_lifecycle_hooks
from .model_resolver import resolve_model
from .prompt_builder import build_runtime_system_prompt
from .tool_assembler import assemble_tools
from .types import AgentSessionOptions, CreateAgentSessionOptions


def create_agent_session(options: AgentSessionOptions | CreateAgentSessionOptions) -> AgentSession:
    """
    创建 AgentSession，支持两种输入形式：
    - AgentSessionOptions: 低层参数，model 必填；
    - CreateAgentSessionOptions: 更友好，支持 provider/model_id 或 session 恢复。
    """

    if isinstance(options, AgentSessionOptions):
        return AgentSession(options)

    workspace, resources = load_workspace_resources(options)
    restored_meta = read_restored_session_meta(workspace, options.session_id)
    resolved_model = resolve_model(options, resources, restored_meta)
    model = resolved_model.model
    config = resolve_runtime_config(options, resources, restored_meta)
    assembled_tools = assemble_tools(workspace, options, config)
    context_sources = build_runtime_context_sources(
        workspace,
        config,
        assembled_tools.loaded_extensions,
        assembled_tools.loaded_skills,
    )
    system_prompt = build_runtime_system_prompt(
        base_system_prompt=config.system_prompt,
        tools=assembled_tools.tools,
        context_sources=context_sources,
        workspace=workspace,
    )

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

    concrete = AgentSessionOptions(
        model=model,
        workspace_dir=workspace,
        system_prompt=system_prompt,
        tools=assembled_tools.tools,
        session_id=options.session_id,
        messages=options.messages,
        thinking_level=config.thinking_level,
        tool_execution=config.tool_execution,
        convert_to_llm=convert_to_llm,
        get_api_key=resolved_model.get_api_key,
        max_context_messages=config.max_context_messages,
        max_context_tokens=config.max_context_tokens,
        retain_recent_messages=config.retain_recent_messages,
        summary_builder=options.summary_builder,
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
    )
    return AgentSession(concrete)
