from __future__ import annotations

"""
coding_agent 工厂入口。
"""

from codepilot.sessions.session import AgentSession
from codepilot.core.message_conversion import convert_to_llm

from .config import load_runtime_inputs, resolve_runtime_config
from .context import build_runtime_context
from .hook_pipeline import compose_after_tool_call, compose_before_tool_call, compose_lifecycle_hooks
from .model_resolver import resolve_model
from .prompt import build_runtime_system_prompt
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

    concrete = build_agent_session_options(options)
    return AgentSession(concrete)


def build_agent_session_options(options: CreateAgentSessionOptions) -> AgentSessionOptions:
    """Resolve friendly runtime options into concrete session options."""

    inputs = load_runtime_inputs(options)
    resolved_model = resolve_model(options, inputs)
    config = resolve_runtime_config(options, inputs)
    assembled_tools = assemble_tools(inputs.workspace, options, config)
    runtime_context = build_runtime_context(
        inputs.workspace,
        config,
        assembled_tools.loaded_extensions,
        assembled_tools.loaded_skills,
    )
    system_prompt = build_runtime_system_prompt(
        base_system_prompt=config.system_prompt,
        tools=assembled_tools.tools,
        runtime_context=runtime_context,
        workspace=inputs.workspace,
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

    return AgentSessionOptions(
        model=resolved_model.model,
        workspace_dir=inputs.workspace,
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
