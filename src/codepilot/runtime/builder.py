from __future__ import annotations

"""Open one runtime session by wiring config, model, tools, prompt, and ports."""

from dataclasses import replace

from codepilot.core.model_step import convert_to_llm
from codepilot.llm.adapter import ProviderModelPort
from codepilot.llm.registry import register_builtin_api_providers
from codepilot.sessions.contracts import SessionOptions
from codepilot.sessions.controller import create_session_controller
from codepilot.tools.adapter import ToolRuntimePort

from .config import load_runtime_config
from .hooks import (
    compose_after_tool_call,
    compose_before_tool_call,
    compose_lifecycle_hooks,
)
from .model import resolve_runtime_model
from .opening import SessionOpenIntent
from .prompt import build_system_prompt
from .sessions import RuntimeSession, RuntimeStatusInfo
from .tools import build_runtime_tools


def build_runtime_session(intent: SessionOpenIntent) -> RuntimeSession:
    """Build a runnable session from the public open-session intent."""

    register_builtin_api_providers()

    config = load_runtime_config(intent)
    model = resolve_runtime_model(intent, config)
    tools = build_runtime_tools(config.workspace, intent, config)
    system_prompt = build_system_prompt(
        workspace=config.workspace,
        config=config,
        tools=tools,
    )

    before_tool_call = compose_before_tool_call(
        intent.before_tool_call,
        tools.before_tool_hooks,
    )
    after_tool_call = compose_after_tool_call(
        intent.after_tool_call,
        tools.after_tool_hooks,
    )
    before_prompt_hooks = compose_lifecycle_hooks(
        intent.before_prompt_hooks,
        tools.before_prompt_hooks,
    )
    after_prompt_hooks = compose_lifecycle_hooks(
        intent.after_prompt_hooks,
        tools.after_prompt_hooks,
    )

    session_options = SessionOptions(
        model=model.model,
        workspace_dir=config.workspace,
        system_prompt=system_prompt,
        session_id=intent.session_id,
        messages=list(intent.messages),
        thinking_level=config.thinking_level,
        tool_execution=config.tool_execution,
        max_tool_calls_per_turn=config.max_tool_calls_per_turn,
        memory_enabled=intent.memory_enabled,
        task_control_enabled=intent.task_control_enabled,
        task_mode=config.task_mode,
        planning_budget_profile=config.planning_budget_profile,
        max_task_replans_per_run=intent.max_task_replans_per_run or 2,
        convert_to_llm=convert_to_llm,
        get_api_key=model.get_api_key,
        retry_enabled=config.retry_enabled,
        max_retries=config.max_retries,
        retry_base_delay_ms=config.retry_base_delay_ms,
        extension_commands={
            **intent.extension_commands,
            **tools.commands,
        },
        before_prompt_hooks=before_prompt_hooks,
        after_prompt_hooks=after_prompt_hooks,
        before_tool_call=before_tool_call,
        after_tool_call=after_tool_call,
        stream_fn=intent.stream_fn,
        prepare_context=None,
    )

    controller = create_session_controller(session_options)
    effective_options = replace(session_options, session_id=controller.session_id)
    model_port = ProviderModelPort(
        model=effective_options.model,
        stream_fn=effective_options.stream_fn,
        convert_messages=effective_options.convert_to_llm,
        get_api_key=effective_options.get_api_key,
    )
    tool_port = ToolRuntimePort(
        tools.runtime,
        before_tool_call=effective_options.before_tool_call,
        after_tool_call=effective_options.after_tool_call,
    )

    return RuntimeSession(
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
    )


__all__ = [
    "build_runtime_session",
]
