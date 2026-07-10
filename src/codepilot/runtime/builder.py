from __future__ import annotations

"""Open one runtime session by wiring config, model, tools, prompt, and ports."""

from dataclasses import replace

from codepilot.core.model_step import convert_to_llm
from codepilot.llm.adapter import ProviderModelPort
from codepilot.llm.registry import register_builtin_api_providers
from codepilot.sessions.contracts import SessionOptions
from codepilot.sessions.controller import create_session_controller
from codepilot.tools import DeferredApprovalProvider, PermissionPolicy, ToolRuntime

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
from .subagents import create_exploration_tools
from .tools import build_runtime_tools


def build_runtime_session(intent: SessionOpenIntent) -> RuntimeSession:
    """Build a runnable session from the public open-session intent."""

    register_builtin_api_providers()

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
        model=runtime_model,
        workspace_dir=config.workspace,
        system_prompt=system_prompt,
        system_prompt_builder=system_prompt_for,
        session_id=intent.session_id,
        messages=list(intent.messages),
        thinking_level=config.thinking_level,
        tool_execution=config.tool_execution,
        max_tool_calls_per_turn=config.max_tool_calls_per_turn,
        memory_enabled=intent.memory_enabled,
        current_mode=config.current_mode,
        planning_budget_profile=config.planning_budget_profile,
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
        prepare_context=intent.prepare_context,
    )

    controller = create_session_controller(session_options)
    effective_options = replace(session_options, session_id=controller.session_id)
    model_port = ProviderModelPort(
        model=effective_options.model,
        stream_fn=effective_options.stream_fn,
        convert_messages=effective_options.convert_to_llm,
        get_api_key=effective_options.get_api_key,
    )
    tool_port = ToolRuntime(
        registry=tools.registry,
        permission_policy=PermissionPolicy(
            permission_mode=config.tool_permission_mode,
            block_dangerous_bash=config.block_dangerous_bash,
            bash_allow_patterns=config.bash_allow_patterns,
            bash_block_patterns=config.bash_block_patterns,
        ),
        approval_provider=intent.approval_provider or DeferredApprovalProvider(),
        before_tool_call=effective_options.before_tool_call,
        after_tool_call=effective_options.after_tool_call,
    )

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
    )
    tools.registry.extend(
        create_exploration_tools(
            workspace=config.workspace,
            session_provider=lambda: runtime_session,
        )
    )
    return runtime_session


__all__ = [
    "build_runtime_session",
]
