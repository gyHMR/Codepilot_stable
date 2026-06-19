from __future__ import annotations

"""
Agent 会话工厂模块。

提供 create_agent_session 工厂函数，将上层友好的 CreateAgentSessionOptions
解析为底层的 AgentSessionOptions，并创建 AgentSession 实例。

解析流程：
1) 加载运行时输入（工作区资源、会话恢复元数据）
2) 解析模型配置（从 options / 会话元数据 / 工作区配置中确定模型）
3) 解析运行时配置（多源合并）
4) 组装工具列表（内置工具 + 扩展工具 + MCP 工具）
5) 构建运行时上下文（仓库信息、提示词准则等）
6) 构建系统提示词
7) 组装钩子（before/after tool call、生命周期钩子）
8) 构造最终的 AgentSessionOptions 并创建 AgentSession
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
    """创建 AgentSession 实例（工厂入口）。

    支持两种输入形式：
    - AgentSessionOptions: 底层参数，model 必填，直接创建。
    - CreateAgentSessionOptions: 友好参数，支持 provider/model_id 或会话恢复，
      自动解析为 AgentSessionOptions 后创建。

    Args:
        options: 会话配置选项。

    Returns:
        创建好的 AgentSession 实例。
    """
    if isinstance(options, AgentSessionOptions):
        return AgentSession(options)

    concrete = build_agent_session_options(options)
    return AgentSession(concrete)


def build_agent_session_options(options: CreateAgentSessionOptions) -> AgentSessionOptions:
    """将友好的 CreateAgentSessionOptions 解析为底层的 AgentSessionOptions。

    完整解析流程：
    1. load_runtime_inputs: 加载工作区资源和会话恢复元数据。
    2. resolve_model: 解析模型配置。
    3. resolve_runtime_config: 多源合并运行时配置。
    4. assemble_tools: 组装工具列表（内置 + 扩展 + MCP）。
    5. build_runtime_context: 构建运行时上下文。
    6. build_runtime_system_prompt: 构建系统提示词。
    7. compose_*: 组装钩子函数。

    Args:
        options: 友好的创建会话选项。

    Returns:
        解析完成的底层 AgentSessionOptions。
    """
    # 步骤 1：加载运行时输入数据
    inputs = load_runtime_inputs(options)
    # 步骤 2：解析模型
    resolved_model = resolve_model(options, inputs)
    # 步骤 3：解析运行时配置
    config = resolve_runtime_config(options, inputs)
    # 步骤 4：组装工具
    assembled_tools = assemble_tools(inputs.workspace, options, config)
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
