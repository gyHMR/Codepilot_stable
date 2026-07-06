from __future__ import annotations

import sys
from importlib.util import find_spec
from importlib import import_module
import inspect
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_codepilot_namespace_imports() -> None:
    import codepilot.core
    import codepilot.extensions
    import codepilot.evaluation
    import codepilot.llm
    import codepilot.runtime
    import codepilot.sessions
    import codepilot.tools

    cli_main = import_module("codepilot.interfaces.cli.main")
    dingtalk_main = import_module("codepilot.interfaces.dingtalk.main")

    assert codepilot.core is not None
    assert codepilot.extensions is not None
    assert codepilot.evaluation is not None
    assert codepilot.llm is not None
    assert codepilot.runtime is not None
    assert codepilot.sessions is not None
    assert codepilot.tools is not None
    assert cli_main.main is not None
    assert dingtalk_main.main is not None


def test_im_interface_source_package_is_removed() -> None:
    im_dir = SRC / "codepilot" / "interfaces" / "im"
    remaining_sources = sorted(path.name for path in im_dir.glob("*.py")) if im_dir.exists() else []

    assert remaining_sources == []
    try:
        spec = find_spec("codepilot.interfaces.im.cli")
    except ModuleNotFoundError:
        spec = None
    assert spec is None


def test_cli_parser_builds() -> None:
    from codepilot.interfaces.cli.main import build_parser

    parser = build_parser()
    args = parser.parse_args(["--prompt", "hello"])

    assert args.prompt == "hello"


def test_sessions_memory_api_is_global_only() -> None:
    import codepilot.sessions as sessions

    assert "SessionController" in sessions.__all__
    assert "SessionRunIntent" in sessions.__all__
    assert "SessionRunRecord" in sessions.__all__
    assert "SessionOpenMetadata" in sessions.__all__
    assert "load_session_open_metadata" in sessions.__all__
    assert "RepositoryBootstrap" in sessions.__all__
    assert "build_repository_bootstrap" in sessions.__all__
    assert "SessionOptions" in sessions.__all__
    assert "AgentSession" not in sessions.__all__
    assert "AgentSessionOptions" not in sessions.__all__
    assert not hasattr(sessions, "AgentSession")
    assert not hasattr(sessions, "AgentSessionOptions")
    assert "SessionStore" not in sessions.__all__
    assert "RunStore" not in sessions.__all__
    assert "MemoryStore" not in sessions.__all__
    assert "MemoryWriter" not in sessions.__all__
    assert "ContextGovernor" not in sessions.__all__
    assert "load_global_memory" not in sessions.__all__
    assert "save_global_memory" not in sessions.__all__
    assert not hasattr(sessions, "SessionStore")
    assert not hasattr(sessions, "RunStore")
    assert not hasattr(sessions, "MemoryStore")
    assert not hasattr(sessions, "MemoryWriter")
    assert not hasattr(sessions, "ContextGovernor")
    assert not hasattr(sessions, "load_global_memory")
    assert not hasattr(sessions, "save_global_memory")
    assert hasattr(sessions, "SessionOpenMetadata")
    assert hasattr(sessions, "load_session_open_metadata")
    assert hasattr(sessions, "RepositoryBootstrap")
    assert hasattr(sessions, "build_repository_bootstrap")


def test_session_controller_public_surface_matches_v2_contract() -> None:
    import inspect

    import codepilot.sessions.controller as controller_module
    from codepilot.sessions.controller import SessionController

    methods = {
        name
        for name, value in inspect.getmembers(SessionController, inspect.isfunction)
        if not name.startswith("_")
    }

    assert methods == {
        "apply_command",
        "claim_derived_controller",
        "close",
        "commit_run",
        "describe",
        "prepare_resume",
        "prepare_run",
        "stage_derived_session",
    }
    assert not hasattr(controller_module, "_controller_from_runtime_session")
    assert not hasattr(SessionController, "_stage_derived_session")
    assert not hasattr(SessionController, "_pop_derived_controller")


def test_extension_command_context_exposes_only_session_view() -> None:
    from dataclasses import fields

    from codepilot.protocols.commands import SessionCommandContext, SessionCommandView

    context_fields = {field.name for field in fields(SessionCommandContext)}
    view_fields = {field.name for field in fields(SessionCommandView)}

    assert context_fields == {"name", "args", "raw_text", "session_view"}
    assert {"session", "message"}.isdisjoint(context_fields)
    assert view_fields == {
        "session_id",
        "workspace_dir",
        "message_count",
        "task_mode",
        "leaf_id",
    }


def test_lifecycle_hook_context_exposes_only_session_view() -> None:
    from dataclasses import fields

    from codepilot.protocols.commands import SessionLifecycleContext, SessionLifecycleView

    context_fields = {field.name for field in fields(SessionLifecycleContext)}
    view_fields = {field.name for field in fields(SessionLifecycleView)}

    assert context_fields == {"text", "is_continue", "message_count", "session_view"}
    assert "session" not in context_fields
    assert view_fields == {
        "session_id",
        "workspace_dir",
        "message_count",
        "task_mode",
    }


def test_session_persistence_exports_freshness_contract() -> None:
    import codepilot.sessions.storage as persistence
    from codepilot.sessions.storage import __all__ as persistence_exports

    assert hasattr(persistence, "FreshnessResult")
    assert hasattr(persistence, "FreshnessStatus")
    assert "FreshnessResult" in persistence_exports
    assert "FreshnessStatus" in persistence_exports


def test_session_context_exports_state_contracts() -> None:
    import codepilot.sessions.context as context
    from codepilot.sessions.context import __all__ as context_exports
    from importlib.util import find_spec

    assert hasattr(context, "ContextFileRole")
    assert hasattr(context, "ContextEvidenceKind")
    assert "ContextFileRole" in context_exports
    assert "ContextEvidenceKind" in context_exports
    assert "RepositoryBootstrap" not in context_exports
    assert "build_repository_bootstrap" not in context_exports
    assert find_spec("codepilot.sessions.context.repository_context") is None


def test_no_legacy_top_level_imports() -> None:
    legacy_patterns = (
        "from ai.",
        "from ai import ",
        "from agent_core.",
        "from agent_core import ",
        "from coding_agent.",
        "from coding_agent import ",
        "from im.",
        "from im import ",
    )

    offenders: list[str] = []
    for path in (SRC / "codepilot").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(pattern in text for pattern in legacy_patterns):
            offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []


def test_runtime_does_not_import_interfaces() -> None:
    forbidden_patterns = (
        "codepilot.interfaces",
        "from ..interfaces",
        "from interfaces",
    )

    offenders: list[str] = []
    boundary_dirs = (
        SRC / "codepilot" / "runtime",
        SRC / "codepilot" / "core",
        SRC / "codepilot" / "tools",
        SRC / "codepilot" / "sessions",
    )
    for directory in boundary_dirs:
        for path in directory.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if any(pattern in text for pattern in forbidden_patterns):
                offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []


def test_sessions_layer_does_not_import_extensions() -> None:
    forbidden_patterns = (
        "codepilot.extensions",
        "from ..extensions",
        "from extensions",
    )

    offenders: list[str] = []
    for path in (SRC / "codepilot" / "sessions").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(pattern in text for pattern in forbidden_patterns):
            offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []


def test_tools_layer_does_not_import_core() -> None:
    offenders: list[str] = []
    for path in (SRC / "codepilot" / "tools").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "codepilot.core" in text or "from ..core" in text:
            offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []


def test_removed_runtime_compat_modules_are_gone() -> None:
    removed_names = (
        "approval_flow",
        "agent_session",
        "command_registry",
        "config",
        "context",
        "builtin_tools",
        "cli",
        "context_compiler",
        "errors",
        "factory",
        "hook_pipeline",
        "model_resolver",
        "prompt",
        "repository_tracker",
        "resources",
        "runner",
        "session_store",
        "serde",
        "memory",
        "tool_assembler",
        "types",
        "__main__",
    )
    removed_modules = tuple(f"codepilot.runtime.{name}" for name in removed_names)

    existing = [module for module in removed_modules if find_spec(module) is not None]

    assert existing == []


def test_runtime_public_contracts_and_views_are_separate() -> None:
    import codepilot.runtime as runtime
    import codepilot.runtime.assemble as assembly
    import codepilot.runtime.gateway as gateway_module
    import codepilot.runtime.opening as session_opening
    import codepilot.runtime.sessions as runtime_sessions
    import codepilot.runtime.views as views
    from importlib.util import find_spec

    assert hasattr(runtime, "SessionOpenIntent")
    assert hasattr(assembly, "RuntimeAssemblyIntent")
    assert hasattr(assembly, "RuntimePermissionMode")
    assert hasattr(assembly, "RuntimeAssembly")
    assert hasattr(assembly, "RuntimeDiagnostic")
    assert hasattr(assembly, "CapabilityCatalog")
    assert hasattr(views, "CommandDescriptor")
    assert hasattr(views, "SessionStatus")
    assert not hasattr(assembly, "SessionOpenRequest")
    assert not hasattr(assembly, "UserTurn")
    assert not hasattr(assembly, "RuntimeOutput")
    assert not hasattr(assembly, "ApprovalDecision")
    assert not hasattr(assembly, "ApprovalSnapshot")
    assert not hasattr(assembly, "CancellationResult")
    assert not hasattr(assembly, "CommandRequest")
    assert not hasattr(assembly, "SessionRef")
    assert not hasattr(assembly, "CommandDescriptor")
    assert not hasattr(assembly, "CommandResult")
    assert not hasattr(assembly, "SessionSnapshot")
    assert not hasattr(assembly, "SessionStatus")
    assert not hasattr(assembly, "ModelSelection")
    assert not hasattr(assembly, "AgentSessionOptions")
    assert not hasattr(runtime, "WorkspaceResourceLoader")
    assert not hasattr(runtime, "build_default_system_prompt")
    assert not hasattr(runtime, "format_commands_for_help")
    assert not hasattr(runtime, "list_runtime_commands")
    assert not hasattr(runtime.RuntimeGateway, "aclose_all")
    assert not hasattr(assembly, "create_session_controller")
    assert gateway_module.__all__ == ["RuntimeGateway"]
    assert not hasattr(gateway_module, "SessionOpenIntent")
    assert not hasattr(views, "SessionSnapshot")
    assert not hasattr(views, "CommandResult")
    assert session_opening.__all__ == [
        "AppSessionView",
        "SessionOpenIntent",
        "SessionRef",
    ]
    assert runtime_sessions.__all__ == [
        "ActiveRunRegistry",
        "RuntimeSessionEntry",
        "RuntimeSessionRegistry",
    ]
    assert not hasattr(runtime, "to_runtime_assembly_intent")
    assert find_spec("codepilot.runtime.service") is None
    assert find_spec("codepilot.runtime.contracts") is None
    assert find_spec("codepilot.runtime.commands") is None
    assert find_spec("codepilot.runtime.execution") is None
    assert find_spec("codepilot.runtime.assemble_input") is None
    assert find_spec("codepilot.runtime.assemble_types") is None
    assert find_spec("codepilot.runtime.session_opening") is None
    assert find_spec("codepilot.runtime.command_catalog") is None
    assert find_spec("codepilot.runtime.bootstrap") is None
    assert not hasattr(runtime, "create_agent_session")


def test_removed_sessions_compat_modules_are_gone() -> None:
    removed_names = (
        "branching",
        "checkpoint",
        "compaction",
        "context_compiler",
        "context_state",
        "repository_context",
        "repository_tracker",
        "run_store",
        "serde",
        "store",
    )
    removed_modules = tuple(f"codepilot.sessions.{name}" for name in removed_names)

    existing = [module for module in removed_modules if find_spec(module) is not None]

    assert existing == []


def test_removed_llm_forwarding_modules_are_gone() -> None:
    removed_modules = (
        "codepilot.llm.types",
    )

    existing = [module for module in removed_modules if find_spec(module) is not None]

    assert existing == []


def test_removed_protocol_and_llm_aliases_are_gone() -> None:
    import codepilot.llm as llm
    import codepilot.protocols as protocols
    import codepilot.protocols.tools as protocol_tools

    assert not hasattr(llm, "LLMProviderDescriptor")
    assert not hasattr(protocols, "ToolSpec")
    assert not hasattr(protocol_tools, "ToolSpec")


def test_removed_builtin_file_tool_aliases_are_gone(tmp_path: Path) -> None:
    from codepilot.tools.builtins import create_builtin_tools, get_builtin_tool_metadata
    from codepilot.tools.authoring import MUTATING_TOOL_NAMES, READ_ONLY_TOOL_NAMES

    removed_aliases = {"list_dir", "read_file", "write_file"}
    tool_names = {tool.name for tool in create_builtin_tools(tmp_path)}

    assert tool_names.isdisjoint(removed_aliases)
    assert "complete_task_step" in tool_names
    assert "task_update" in tool_names
    assert READ_ONLY_TOOL_NAMES.isdisjoint(removed_aliases)
    assert MUTATING_TOOL_NAMES.isdisjoint(removed_aliases)
    assert all(get_builtin_tool_metadata(name) is None for name in removed_aliases)

    task_update_metadata = get_builtin_tool_metadata("task_update")
    assert task_update_metadata is not None
    assert task_update_metadata.category == "task_control"
    assert task_update_metadata.read_only is True
    assert task_update_metadata.risk_level == "low"


def test_tools_refactor_exposes_new_lifecycle_modules() -> None:
    from importlib.util import find_spec

    expected_modules = (
        "codepilot.tools.authoring",
        "codepilot.tools.authoring",
        "codepilot.tools.authoring",
        "codepilot.tools.policy",
        "codepilot.tools.engine",
        "codepilot.tools.engine",
        "codepilot.tools.engine",
        "codepilot.tools.workspace",
        "codepilot.tools.workspace",
        "codepilot.tools.builtins",
        "codepilot.tools.builtins.files",
        "codepilot.tools.builtins.search",
        "codepilot.tools.builtins.shell",
        "codepilot.tools.builtins.task_control",
        "codepilot.tools.builtins.workspace_status",
    )

    missing = []
    for module in expected_modules:
        try:
            spec = find_spec(module)
        except ModuleNotFoundError:
            spec = None
        if spec is None:
            missing.append(module)

    assert missing == []


def test_removed_tools_compat_modules_are_gone() -> None:
    from importlib.util import find_spec

    removed_modules = (
        "codepilot.tools.types",
        "codepilot.tools.permissions",
        "codepilot.tools.runtime",
        "codepilot.tools.sandbox",
        "codepilot.tools.shell_policy",
        "codepilot.tools.schema_validation",
        "codepilot.tools.result_guard",
        "codepilot.tools.builtin",
    )
    existing = [module for module in removed_modules if find_spec(module) is not None]

    assert existing == []


def test_removed_tool_port_transition_adapters_are_gone() -> None:
    import codepilot.tools.ports as ports
    import codepilot.tools.adapter as adapters
    from codepilot.tools.engine import ToolRuntime

    assert not hasattr(ports, "ToolRuntimePort")
    assert hasattr(adapters, "ToolRuntimePort")
    assert not hasattr(ports, "ExecutableToolPort")
    assert not hasattr(ToolRuntime, "execute_approved")


def test_tools_top_level_exports_contracts_not_live_runtime() -> None:
    import codepilot.tools as tools
    from codepilot.tools import __all__ as tool_exports

    assert "AgentTool" in tool_exports
    assert "AgentToolResult" in tool_exports
    assert "ToolPort" not in tool_exports
    assert "ToolRuntimePort" not in tool_exports
    assert "ToolRuntime" not in tool_exports
    assert "ToolRuntimeRequest" not in tool_exports
    assert "ToolRuntimeResult" not in tool_exports
    assert "WorkspaceSandbox" not in tool_exports
    assert "SchemaValidator" not in tool_exports
    assert "ToolResultGuard" not in tool_exports
    assert not hasattr(tools, "ToolRuntime")
    assert not hasattr(tools, "ToolPort")
    assert not hasattr(tools, "ToolRuntimePort")
    assert not hasattr(tools, "ToolRuntimeRequest")
    assert not hasattr(tools, "ToolRuntimeResult")
    assert not hasattr(tools, "WorkspaceSandbox")
    assert not hasattr(tools, "SchemaValidator")
    assert not hasattr(tools, "ToolResultGuard")


def test_removed_task_control_helper_modules_are_gone() -> None:
    removed_modules = (
        "codepilot.core.task.evidence",
        "codepilot.core.task.verifier",
        "codepilot.core.task.replanner",
        "codepilot.core.task.stop",
    )

    existing = [module for module in removed_modules if find_spec(module) is not None]

    assert existing == []


def test_removed_run_result_compat_entries_are_gone() -> None:
    import codepilot.core as core
    from codepilot.core import __all__ as core_exports
    import codepilot.sessions.prepare as session_module
    from codepilot.sessions.conversation import SessionConversationState
    from codepilot.sessions.contracts import SessionOptions

    assert not hasattr(core, "Agent")
    assert not hasattr(core, "complete_task_step_tool")
    assert not hasattr(core, "has_complete_task_step_tool")
    assert "Agent" not in core_exports
    assert "complete_task_step_tool" not in core_exports
    assert "has_complete_task_step_tool" not in core_exports
    assert find_spec("codepilot.core.agent") is None
    assert find_spec("codepilot.core.agent_loop") is None
    assert not hasattr(session_module, "AgentSession")
    assert not hasattr(session_module.SessionRuntime, "prompt")
    assert not hasattr(session_module.SessionRuntime, "prompt_message")
    assert not hasattr(session_module.SessionRuntime, "run")
    assert not hasattr(session_module.SessionRuntime, "continue_run")
    assert not hasattr(session_module.SessionRuntime, "_start_run_lifecycle")
    assert not hasattr(session_module.SessionRuntime, "_complete_run_lifecycle")
    assert not hasattr(session_module.SessionRuntime, "_admit_prompt_memory")
    assert not hasattr(session_module.SessionRuntime, "_begin_task_recovery")
    assert not hasattr(session_module.SessionRuntime, "_observe_tool_memory")
    assert not hasattr(session_module.SessionRuntime, "_finalize_memory")
    assert not hasattr(session_module.SessionRuntime, "_finalize_task_recovery")
    assert not hasattr(session_module.SessionRuntime, "_check_context_freshness")
    assert not hasattr(session_module.SessionRuntime, "_run_lifecycle_hooks")
    assert not hasattr(session_module.SessionRuntime, "_write_rollback_metadata")
    assert not hasattr(session_module.SessionRuntime, "messages")
    assert not hasattr(session_module.SessionRuntime, "last_run_result")
    assert not hasattr(session_module.SessionRuntime, "last_session_run_record")
    assert "tools" not in SessionOptions.__dataclass_fields__
    assert "tools" not in SessionConversationState.__dataclass_fields__
    assert not hasattr(session_module.SessionRuntime, "last_usage")
    assert not hasattr(session_module.SessionRuntime, "cumulative_usage")
    assert not hasattr(session_module.SessionRuntime, "_last_usage")
    assert not hasattr(session_module.SessionRuntime, "_cumulative_usage")
    assert not hasattr(session_module.SessionRuntime, "_set_task_mode")
    assert not hasattr(session_module.SessionRuntime, "_list_entry_ids")
    assert not hasattr(session_module.SessionRuntime, "_list_entries")
    assert not hasattr(session_module.SessionRuntime, "_get_leaf_id")
    assert not hasattr(session_module.SessionRuntime, "_get_entry_path")
    assert not hasattr(session_module.SessionRuntime, "_get_session_tree")
    assert not hasattr(session_module.SessionRuntime, "_fork_session")
    assert not hasattr(session_module.SessionRuntime, "_create_fresh_session")
    assert not hasattr(session_module.SessionRuntime, "_fork_from_entry")
    assert not hasattr(session_module.SessionRuntime, "_switch_to_entry")
    assert not hasattr(session_module.SessionRuntime, "_switch_session")
    assert not hasattr(session_module.SessionRuntime, "_record_checkpoint")
    assert not hasattr(session_module.SessionRuntime, "_memory_summary")
    assert not hasattr(session_module.SessionRuntime, "_list_memory_records")
    assert not hasattr(session_module.SessionRuntime, "_add_project_memory")
    assert not hasattr(session_module.SessionRuntime, "_promote_memory")
    assert not hasattr(session_module.SessionRuntime, "_forget_memory")
    assert not hasattr(session_module.SessionRuntime, "_memory_status")
    assert not hasattr(session_module.SessionRuntime, "_context_command_view")
    assert not hasattr(session_module.SessionRuntime, "_capture_run_rollback_baseline")
    assert not hasattr(session_module.SessionRuntime, "_rollback_preview_view")
    assert not hasattr(session_module.SessionRuntime, "_rollback_apply_view")
    assert not hasattr(session_module.SessionRuntime, "_revert_last_run")
    assert not hasattr(session_module.SessionRuntime, "_preview_last_run_rollback")
    assert not hasattr(session_module.SessionRuntime, "_preview_run_rollback")
    assert not hasattr(session_module.SessionRuntime, "_revert_run")
    assert not hasattr(session_module.SessionRuntime, "set_task_mode")
    assert not hasattr(session_module.SessionRuntime, "list_entry_ids")
    assert not hasattr(session_module.SessionRuntime, "list_entries")
    assert not hasattr(session_module.SessionRuntime, "get_leaf_id")
    assert not hasattr(session_module.SessionRuntime, "get_entry_path")
    assert not hasattr(session_module.SessionRuntime, "get_session_tree")
    assert not hasattr(session_module.SessionRuntime, "fork_session")
    assert not hasattr(session_module.SessionRuntime, "fork_from_entry")
    assert not hasattr(session_module.SessionRuntime, "switch_to_entry")
    assert not hasattr(session_module.SessionRuntime, "switch_session")
    assert not hasattr(session_module.SessionRuntime, "record_checkpoint")
    assert not hasattr(session_module.SessionRuntime, "capture_run_rollback_baseline")
    assert not hasattr(session_module.SessionRuntime, "revert_last_run")
    assert not hasattr(session_module.SessionRuntime, "preview_last_run_rollback")
    assert not hasattr(session_module.SessionRuntime, "preview_run_rollback")
    assert not hasattr(session_module.SessionRuntime, "revert_run")
    assert "run_agent_loop_result" not in core_exports
    assert "run_agent_loop_continue_result" not in core_exports


def test_removed_evaluation_service_facade_is_gone() -> None:
    import codepilot.evaluation as evaluation
    from codepilot.evaluation import __all__ as evaluation_exports

    assert find_spec("codepilot.evaluation.service") is None
    assert not hasattr(evaluation, "EvaluationService")
    assert "EvaluationService" not in evaluation_exports


def test_core_namespace_keeps_cross_layer_contracts_out() -> None:
    import codepilot.core as core
    from codepilot.core import __all__ as core_exports
    from codepilot.extensions import AfterToolCallResult
    from codepilot.protocols import AgentRunResult
    from codepilot.tools import AgentToolResult

    assert not hasattr(core, "AgentEvent")
    assert not hasattr(core, "AgentRunResult")
    assert not hasattr(core, "AgentTool")
    assert not hasattr(core, "AgentToolResult")
    assert not hasattr(core, "AfterToolCallResult")
    assert not hasattr(core, "BeforeToolCallResult")
    assert not hasattr(core, "AgentLoopConfig")
    assert not hasattr(core, "AgentState")
    assert not hasattr(core, "LLMStreamRunner")
    assert not hasattr(core, "ToolCallCoordinator")
    assert "AgentEvent" not in core_exports
    assert "AgentRunResult" not in core_exports
    assert "AgentTool" not in core_exports
    assert "AgentToolResult" not in core_exports
    assert "AfterToolCallResult" not in core_exports
    assert "BeforeToolCallResult" not in core_exports
    assert "AgentLoopConfig" not in core_exports
    assert "AgentState" not in core_exports
    assert "LLMStreamRunner" not in core_exports
    assert "ToolCallCoordinator" not in core_exports
    assert AgentRunResult.__module__.startswith("codepilot.protocols")
    assert AgentToolResult.__module__.startswith("codepilot.protocols")
    assert AfterToolCallResult.__module__ == "codepilot.protocols.commands"


def test_web_interface_package_is_removed() -> None:
    try:
        spec = find_spec("codepilot.interfaces.web")
    except ModuleNotFoundError:
        spec = None

    assert spec is None


def test_dingtalk_namespace_exports_complete_public_contract_types() -> None:
    import codepilot.interfaces.dingtalk as dingtalk
    from codepilot.interfaces.dingtalk import __all__ as dingtalk_exports

    assert hasattr(dingtalk, "DingTalkBridge")
    assert hasattr(dingtalk, "DingTalkBridgeConfig")
    assert hasattr(dingtalk, "DingTalkInboundMessage")
    assert hasattr(dingtalk, "DingTalkOutboundMessage")
    assert hasattr(dingtalk, "describe_dingtalk_contract")
    assert "DingTalkBridge" in dingtalk_exports
    assert "DingTalkBridgeConfig" in dingtalk_exports
    assert "DingTalkInboundMessage" in dingtalk_exports
    assert "DingTalkOutboundMessage" in dingtalk_exports
    assert "describe_dingtalk_contract" in dingtalk_exports

def test_cli_interface_does_not_import_dingtalk() -> None:
    offenders: list[str] = []
    for path in (SRC / "codepilot" / "interfaces" / "cli").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "codepilot.interfaces.dingtalk" in text or "interfaces.dingtalk" in text:
            offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []


def test_dingtalk_interface_does_not_bypass_runtime_boundary() -> None:
    offenders: list[str] = []
    forbidden_patterns = (
        "import codepilot.core",
        "from codepilot.core",
        "from ..core",
        "import codepilot.tools",
        "from codepilot.tools",
        "from ..tools",
    )
    for path in (SRC / "codepilot" / "interfaces" / "dingtalk").rglob("*.py"):
        lines = path.read_text(encoding="utf-8").splitlines()
        if any(
            any(line.strip().startswith(pattern) for pattern in forbidden_patterns)
            for line in lines
        ):
            offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []


def test_cli_startup_contract_is_separate_from_renderer_exports() -> None:
    from codepilot.interfaces.cli.render import (
        CliStartupState,
        __all__ as renderer_exports,
        build_startup_state,
    )

    assert CliStartupState.__module__ == "codepilot.interfaces.cli.render"
    assert build_startup_state.__module__ == "codepilot.interfaces.cli.render"
    assert "CliStartupState" in renderer_exports
    assert "build_startup_state" in renderer_exports


def test_cli_runner_exports_compact_run_and_rpc_entrypoints() -> None:
    from codepilot.interfaces.cli.runner import __all__ as runner_exports

    assert set(runner_exports) == {
        "InputFn",
        "OutputFn",
        "RPC_PROTOCOL_VERSION",
        "RpcEmit",
        "RpcError",
        "RunMode",
        "RunOptions",
        "emit_rpc_error",
        "emit_rpc_ok",
        "emit_rpc_ready",
        "rpc_error_from_exception",
        "rpc_json_default",
        "run",
        "run_interactive",
        "run_print",
        "run_rpc",
    }


def test_cli_run_mode_types_stay_out_of_runtime_contracts() -> None:
    import codepilot.runtime.assemble as runtime_types
    import codepilot.interfaces.cli.runner as runner

    assert hasattr(runner, "RunMode")
    assert hasattr(runner, "OutputFn")
    assert hasattr(runner, "InputFn")

    assert not hasattr(runtime_types, "RunMode")
    assert not hasattr(runtime_types, "OutputFn")
    assert not hasattr(runtime_types, "InputFn")


def test_cli_namespace_exports_public_adapter_contracts() -> None:
    import codepilot.interfaces.cli as cli
    from codepilot.interfaces.cli import __all__ as cli_exports

    assert hasattr(cli, "SimpleRenderer")
    assert hasattr(cli, "TerminalRenderer")
    assert hasattr(cli, "CliStartupState")
    assert "SimpleRenderer" in cli_exports
    assert "TerminalRenderer" in cli_exports
    assert "CliStartupState" in cli_exports


def test_cli_main_module_owns_parser_and_entrypoint() -> None:
    from codepilot.interfaces.cli.main import build_parser, main

    assert find_spec("codepilot.interfaces.cli.cli") is None
    assert build_parser.__module__ == "codepilot.interfaces.cli.main"
    assert main.__module__ == "codepilot.interfaces.cli.main"


def test_cli_package_does_not_shadow_main_submodule() -> None:
    import codepilot.interfaces.cli.main as cli_main_module
    from codepilot.interfaces.cli import __all__ as cli_exports

    assert inspect.ismodule(cli_main_module)
    assert cli_main_module.__name__ == "codepilot.interfaces.cli.main"
    assert "main" not in cli_exports
