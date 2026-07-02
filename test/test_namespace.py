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
    web_api = import_module("codepilot.interfaces.web.api")

    assert codepilot.core is not None
    assert codepilot.extensions is not None
    assert codepilot.evaluation is not None
    assert codepilot.llm is not None
    assert codepilot.runtime is not None
    assert codepilot.sessions is not None
    assert codepilot.tools is not None
    assert cli_main.main is not None
    assert web_api.describe_web_contract is not None


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

    assert "load_global_memory" in sessions.__all__
    assert "save_global_memory" in sessions.__all__
    assert "load_channel_memory" not in sessions.__all__
    assert "load_merged_memory" not in sessions.__all__
    assert "save_channel_memory" not in sessions.__all__


def test_session_persistence_exports_freshness_contract() -> None:
    import codepilot.sessions.persistence as persistence
    from codepilot.sessions.persistence import __all__ as persistence_exports

    assert hasattr(persistence, "FreshnessResult")
    assert hasattr(persistence, "FreshnessStatus")
    assert "FreshnessResult" in persistence_exports
    assert "FreshnessStatus" in persistence_exports


def test_session_context_exports_state_contracts() -> None:
    import codepilot.sessions.context as context
    from codepilot.sessions.context import __all__ as context_exports

    assert hasattr(context, "ContextFileRole")
    assert hasattr(context, "ContextEvidenceKind")
    assert "ContextFileRole" in context_exports
    assert "ContextEvidenceKind" in context_exports


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


def test_runtime_public_contract_uses_contracts_module() -> None:
    import codepilot.runtime as runtime
    import codepilot.runtime.contracts as contracts

    assert hasattr(contracts, "CreateAgentSessionOptions")
    assert hasattr(contracts, "UserInput")
    assert not hasattr(runtime, "WorkspaceResourceLoader")
    assert not hasattr(runtime, "build_default_system_prompt")
    assert not hasattr(runtime, "format_commands_for_help")
    assert not hasattr(runtime, "list_runtime_commands")


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
        "codepilot.llm.stream",
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
    from codepilot.tools.metadata import MUTATING_TOOL_NAMES, READ_ONLY_TOOL_NAMES

    removed_aliases = {"list_dir", "read_file", "write_file"}
    tool_names = {tool.name for tool in create_builtin_tools(tmp_path)}

    assert tool_names.isdisjoint(removed_aliases)
    assert READ_ONLY_TOOL_NAMES.isdisjoint(removed_aliases)
    assert MUTATING_TOOL_NAMES.isdisjoint(removed_aliases)
    assert all(get_builtin_tool_metadata(name) is None for name in removed_aliases)


def test_tools_refactor_exposes_new_lifecycle_modules() -> None:
    from importlib.util import find_spec

    expected_modules = (
        "codepilot.tools.contracts",
        "codepilot.tools.registry",
        "codepilot.tools.metadata",
        "codepilot.tools.policy",
        "codepilot.tools.execution",
        "codepilot.tools.argument_schema",
        "codepilot.tools.result_safety",
        "codepilot.tools.workspace_safety",
        "codepilot.tools.shell_safety",
        "codepilot.tools.builtins",
        "codepilot.tools.builtins.files",
        "codepilot.tools.builtins.search",
        "codepilot.tools.builtins.shell",
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


def test_removed_run_result_compat_entries_are_gone() -> None:
    import codepilot.core.agent_loop as agent_loop
    from codepilot.core import Agent, __all__ as core_exports
    from codepilot.sessions.session import AgentSession

    assert not hasattr(Agent, "prompt")
    assert not hasattr(Agent, "continue_run_result")
    assert not hasattr(AgentSession, "prompt")
    assert not hasattr(AgentSession, "prompt_message")
    assert not hasattr(agent_loop, "run_agent_loop_result")
    assert not hasattr(agent_loop, "run_agent_loop_continue_result")
    assert "run_agent_loop_result" not in core_exports
    assert "run_agent_loop_continue_result" not in core_exports


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
    assert not hasattr(core, "ToolCallCoordinator")
    assert "AgentEvent" not in core_exports
    assert "AgentRunResult" not in core_exports
    assert "AgentTool" not in core_exports
    assert "AgentToolResult" not in core_exports
    assert "ToolCallCoordinator" not in core_exports
    assert AgentRunResult.__module__.startswith("codepilot.protocols")
    assert AgentToolResult.__module__.startswith("codepilot.protocols")
    assert AfterToolCallResult.__module__ == "codepilot.core.types"


def test_web_namespace_exports_complete_public_contract_types() -> None:
    import codepilot.interfaces.web as web
    from codepilot.interfaces.web import __all__ as web_exports

    assert hasattr(web, "ApprovalDecision")
    assert hasattr(web, "WebEventKind")
    assert "ApprovalDecision" in web_exports
    assert "WebEventKind" in web_exports


def test_cli_startup_contract_is_separate_from_renderer_exports() -> None:
    from codepilot.interfaces.cli.startup import CliStartupState, build_startup_state
    from codepilot.interfaces.cli.renderer import __all__ as renderer_exports

    assert CliStartupState.__module__ == "codepilot.interfaces.cli.startup"
    assert build_startup_state.__module__ == "codepilot.interfaces.cli.startup"
    assert "CliStartupState" not in renderer_exports
    assert "build_startup_state" not in renderer_exports


def test_cli_runner_exports_only_run_mode_entrypoints() -> None:
    from codepilot.interfaces.cli.runner import __all__ as runner_exports

    assert set(runner_exports) == {
        "RunOptions",
        "run",
        "run_interactive",
        "run_print",
        "run_rpc",
    }


def test_cli_run_mode_types_stay_out_of_runtime_contracts() -> None:
    import codepilot.runtime.contracts as runtime_types
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
