from __future__ import annotations

import sys
from importlib.util import find_spec
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_codepilot_namespace_imports() -> None:
    import codepilot.core
    import codepilot.extensions
    import codepilot.llm
    import codepilot.runtime
    import codepilot.sessions
    import codepilot.tools

    cli_main = import_module("codepilot.interfaces.cli.main")
    web_api = import_module("codepilot.interfaces.web.api")

    assert codepilot.core is not None
    assert codepilot.extensions is not None
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
    assert find_spec("codepilot.interfaces.im.cli") is None


def test_cli_parser_builds() -> None:
    from codepilot.interfaces.cli.main import build_parser

    parser = build_parser()
    args = parser.parse_args(["--mode", "print", "--prompt", "hello"])

    assert args.mode == "print"
    assert args.prompt == "hello"


def test_sessions_memory_api_is_global_only() -> None:
    import codepilot.sessions as sessions

    assert "load_global_memory" in sessions.__all__
    assert "save_global_memory" in sessions.__all__
    assert "load_channel_memory" not in sessions.__all__
    assert "load_merged_memory" not in sessions.__all__
    assert "save_channel_memory" not in sessions.__all__


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
        "agent_session",
        "builtin_tools",
        "cli",
        "runner",
        "session_store",
        "serde",
        "memory",
        "__main__",
    )
    removed_modules = tuple(f"codepilot.runtime.{name}" for name in removed_names)

    existing = [module for module in removed_modules if find_spec(module) is not None]

    assert existing == []


def test_removed_llm_forwarding_modules_are_gone() -> None:
    removed_modules = (
        "codepilot.llm.types",
        "codepilot.llm.stream",
    )

    existing = [module for module in removed_modules if find_spec(module) is not None]

    assert existing == []


def test_removed_builtin_file_tool_aliases_are_gone(tmp_path: Path) -> None:
    from codepilot.tools.builtin import create_builtin_tools, get_builtin_tool_metadata
    from codepilot.tools.permissions import MUTATING_TOOL_NAMES, READ_ONLY_TOOL_NAMES

    removed_aliases = {"list_dir", "read_file", "write_file"}
    tool_names = {tool.name for tool in create_builtin_tools(tmp_path)}

    assert tool_names.isdisjoint(removed_aliases)
    assert READ_ONLY_TOOL_NAMES.isdisjoint(removed_aliases)
    assert MUTATING_TOOL_NAMES.isdisjoint(removed_aliases)
    assert all(get_builtin_tool_metadata(name) is None for name in removed_aliases)


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
