from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "codepilot"
DESIGN = ROOT / "docs" / "design"


def _python_files(package: str) -> list[Path]:
    root = SRC / package
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _top_level_imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _assert_no_imports(package: str, forbidden_prefixes: set[str]) -> None:
    violations: list[str] = []
    for path in _python_files(package):
        for module in _imported_modules(path):
            if any(
                module == forbidden or module.startswith(f"{forbidden}.")
                for forbidden in forbidden_prefixes
            ):
                rel = path.relative_to(ROOT).as_posix()
                violations.append(f"{rel}: {module}")
    assert violations == []


def test_interfaces_do_not_import_internal_layers() -> None:
    _assert_no_imports(
        "interfaces",
        {
            "codepilot.core",
            "codepilot.llm",
            "codepilot.sessions",
            "codepilot.tools",
        },
    )


def test_interfaces_do_not_import_runtime_assembly_internals() -> None:
    _assert_no_imports(
        "interfaces",
        {
            "codepilot.runtime.assembly",
            "codepilot.runtime.assembly_input",
            "codepilot.runtime.bootstrap",
            "codepilot.runtime.command_catalog",
            "codepilot.runtime.contracts",
            "codepilot.runtime.views",
        },
    )


def test_evaluation_uses_open_session_intent_not_assembly_request() -> None:
    _assert_no_imports(
        "evaluation",
        {
            "codepilot.runtime.assembly",
            "codepilot.runtime.assembly_input",
            "codepilot.runtime.bootstrap",
            "codepilot.runtime.contracts",
        },
    )


def test_runtime_configuration_does_not_import_assembly_at_module_load() -> None:
    modules = _top_level_imported_modules(SRC / "runtime" / "configuration.py")
    assert "assembly" not in modules
    assert "bootstrap" not in modules
    assert not any(module.startswith("bootstrap.") for module in modules)
    assert "codepilot.runtime.assembly" not in modules
    assert "codepilot.runtime.bootstrap" not in modules
    assert not any(module.startswith("codepilot.runtime.bootstrap.") for module in modules)


def test_interfaces_do_not_reach_runtime_or_session_internals() -> None:
    forbidden_patterns = {
        "approve_tool_call(": re.compile(r"\bapprove_tool_call\s*\("),
        "submit_turn(": re.compile(r"\bsubmit_turn\s*\("),
        "execute_command(": re.compile(r"\bexecute_command\s*\("),
        "approve(": re.compile(r"\bapprove\s*\("),
        "cancel(": re.compile(r"\bcancel\s*\("),
        "create_session(": re.compile(r"\bcreate_session\s*\("),
        "get_session(": re.compile(r"\bget_session\s*\("),
        "describe_session(": re.compile(r"\bdescribe_session\s*\("),
        "aclose_session(": re.compile(r"\baclose_session\s*\("),
        "cancel_run(": re.compile(r"\bcancel_run\s*\("),
        "continue_session(": re.compile(r"\bcontinue_session\s*\("),
        "get_latest_assistant_message(": re.compile(r"\bget_latest_assistant_message\s*\("),
        "get_session_status(": re.compile(r"\bget_session_status\s*\("),
        "get_session_state(": re.compile(r"\bget_session_state\s*\("),
        "get_workspace(": re.compile(r"\bget_workspace\s*\("),
        "set_task_mode(": re.compile(r"\bset_task_mode\s*\("),
        "list_session_entries(": re.compile(r"\blist_session_entries\s*\("),
        "get_session_tree(": re.compile(r"\bget_session_tree\s*\("),
        "get_entry_path(": re.compile(r"\bget_entry_path\s*\("),
        "fork_session(": re.compile(r"\bfork_session\s*\("),
        "switch_entry(": re.compile(r"\bswitch_entry\s*\("),
        ".agent": re.compile(r"\.agent\b"),
        ".store": re.compile(r"\.store\b"),
        ".memory_": re.compile(r"\.memory_"),
    }
    violations: list[str] = []
    for path in _python_files("interfaces"):
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for label, pattern in forbidden_patterns.items():
            if pattern.search(text):
                violations.append(f"{rel}: {label}")
    assert violations == []


def test_core_does_not_import_application_layers() -> None:
    _assert_no_imports(
        "core",
        {
            "codepilot.interfaces",
            "codepilot.runtime",
            "codepilot.sessions",
        },
    )


def test_core_does_not_import_executable_tool_implementation_contracts() -> None:
    violations: list[str] = []
    forbidden_prefixes = {
        "codepilot.tools.builtins",
        "codepilot.tools.contracts",
        "codepilot.tools.execution",
        "codepilot.tools.registry",
    }
    for path in _python_files("core"):
        rel = path.relative_to(ROOT).as_posix()
        for module in _imported_modules(path):
            if any(
                module == forbidden or module.startswith(f"{forbidden}.")
                for forbidden in forbidden_prefixes
            ):
                violations.append(f"{rel}: import {module}")
    assert violations == []


def test_runtime_gateway_uses_v2_core_loop_entry() -> None:
    modules = _imported_modules(SRC / "runtime" / "gateway.py")
    assert "codepilot.core.loop" in modules
    assert "codepilot.core.agent_loop" not in modules


def test_v2_core_loop_keeps_turn_modules_explicit() -> None:
    modules = _imported_modules(SRC / "core" / "loop.py")
    assert "codepilot.llm.ports" not in modules
    assert "codepilot.tools.ports" in modules
    assert "model_turn" in (SRC / "core" / "model_turn.py").read_text(encoding="utf-8")
    assert "execute_tool_turn" in (SRC / "core" / "tool_turn.py").read_text(encoding="utf-8")
    assert "completed_outcome" in (SRC / "core" / "stopping.py").read_text(encoding="utf-8")


def test_sessions_do_not_import_runtime_or_interfaces() -> None:
    _assert_no_imports(
        "sessions",
        {
            "codepilot.interfaces",
            "codepilot.runtime",
        },
    )


def test_extensions_do_not_import_session_layer() -> None:
    _assert_no_imports(
        "extensions",
        {
            "codepilot.sessions",
        },
    )


def test_extensions_do_not_import_core_layer() -> None:
    _assert_no_imports(
        "extensions",
        {
            "codepilot.core",
        },
    )


def test_sessions_do_not_use_core_agent_as_state_holder() -> None:
    violations: list[str] = []
    forbidden_imports = {
        "codepilot.core.agent",
    }
    for path in _python_files("sessions"):
        rel = path.relative_to(ROOT).as_posix()
        for module in _imported_modules(path):
            if module in forbidden_imports:
                violations.append(f"{rel}: import {module}")
        text = path.read_text(encoding="utf-8")
        if re.search(r"\.agent\b", text):
            violations.append(f"{rel}: .agent")
    assert violations == []


def test_session_commands_do_not_import_subsystem_internals() -> None:
    modules = _imported_modules(SRC / "sessions" / "commands.py")
    assert not any(
        module == "codepilot.sessions.memory"
        or module.startswith("codepilot.sessions.memory.")
        or module == "codepilot.sessions.history"
        or module.startswith("codepilot.sessions.history.")
        or module == "codepilot.sessions.context"
        or module.startswith("codepilot.sessions.context.")
        for module in modules
    )


def test_protocols_do_not_import_codepilot_business_layers() -> None:
    forbidden = {
        "codepilot.core",
        "codepilot.extensions",
        "codepilot.interfaces",
        "codepilot.llm",
        "codepilot.observability",
        "codepilot.runtime",
        "codepilot.sessions",
        "codepilot.tools",
    }
    _assert_no_imports("protocols", forbidden)


def test_production_code_does_not_import_legacy_runtime_service() -> None:
    assert not (SRC / "runtime" / "service.py").exists()
    violations: list[str] = []
    for path in SRC.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        for module in _imported_modules(path):
            if module == "codepilot.runtime.service":
                violations.append(path.relative_to(ROOT).as_posix())
    assert violations == []


def test_runtime_does_not_create_agent_session_live_objects() -> None:
    violations: list[str] = []
    forbidden_patterns = {
        "AgentSession(": re.compile(r"\bAgentSession\s*\("),
        "from_agent_session(": re.compile(r"\bfrom_agent_session\s*\("),
    }
    for path in _python_files("runtime"):
        rel = path.relative_to(ROOT).as_posix()
        for module in _imported_modules(path):
            if module == "codepilot.sessions.session":
                violations.append(f"{rel}: import {module}")
        text = path.read_text(encoding="utf-8")
        for label, pattern in forbidden_patterns.items():
            if pattern.search(text):
                violations.append(f"{rel}: {label}")
    assert violations == []


def test_runtime_does_not_import_session_internal_state_modules() -> None:
    violations: list[str] = []
    forbidden_prefixes = {
        "codepilot.sessions.context",
        "codepilot.sessions.persistence",
        "codepilot.sessions.memory",
        "codepilot.sessions.history",
    }
    for path in _python_files("runtime"):
        rel = path.relative_to(ROOT).as_posix()
        for module in _imported_modules(path):
            if any(
                module == forbidden or module.startswith(f"{forbidden}.")
                for forbidden in forbidden_prefixes
            ):
                violations.append(f"{rel}: import {module}")
    assert violations == []


def test_production_code_does_not_use_legacy_approval_bypass() -> None:
    forbidden_patterns = {
        "execute_approved(": re.compile(r"\bexecute_approved\s*\("),
        "execute_approved_tool_call(": re.compile(r"\bexecute_approved_tool_call\s*\("),
        "continue_after_tool_approval(": re.compile(r"\bcontinue_after_tool_approval\s*\("),
        "replace_tool_result_message(": re.compile(r"\breplace_tool_result_message\s*\("),
    }
    violations: list[str] = []
    for path in SRC.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT).as_posix()
        for label, pattern in forbidden_patterns.items():
            if pattern.search(text):
                violations.append(f"{rel}: {label}")
    assert violations == []


def test_production_code_does_not_build_runtime_managed_agent_tool_adapters() -> None:
    violations: list[str] = []
    for path in SRC.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "as_agent_tools(" in text:
            violations.append(path.relative_to(ROOT).as_posix())
    assert violations == []


def test_v2_boundary_documents_exist_and_name_the_spine() -> None:
    required = {
        "AGENT_LOOP_CONTRACTS_V2.md",
        "ARCHITECTURE_BOUNDARIES.md",
        "NEXT_CONVERSATION_SUMMARY.md",
        "V2_COMPLETION_AUDIT.md",
    }
    missing = [name for name in sorted(required) if not (DESIGN / name).exists()]
    assert missing == []

    for name in required:
        text = (DESIGN / name).read_text(encoding="utf-8")
        assert "RuntimeGateway.dispatch" in text
        assert "SessionController" in text
        assert "RuntimeFrame" in text
