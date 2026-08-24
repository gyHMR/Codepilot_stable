from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from typing import Any, get_type_hints

from codepilot.core import contracts as core_contracts
from codepilot.core.contracts import (
    ContextPreparationPort,
    ContextPrepareRequest,
    CorePorts,
    PreparedModelContext,
)
from codepilot.sessions.contracts import PreparedAgentRun
from codepilot.tools import contracts as tool_contracts
from codepilot.tools.contracts import (
    ToolCheckpointPort,
    ToolControlPort,
    ToolExecutionPort,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "codepilot"


# ── Helpers ─────────────────────────────────────────────────────────


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.level:
                package = path.relative_to(SRC).with_suffix("").parts[:-1]
                base = ("codepilot", *package)
                resolved = ".".join((*base[: len(base) - node.level + 1], node.module))
                modules.add(resolved)
            else:
                modules.add(node.module)
    return modules


def _has_forbidden_import(path: Path, forbidden: set[str]) -> list[str]:
    found: list[str] = []
    for module in _imports(path):
        if any(module == item or module.startswith(f"{item}.") for item in forbidden):
            found.append(module)
    return sorted(found)


def _protocol_methods(protocol: type[object]) -> tuple[str, ...]:
    return tuple(
        name
        for name, value in inspect.getmembers(protocol)
        if not name.startswith("_") and callable(value)
    )


# ── LLM boundary tests ──────────────────────────────────────────────


def test_llm_ports_contains_only_model_port_contracts() -> None:
    ports = SRC / "llm" / "ports.py"
    forbidden = {
        "codepilot.llm.registry",
        "codepilot.llm.stream",
        "codepilot.llm.providers",
    }
    assert _has_forbidden_import(ports, forbidden) == []

    import codepilot.llm.ports as ports_module
    assert hasattr(ports_module, "ModelPort")
    assert hasattr(ports_module, "LLMRequest")
    assert not hasattr(ports_module, "ProviderModelPort")


def test_llm_top_level_is_not_a_protocol_or_registry_facade() -> None:
    import codepilot.llm as llm

    protocol_dtos = {
        "AssistantMessage", "Context", "Message", "Model",
        "Tool", "ToolCall", "Usage", "UserMessage",
    }
    registry_functions = {
        "complete", "complete_simple", "stream", "stream_simple",
        "get_api_provider", "register_api_provider",
    }
    assert protocol_dtos.isdisjoint(set(llm.__all__))
    assert registry_functions.isdisjoint(set(llm.__all__))
    assert not any(hasattr(llm, name) for name in protocol_dtos)


def test_llm_providers_root_is_not_a_provider_facade() -> None:
    providers = SRC / "llm" / "providers" / "__init__.py"
    imports = _imports(providers)
    assert "codepilot.llm.providers.anthropic" not in imports
    assert "codepilot.llm.providers.openai" not in imports

    import codepilot.llm.providers as providers_module
    assert providers_module.__all__ == []
    assert not hasattr(providers_module, "stream_anthropic")
    assert not hasattr(providers_module, "stream_openai_compatible")


def test_registry_import_does_not_register_providers() -> None:
    from codepilot.llm import registry
    from codepilot.llm.registry import clear_api_providers, get_api_provider

    clear_api_providers()
    importlib.reload(registry)

    assert get_api_provider("anthropic-messages") is None
    assert get_api_provider("openai-compatible") is None

    registry.register_builtin_api_providers()
    assert get_api_provider("anthropic-messages") is not None
    assert get_api_provider("openai-compatible") is not None


# ── Protocol boundary tests ─────────────────────────────────────────


def test_protocols_do_not_import_higher_layers() -> None:
    forbidden = {
        "codepilot.llm", "codepilot.tools", "codepilot.core",
        "codepilot.sessions", "codepilot.observability",
        "codepilot.extensions", "codepilot.runtime", "codepilot.interfaces",
    }
    protocol_files = sorted((SRC / "protocols").rglob("*.py"))
    for path in protocol_files:
        assert _has_forbidden_import(path, forbidden) == []


def test_tool_result_message_has_status_as_its_only_error_truth() -> None:
    from codepilot.protocols import ToolResultMessage
    assert "is_error" not in ToolResultMessage.__dataclass_fields__
    assert ToolResultMessage(status="error").is_error is True
    assert ToolResultMessage(status="success").is_error is False


# ── Core boundary tests ────────────────────────────────────────────


def test_core_does_not_import_concrete_llm_or_tool_adapters() -> None:
    forbidden = {
        "codepilot.llm.adapter", "codepilot.llm.registry",
        "codepilot.llm.stream", "codepilot.tools.adapter",
        "codepilot.tools.engine",
    }
    core_files = sorted((SRC / "core").rglob("*.py"))
    for path in core_files:
        assert _has_forbidden_import(path, forbidden) == []


def test_core_does_not_import_sessions_or_memory() -> None:
    forbidden: list[tuple[Path, str]] = []
    for path in (ROOT / "src" / "codepilot" / "core").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = ""
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(("codepilot.sessions", "codepilot.memory")):
                        forbidden.append((path, alias.name))
            if module.startswith(("codepilot.sessions", "codepilot.memory")):
                forbidden.append((path, module))
    assert forbidden == []


def test_core_no_longer_reads_session_identity_from_context_seed() -> None:
    model_source = (ROOT / "src" / "codepilot" / "core" / "model_step.py").read_text(encoding="utf-8")
    tool_source = (ROOT / "src" / "codepilot" / "core" / "tool_step.py").read_text(encoding="utf-8")

    assert 'context_seed.get("session_id")' not in model_source
    assert 'context_seed.get("session_id")' not in tool_source
    assert "session_id=input.session_id" in model_source
    assert "session_id=input.session_id" in tool_source


def test_context_port_has_typed_request_and_response() -> None:
    hints = get_type_hints(ContextPreparationPort.prepare)
    assert hints["request"] is ContextPrepareRequest
    assert PreparedModelContext in getattr(hints["return"], "__args__", ())
    assert Any not in hints.values()
    assert not hasattr(core_contracts, "ContextPort")


def test_core_ports_expose_only_tool_execution_capability() -> None:
    hints = get_type_hints(CorePorts)
    prepared_hints = get_type_hints(PreparedAgentRun)

    assert ToolExecutionPort in getattr(hints["tools"], "__args__", ())
    assert prepared_hints["context_port"] == ContextPreparationPort | None
    assert "memory" not in hints
    assert "checkpoint" not in hints


def test_tool_ports_separate_core_runtime_and_checkpoint_capabilities() -> None:
    assert not hasattr(tool_contracts, "ToolPort")
    assert set(_protocol_methods(ToolExecutionPort)) == {
        "catalog_snapshot", "prepare_batch", "execute_prepared",
    }
    assert "prepare_resume" in _protocol_methods(ToolControlPort)
    assert "execute_prepared_resume" in _protocol_methods(ToolControlPort)
    assert set(_protocol_methods(ToolCheckpointPort)) == {
        "checkpoint_state", "restore_checkpoint_state",
    }


# ── Runtime / Sessions boundary tests ───────────────────────────────


def test_runtime_gateway_does_not_construct_concrete_ports() -> None:
    gateway = SRC / "runtime" / "gateway.py"
    forbidden = {
        "codepilot.llm.adapter",
        "codepilot.tools.adapter",
        "codepilot.tools.engine",
    }
    assert _has_forbidden_import(gateway, forbidden) == []


def test_sessions_do_not_import_llm_or_tools_internals() -> None:
    forbidden = {
        "codepilot.llm.adapter",
        "codepilot.llm.stream",
        "codepilot.tools.workspace",
    }
    session_files = sorted((SRC / "sessions").rglob("*.py"))
    for path in session_files:
        assert _has_forbidden_import(path, forbidden) == []


def test_session_config_does_not_import_concrete_model_port() -> None:
    checked = [SRC / "sessions" / "contracts.py"]
    for path in checked:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "codepilot.llm.adapter"
            for alias in node.names
        }
        assert "ProviderModelPort" not in imported_names
