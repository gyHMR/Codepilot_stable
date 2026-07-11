from __future__ import annotations

import ast
import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "codepilot"


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
    assert hasattr(ports_module, "LLMCorrelation")
    assert hasattr(ports_module, "LLMReasoningDelta")
    assert hasattr(ports_module, "LLMToolCallDelta")
    assert not hasattr(ports_module, "ProviderModelPort")


def test_protocols_do_not_import_higher_layers() -> None:
    forbidden = {
        "codepilot.llm",
        "codepilot.tools",
        "codepilot.core",
        "codepilot.sessions",
        "codepilot.observability",
        "codepilot.extensions",
        "codepilot.runtime",
        "codepilot.interfaces",
    }
    protocol_files = sorted((SRC / "protocols").rglob("*.py"))

    for path in protocol_files:
        assert _has_forbidden_import(path, forbidden) == []


def test_llm_top_level_is_not_a_protocol_or_registry_facade() -> None:
    import codepilot.llm as llm

    protocol_dtos = {
        "AssistantMessage",
        "Context",
        "Message",
        "Model",
        "Tool",
        "ToolCall",
        "Usage",
        "UserMessage",
    }
    registry_functions = {
        "complete",
        "complete_simple",
        "stream",
        "stream_simple",
        "get_api_provider",
        "register_api_provider",
    }

    assert protocol_dtos.isdisjoint(set(llm.__all__))
    assert registry_functions.isdisjoint(set(llm.__all__))
    assert not any(hasattr(llm, name) for name in protocol_dtos)


def test_llm_providers_root_is_not_a_provider_facade() -> None:
    providers = SRC / "llm" / "providers" / "__init__.py"
    imports = _imports(providers)

    assert "codepilot.llm.providers.anthropic" not in imports
    assert "codepilot.llm.providers.openai" not in imports
    assert "codepilot.llm.providers.register_builtins" not in imports

    import codepilot.llm.providers as providers_module

    assert providers_module.__all__ == []
    assert not hasattr(providers_module, "stream_anthropic")
    assert not hasattr(providers_module, "stream_openai_compatible")
    assert not hasattr(providers_module, "register_builtin_api_providers")


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


def test_tools_ports_contains_only_tool_port_contracts() -> None:
    ports = SRC / "tools" / "contracts.py"
    forbidden = {
        "codepilot.tools.runtime",
        "codepilot.tools.adapter",
    }

    assert _has_forbidden_import(ports, forbidden) == []

    import codepilot.tools.contracts as ports_module

    assert hasattr(ports_module, "ToolPort")
    assert hasattr(ports_module, "ToolCatalogView")
    assert hasattr(ports_module, "ToolInvocation")
    assert hasattr(ports_module, "ToolObservation")
    assert hasattr(ports_module, "ToolPolicyContext")
    assert not hasattr(ports_module, "ToolRuntimePort")


def test_tools_top_level_is_not_the_core_port_surface() -> None:
    import codepilot.tools as tools

    public_tool_surface = {
        "ToolCatalogView",
        "ToolDefinition",
        "ToolInvocation",
        "ToolObservation",
        "ToolPort",
        "ToolResumeDecision",
        "ToolRuntime",
        "PermissionPolicy",
    }
    removed_names = {
        "AgentTool",
        "AgentToolResult",
        "ToolRuntimePort",
        "ToolRuntimeRequest",
        "ToolRuntimeResult",
    }

    assert public_tool_surface <= set(tools.__all__)
    assert not any(hasattr(tools, name) for name in removed_names)


def test_core_does_not_import_concrete_llm_or_tool_adapters() -> None:
    forbidden = {
        "codepilot.llm.adapter",
        "codepilot.llm.registry",
        "codepilot.llm.stream",
        "codepilot.tools.adapter",
        "codepilot.tools.engine",
    }
    core_files = sorted((SRC / "core").rglob("*.py"))

    for path in core_files:
        assert _has_forbidden_import(path, forbidden) == []


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
    checked = [
        SRC / "sessions" / "contracts.py",
    ]

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
