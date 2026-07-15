from __future__ import annotations

import ast
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
from codepilot.sessions import memory as memory_package
from codepilot.sessions import context as context_package
from codepilot.sessions.context import ContextService
from codepilot.sessions.context.compaction import ContextCompactor
from codepilot.sessions.memory import MemoryService
from codepilot.tools import contracts as tool_contracts
from codepilot.tools.contracts import (
    ToolCheckpointPort,
    ToolControlPort,
    ToolExecutionPort,
)


ROOT = Path(__file__).resolve().parents[1]


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
        "catalog_snapshot",
        "prepare_batch",
        "execute_prepared",
    }
    assert "prepare_resume" in _protocol_methods(ToolControlPort)
    assert "execute_prepared_resume" in _protocol_methods(ToolControlPort)
    assert set(_protocol_methods(ToolCheckpointPort)) == {
        "checkpoint_state",
        "restore_checkpoint_state",
    }


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
    model_source = (ROOT / "src" / "codepilot" / "core" / "model_step.py").read_text(
        encoding="utf-8"
    )
    tool_source = (ROOT / "src" / "codepilot" / "core" / "tool_step.py").read_text(
        encoding="utf-8"
    )

    assert 'context_seed.get("session_id")' not in model_source
    assert 'context_seed.get("session_id")' not in tool_source
    assert "session_id=input.session_id" in model_source
    assert "session_id=input.session_id" in tool_source


def test_stage_two_exposes_only_the_new_memory_surface() -> None:
    for legacy_name in (
        "MemoryStore",
        "MemoryWriter",
        "MemoryWriteContext",
        "MemoryAdmissionDecision",
        "MemoryAdmissionPolicy",
        "MemoryConflictResolver",
        "MemoryRetriever",
        "MemoryRepository",
        "RetrievedMemory",
        "MemoryMigrationReport",
    ):
        assert not hasattr(memory_package, legacy_name)

    session_source = (
        ROOT / "src" / "codepilot" / "runtime" / "session_coordinator.py"
    ).read_text(encoding="utf-8")
    context_source = (
        ROOT / "src" / "codepilot" / "sessions" / "context" / "service.py"
    ).read_text(encoding="utf-8")

    assert "MemoryService" in session_source
    assert "_finalize_memory" not in session_source
    assert "memory_writer" not in session_source
    assert "MemoryRecallPort" in context_source
    assert "MemoryRepository" not in context_source
    assert "auto_migrate" not in inspect.signature(MemoryService).parameters
    assert not hasattr(MemoryService, "migrate_legacy")
    assert not hasattr(MemoryService, "dry_run_legacy_migration")
    assert not hasattr(ContextCompactor, "_restore_legacy_checkpoint")


def test_stage_three_uses_the_target_context_layout_and_service_port() -> None:
    context_dir = ROOT / "src" / "codepilot" / "sessions" / "context"
    assert {path.name for path in context_dir.glob("*.py")} == {
        "__init__.py",
        "contracts.py",
        "service.py",
        "state.py",
        "projection.py",
        "layers.py",
        "thinning.py",
        "budget.py",
        "compaction.py",
    }
    hints = get_type_hints(ContextService.prepare)
    assert hints["request"] is ContextPrepareRequest
    assert hints["return"] is PreparedModelContext
    for legacy_name in (
        "ContextGovernor",
        "SessionContextState",
        "ContextPressurePolicy",
        "ToolArtifactLedger",
    ):
        assert not hasattr(context_package, legacy_name)

    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in context_dir.glob("*.py")
    )
    assert "_deterministic_compact_summary" not in sources
    assert "SessionStateService" not in sources
    assert "ToolRuntime" not in sources
    assert "MemoryProposal" not in sources


def test_stage_four_has_no_legacy_context_adapter_or_session_dtos() -> None:
    runtime_source = (
        ROOT / "src" / "codepilot" / "runtime" / "session_coordinator.py"
    ).read_text(encoding="utf-8")
    session_source = (
        ROOT / "src" / "codepilot" / "sessions" / "contracts.py"
    ).read_text(encoding="utf-8")
    environment_source = (
        ROOT / "src" / "codepilot" / "runtime" / "environment.py"
    ).read_text(encoding="utf-8")

    assert "RuntimeSessionContextPort" not in runtime_source
    assert "AgentContext" not in session_source
    assert "ContextPreparationRequest" not in session_source
    assert "PreparedAgentContext" not in session_source
    assert "prepare_context" not in session_source
    assert "memory: Any" not in environment_source
    assert "context_port=self.context_service" in runtime_source


def _protocol_methods(protocol: type[object]) -> tuple[str, ...]:
    return tuple(
        name
        for name, value in inspect.getmembers(protocol)
        if not name.startswith("_") and callable(value)
    )
