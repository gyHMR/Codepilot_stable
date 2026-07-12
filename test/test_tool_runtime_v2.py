from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import replace
from pathlib import Path


def _workspace_runtime(workspace: Path, *, registration=None):
    from codepilot.tools.builtins.workspace import create_workspace_status_registration
    from codepilot.tools.registry import ToolRegistry
    from codepilot.tools.runtime import ToolRuntime
    from codepilot.tools.sandbox import WorkspaceSandbox

    registry = ToolRegistry()
    registration = registration or create_workspace_status_registration(WorkspaceSandbox(workspace))
    registration_id = registry.register(registration)
    runtime = ToolRuntime(registry=registry)
    return runtime, registration, registration_id


def _request(registration_id: str, *, arguments=None):
    from codepilot.tools.contracts import ToolExecutionRequest

    return ToolExecutionRequest(
        run_id="run-v2",
        session_id="session-v2",
        tool_call_id="call-v2",
        tool_name="workspace_status",
        arguments=arguments or {},
        mode="execute",
        registration_id=registration_id,
    )


def test_workspace_status_executes_through_canonical_runtime(tmp_path: Path) -> None:
    from codepilot.tools.results import ToolResult

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "example.py").write_text("print('ok')\n", encoding="utf-8", newline="\n")
    runtime, _, registration_id = _workspace_runtime(tmp_path)
    snapshot = runtime.catalog_snapshot(mode="execute")

    result = asyncio.run(runtime.execute(_request(registration_id)))

    assert isinstance(result, ToolResult)
    assert result.status == "success"
    assert snapshot.entries[0].registration_id == registration_id
    assert result.registration_id == registration_id
    assert result.data["details"]["is_git_repository"] is True
    assert result.data["details"]["entry_count"] == 1
    assert "Git repository: True" in result.content[0].text
    assert {effect.kind for effect in result.effects} == {"filesystem_read"}


def test_canonical_runtime_rejects_invalid_input_before_handler(tmp_path: Path) -> None:
    from codepilot.tools.builtins.workspace import create_workspace_status_registration
    from codepilot.tools.sandbox import WorkspaceSandbox

    calls = 0
    registration = create_workspace_status_registration(WorkspaceSandbox(tmp_path))

    async def handler(input, context):
        nonlocal calls
        _ = input, context
        calls += 1
        raise AssertionError("handler must not run")

    runtime, _, registration_id = _workspace_runtime(
        tmp_path,
        registration=replace(registration, handler=handler),
    )

    result = asyncio.run(runtime.execute(_request(registration_id, arguments={"extra": True})))

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "tool.input.invalid"
    assert calls == 0


def test_canonical_runtime_converts_invalid_output_to_standard_error(tmp_path: Path) -> None:
    from codepilot.tools.builtins.workspace import create_workspace_status_registration
    from codepilot.tools.sandbox import WorkspaceSandbox

    registration = create_workspace_status_registration(WorkspaceSandbox(tmp_path))

    async def invalid_handler(input, context):
        _ = input, context
        return {"unexpected": True}

    runtime, _, registration_id = _workspace_runtime(
        tmp_path,
        registration=replace(registration, handler=invalid_handler),
    )

    result = asyncio.run(runtime.execute(_request(registration_id)))

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "tool.output.invalid"


def test_runtime_rejects_stale_model_call_without_starting_reloaded_handler(
    tmp_path: Path,
) -> None:
    from codepilot.tools.builtins.workspace import create_workspace_status_registration
    from codepilot.tools.sandbox import WorkspaceSandbox

    calls = 0
    runtime, registration, stale_id = _workspace_runtime(tmp_path)

    async def reloaded_handler(input, context):
        nonlocal calls
        _ = input, context
        calls += 1
        return {"details": {}}

    runtime.registry.register(replace(registration, handler=reloaded_handler), replace=True)

    result = asyncio.run(runtime.execute(_request(stale_id)))

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "tool.registration.stale"
    assert calls == 0
