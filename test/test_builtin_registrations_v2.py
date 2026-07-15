from __future__ import annotations

import asyncio
import hashlib
import itertools
import sys
from pathlib import Path

import pytest

from tool_runtime_testkit import execute_tool


_CANONICAL_BUILTIN_NAMES = {
    "workspace_status",
    "ls",
    "read",
    "grep",
    "find",
    "write",
    "edit",
    "apply_patch",
    "command",
    "bash",
}
_CALL_SEQUENCE = itertools.count(1)


def _runtime(
    workspace: Path,
    *,
    enabled_names: list[str] | None = None,
    permission_engine=None,
):
    from codepilot.tools import create_builtin_registrations
    from codepilot.tools.registry import ToolRegistry
    from codepilot.tools.runtime import ToolRuntime
    from codepilot.tools.security import PermissionEngine, PermissionRule

    registry = ToolRegistry()
    registration_ids: dict[str, str] = {}
    for registration in create_builtin_registrations(
        workspace,
        enabled_names=enabled_names,
    ):
        registration_id = registry.register(registration)
        assert isinstance(registration_id, str)
        registration_ids[registration.spec.name] = registration_id
    return (
        ToolRuntime(
            registry=registry,
            permission_engine=permission_engine
            or PermissionEngine(
                rules=(PermissionRule("*", "*", "allow", priority=100),)
            ),
        ),
        registration_ids,
    )


def _execute(runtime, registration_ids, name: str, arguments: dict[str, object]):
    from codepilot.tools.contracts import ToolExecutionRequest

    request = ToolExecutionRequest(
        run_id="run-builtins-v2",
        session_id="session-builtins-v2",
        tool_call_id=f"call-{name}-{next(_CALL_SEQUENCE)}",
        tool_name=name,
        arguments=arguments,
        mode="execute",
        registration_id=registration_ids[name],
    )
    return asyncio.run(execute_tool(runtime, request))


def test_builtin_registration_catalog_is_complete_and_opaque(tmp_path: Path) -> None:
    from codepilot.tools import create_builtin_registrations

    registrations = create_builtin_registrations(tmp_path)

    assert {item.spec.name for item in registrations} == _CANONICAL_BUILTIN_NAMES
    for registration in registrations:
        assert len(registration.spec.description) >= 20
        assert registration.spec.input_schema["$schema"].endswith("draft/2020-12/schema")
        assert registration.spec.output_schema is not None
        assert registration.spec.output_schema["$schema"].endswith("draft/2020-12/schema")
        assert registration.policy.declared_effects
        assert callable(registration.handler)
        assert callable(registration.access_resolver.resolve)
        assert callable(registration.renderer.render)

    filtered = create_builtin_registrations(tmp_path, enabled_names=["read", "grep"])
    assert [item.spec.name for item in filtered] == ["read", "grep"]


def test_builtin_tool_descriptions_explain_selection_and_bounded_results(
    tmp_path: Path,
) -> None:
    from codepilot.tools import create_builtin_registrations

    descriptions = {
        item.spec.name: item.spec.description
        for item in create_builtin_registrations(tmp_path)
    }

    assert "offset and limit" in descriptions["read"]
    assert "hard line and character bounds" in descriptions["read"]
    assert "first unreturned line" in descriptions["read"]
    assert "use find for path-name discovery" in descriptions["grep"]
    assert "do not use it for content search" in descriptions["find"]
    assert "full rewrites" in descriptions["write"]
    assert "expected_file_hash" in descriptions["edit"]
    assert "prevents the batch from being partially applied" in descriptions["apply_patch"]
    assert "argv array without Shell parsing" in descriptions["command"]
    assert "Use only when pipes" in descriptions["bash"]


def test_file_handler_consumes_resolver_canonical_path_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from codepilot.tools.sandbox import WorkspaceSandbox

    target = tmp_path / "sample.txt"
    target.write_text("sample\n", encoding="utf-8", newline="\n")
    calls = 0
    original = WorkspaceSandbox.resolve_path

    def counting_resolve(self, path):
        nonlocal calls
        calls += 1
        return original(self, path)

    monkeypatch.setattr(WorkspaceSandbox, "resolve_path", counting_resolve)
    runtime, ids = _runtime(tmp_path, enabled_names=["read"])

    result = _execute(runtime, ids, "read", {"path": "sample.txt"})

    assert result.status == "success"
    assert calls == 1


@pytest.mark.parametrize(
    ("name", "arguments", "expected_text"),
    [
        ("ls", {"path": "."}, "sample.py"),
        ("read", {"path": "sample.py"}, "needle"),
        ("grep", {"pattern": "needle", "path": ".", "glob": "**/*.py"}, "sample.py:1"),
        ("find", {"path": ".", "pattern": "**/*.py"}, "sample.py"),
    ],
)
def test_readonly_builtins_execute_through_canonical_runtime(
    tmp_path: Path,
    name: str,
    arguments: dict[str, object],
    expected_text: str,
) -> None:
    (tmp_path / "sample.py").write_text("needle = True\n", encoding="utf-8", newline="\n")
    runtime, registration_ids = _runtime(tmp_path, enabled_names=[name])

    result = _execute(runtime, registration_ids, name, arguments)

    assert result.status == "success"
    assert expected_text in result.data["text"]
    assert {effect.kind for effect in result.effects} == {"filesystem_read"}


def test_read_schema_enforces_hard_result_limits(tmp_path: Path) -> None:
    from codepilot.tools import create_builtin_registrations

    registration = next(
        item
        for item in create_builtin_registrations(tmp_path, enabled_names=["read"])
        if item.spec.name == "read"
    )
    properties = registration.spec.input_schema["properties"]

    assert properties["max_chars"]["maximum"] == 30_000
    assert properties["limit"]["maximum"] == 1_000


def test_read_result_projects_file_identity_into_session_message(tmp_path: Path) -> None:
    from codepilot.tools.results import to_tool_result_message

    content = "first line\nsecond line\n"
    (tmp_path / "sample.py").write_text(content, encoding="utf-8", newline="\n")
    runtime, registration_ids = _runtime(tmp_path, enabled_names=["read"])

    result = _execute(runtime, registration_ids, "read", {"path": "sample.py"})
    message = to_tool_result_message(result)

    assert result.data["details"]["sha256"] == hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()
    assert message.details == result.data["details"]
    assert "text" not in message.details


def test_read_batch_rejects_aggregate_output_budget_before_handlers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from codepilot.tools.contracts import ToolExecutionRequest

    for index in range(6):
        (tmp_path / f"sample-{index}.py").write_text(
            f"value_{index} = True\n",
            encoding="utf-8",
            newline="\n",
        )
    runtime, registration_ids = _runtime(
        tmp_path, enabled_names=["read", "write"]
    )
    read_requests = tuple(
        ToolExecutionRequest(
            run_id="run-read-budget",
            session_id="session-read-budget",
            tool_call_id=f"call-read-{index}",
            tool_name="read",
            arguments={"path": f"sample-{index}.py"},
            mode="execute",
            registration_id=registration_ids["read"],
        )
        for index in range(6)
    )
    requests = (
        *read_requests,
        ToolExecutionRequest(
            run_id="run-read-budget",
            session_id="session-read-budget",
            tool_call_id="call-write",
            tool_name="write",
            arguments={"path": "must-not-exist.txt", "content": "blocked\n"},
            mode="execute",
            registration_id=registration_ids["write"],
        ),
    )

    def unexpected_read(_path: Path) -> bytes:
        raise AssertionError("read handler must not run for a rejected batch")

    monkeypatch.setattr(Path, "read_bytes", unexpected_read)
    results = asyncio.run(runtime.execute_batch(requests))

    assert [result.status for result in results] == ["error"] * 7
    assert [result.error.code for result in results] == [
        "tool.batch.output_budget_exceeded"
    ] * 7
    assert all(
        result.error.details == {
            "requested_chars": 120_000,
            "max_chars": 100_000,
        }
        for result in results
    )
    assert not (tmp_path / "must-not-exist.txt").exists()


def test_write_edit_and_apply_patch_share_one_canonical_result_shape(tmp_path: Path) -> None:
    runtime, registration_ids = _runtime(
        tmp_path,
        enabled_names=["write", "edit", "apply_patch"],
    )

    write_result = _execute(
        runtime,
        registration_ids,
        "write",
        {"path": "sample.txt", "content": "alpha\n"},
    )
    edit_result = _execute(
        runtime,
        registration_ids,
        "edit",
        {"path": "sample.txt", "old_text": "alpha", "new_text": "beta"},
    )
    patch_result = _execute(
        runtime,
        registration_ids,
        "apply_patch",
        {
            "edits": [
                {"path": "sample.txt", "old_text": "beta", "new_text": "gamma"},
            ]
        },
    )

    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == "gamma\n"
    for result in (write_result, edit_result, patch_result):
        assert result.status == "success"
        assert result.data["workspace_changed"] is True
        assert result.data["affected_paths"] == ("sample.txt",)
        assert "filesystem_write" in {effect.kind for effect in result.effects}


def test_path_escape_is_rejected_before_builtin_handler(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-tool-test.txt"
    outside.write_text("secret\n", encoding="utf-8", newline="\n")
    runtime, registration_ids = _runtime(tmp_path, enabled_names=["read"])

    result = _execute(
        runtime,
        registration_ids,
        "read",
        {"path": "../outside-tool-test.txt"},
    )

    assert result.status == "denied"
    assert result.error is not None
    assert result.error.code == "tool.access.invalid"
    assert not result.effects


def test_bash_reports_process_effect_and_preserves_nonzero_error_code(tmp_path: Path) -> None:
    runtime, registration_ids = _runtime(tmp_path, enabled_names=["bash"])
    executable = str(Path(sys.executable))

    success = _execute(
        runtime,
        registration_ids,
        "bash",
        {"command": f'"{executable}" -c "print(\'canonical-bash\')"'},
    )
    failed = _execute(
        runtime,
        registration_ids,
        "bash",
        {"command": f'"{executable}" -c "raise SystemExit(7)"'},
    )

    assert success.status == "success"
    assert success.data["exit_code"] == 0
    assert "canonical-bash" in success.data["text"]
    assert "process_spawn" in {effect.kind for effect in success.effects}
    assert failed.status == "error"
    assert failed.error is not None
    assert failed.error.code == "shell_exit_nonzero"


def test_controlled_command_executes_without_shell_parsing(tmp_path: Path) -> None:
    runtime, registration_ids = _runtime(tmp_path, enabled_names=["command"])

    result = _execute(
        runtime,
        registration_ids,
        "command",
        {"argv": [str(Path(sys.executable)), "--version"]},
    )

    assert result.status == "success"
    assert result.data["details"]["command_profile"] == "inspection"
    assert "process_spawn" in {effect.kind for effect in result.effects}


def test_verification_command_projects_passed_and_failed_evidence(tmp_path: Path) -> None:
    runtime, registration_ids = _runtime(tmp_path, enabled_names=["command"])
    executable = str(Path(sys.executable))

    passed = _execute(
        runtime,
        registration_ids,
        "command",
        {"argv": [executable, "-m", "pytest", "--version"]},
    )
    failed = _execute(
        runtime,
        registration_ids,
        "command",
        {"argv": [executable, "-m", "pytest", "--definitely-invalid-option"]},
    )

    assert passed.status == "success"
    assert passed.data["verification"] == {
        "status": "passed",
        "command": f"{executable} -m pytest --version",
    }
    assert failed.status == "error"
    assert failed.data["verification"] == {
        "status": "failed",
        "command": f"{executable} -m pytest --definitely-invalid-option",
    }


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["git", "diff"], "inspection"),
        (["python", "-m", "pytest"], "repository_execution"),
        (["ruff", "format", "src"], "bounded_mutation"),
        (["python", "-m", "pip", "install", "demo"], "external_effect"),
        (["git", "reset", "--hard"], "destructive"),
        (["python", "-c", "print('raw')"], "unknown"),
    ],
)
def test_controlled_command_profiles_are_capability_based(argv, expected) -> None:
    from codepilot.tools.sandbox import assess_command

    assert assess_command(argv, requires_shell=False).profile == expected


def test_workspace_write_mode_allows_bounded_work_but_keeps_escape_hatches_gated(
    tmp_path: Path,
) -> None:
    from codepilot.runtime.builder import _permission_engine

    runtime, registration_ids = _runtime(
        tmp_path,
        enabled_names=["write", "command", "bash"],
        permission_engine=_permission_engine("workspace-write"),
    )
    (tmp_path / "empty").mkdir()

    write_result = _execute(
        runtime,
        registration_ids,
        "write",
        {"path": "sample.txt", "content": "safe\n"},
    )
    command_result = _execute(
        runtime,
        registration_ids,
        "command",
        {
            "argv": [
                str(Path(sys.executable)),
                "-m",
                "compileall",
                "empty",
                "-q",
            ]
        },
    )
    external_result = _execute(
        runtime,
        registration_ids,
        "command",
        {"argv": [str(Path(sys.executable)), "-m", "pip", "install", "example"]},
    )
    bash_result = _execute(
        runtime,
        registration_ids,
        "bash",
        {"command": "echo raw-shell"},
    )

    assert write_result.status == "success"
    assert command_result.status == "approval_required"
    assert command_result.approval is not None
    assert command_result.approval.safe_preview["command_profile"] == "repository_execution"
    assert external_result.status == "approval_required"
    assert external_result.approval is not None
    assert external_result.approval.allowed_scopes == frozenset({"once"})
    assert bash_result.status == "approval_required"
    assert bash_result.approval is not None
    assert bash_result.approval.allowed_scopes == frozenset({"once"})


def test_ask_mode_and_bulk_workspace_writes_require_approval(tmp_path: Path) -> None:
    from codepilot.runtime.builder import _permission_engine

    ask_runtime, ask_ids = _runtime(
        tmp_path,
        enabled_names=["write"],
        permission_engine=_permission_engine("ask"),
    )
    ask_result = _execute(
        ask_runtime,
        ask_ids,
        "write",
        {"path": "ask.txt", "content": "approval\n"},
    )

    workspace_runtime, workspace_ids = _runtime(
        tmp_path,
        enabled_names=["write"],
        permission_engine=_permission_engine("workspace-write"),
    )
    bulk_result = _execute(
        workspace_runtime,
        workspace_ids,
        "write",
        {"path": "large.txt", "content": "x" * 500_001},
    )

    assert ask_result.status == "approval_required"
    assert ask_result.approval is not None
    assert ask_result.approval.allowed_scopes == frozenset({"once", "session", "project"})
    assert bulk_result.status == "approval_required"
    assert bulk_result.approval is not None
    assert bulk_result.approval.allowed_scopes == frozenset({"once"})
    assert not (tmp_path / "ask.txt").exists()
    assert not (tmp_path / "large.txt").exists()


def test_controlled_command_rejects_unknown_or_destructive_argv(tmp_path: Path) -> None:
    runtime, registration_ids = _runtime(tmp_path, enabled_names=["command"])

    unknown = _execute(
        runtime,
        registration_ids,
        "command",
        {"argv": [str(Path(sys.executable)), "-c", "print('not-controlled')"]},
    )
    destructive = _execute(
        runtime,
        registration_ids,
        "command",
        {"argv": ["git", "reset", "--hard"]},
    )

    assert unknown.status == "denied"
    assert unknown.error is not None
    assert unknown.error.code == "tool.access.invalid"
    assert destructive.status == "denied"
    assert destructive.error is not None
    assert destructive.error.code == "tool.access.invalid"


def test_controlled_command_rejects_workspace_escape_in_cwd_or_arguments(
    tmp_path: Path,
) -> None:
    runtime, registration_ids = _runtime(tmp_path, enabled_names=["command"])

    escaped_cwd = _execute(
        runtime,
        registration_ids,
        "command",
        {"argv": [str(Path(sys.executable)), "--version"], "cwd": ".."},
    )
    escaped_argument = _execute(
        runtime,
        registration_ids,
        "command",
        {"argv": [str(Path(sys.executable)), "--version", "../outside.txt"]},
    )

    assert escaped_cwd.status == "denied"
    assert escaped_cwd.error is not None
    assert escaped_cwd.error.code == "tool.access.invalid"
    assert escaped_argument.status == "denied"
    assert escaped_argument.error is not None
    assert escaped_argument.error.code == "tool.access.invalid"
