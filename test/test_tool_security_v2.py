from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


def test_permission_engine_matches_action_resource_effect_and_hard_constraints(tmp_path: Path) -> None:
    from codepilot.tools.builtins import create_builtin_registrations
    from codepilot.tools.contracts import ToolExecutionRequest
    from codepilot.tools.security import (
        PermissionEngine,
        PermissionRule,
        ToolAccessRequest,
        ToolResource,
    )

    registration = next(
        item
        for item in create_builtin_registrations(tmp_path, enabled_names=["write"])
        if item.spec.name == "write"
    )
    request = ToolExecutionRequest(
        run_id="run-security",
        session_id="session-security",
        tool_call_id="call-security",
        tool_name="write",
        arguments={"path": "src/app.py", "content": "ok\n"},
        mode="execute",
        registration_id="toolreg-security",
    )
    access = ToolAccessRequest(
        actions=("write",),
        resources=(ToolResource("workspace:///src/app.py"),),
        effects=frozenset({"filesystem_read", "filesystem_write"}),
        risk="medium",
        reason="write source file",
        safe_preview={"paths": ["src/app.py"]},
    )
    engine = PermissionEngine(
        rules=(
            PermissionRule("write", "workspace:///src/**", "allow", priority=10),
            PermissionRule("write", "workspace:///src/secrets/**", "deny", priority=10),
        )
    )

    allowed = engine.decide(request, registration.policy, access)
    assert allowed.effect == "allow"

    partially_covered = ToolAccessRequest(
        actions=("write", "write.metadata"),
        resources=access.resources,
        effects=access.effects,
        risk=access.risk,
        reason=access.reason,
        safe_preview=access.safe_preview,
    )
    assert engine.decide(request, registration.policy, partially_covered).effect == "ask"

    forged = ToolExecutionRequest(
        run_id=request.run_id,
        session_id=request.session_id,
        tool_call_id="call-forged",
        tool_name=request.tool_name,
        arguments={**dict(request.arguments), "bypass_approval": True},
        mode=request.mode,
        registration_id=request.registration_id,
    )
    denied = engine.decide(forged, registration.policy, access)
    assert denied.effect == "deny"
    assert denied.reason == "model_authorization_forbidden"


def test_sandbox_blocks_sensitive_files_and_dangerous_shell(tmp_path: Path) -> None:
    from codepilot.tools.sandbox import WorkspaceSandbox, validate_shell_command

    sandbox = WorkspaceSandbox(tmp_path)

    with pytest.raises(ValueError, match="Sensitive"):
        sandbox.ensure_readable_path(sandbox.resolve_path(".env"))
    with pytest.raises(ValueError, match="Sensitive"):
        sandbox.ensure_mutable_path(sandbox.resolve_path("keys/id_rsa"))
    with pytest.raises(ValueError, match="high-risk"):
        validate_shell_command("rm -rf .")


def test_sensitive_files_are_not_scanned_by_canonical_grep(tmp_path: Path) -> None:
    from codepilot.tools import create_builtin_registrations
    from codepilot.tools.registry import ToolRegistry
    from codepilot.tools.runtime import ToolRuntime

    (tmp_path / ".env").write_text("SECRET_MARKER=hidden\n", encoding="utf-8", newline="\n")
    (tmp_path / "safe.txt").write_text("public\n", encoding="utf-8", newline="\n")
    registry = ToolRegistry()
    registration = create_builtin_registrations(tmp_path, enabled_names=["grep"])[0]
    registration_id = registry.register(registration)
    runtime = ToolRuntime(registry=registry)

    result = asyncio.run(
        runtime.execute(_request("grep", registration_id, {"pattern": "SECRET_MARKER"}))
    )

    assert result.status == "success"
    assert "SECRET_MARKER" not in result.data["text"]
    assert ".env" not in result.data["text"]


def test_canonical_approval_survives_runtime_recreation_and_is_consumed_once(
    tmp_path: Path,
) -> None:
    from codepilot.tools import create_builtin_registrations
    from codepilot.tools.registry import ToolRegistry
    from codepilot.tools.runtime import ToolRuntime
    from codepilot.tools.security import ApprovalResponse
    from codepilot.sessions.tool_state_store import SessionToolStateStore

    registry = ToolRegistry()
    registration = create_builtin_registrations(tmp_path, enabled_names=["write"])[0]
    registration_id = registry.register(registration)
    store = SessionToolStateStore(tmp_path, "session-security-v2")
    runtime = ToolRuntime(
        registry=registry,
        state_store=store,
    )
    request = _request(
        "write",
        registration_id,
        {"path": "approved.txt", "content": "approved\n"},
    )

    waiting = asyncio.run(runtime.execute(request))

    assert waiting.status == "approval_required"
    assert waiting.approval is not None
    assert not (tmp_path / "approved.txt").exists()

    resumed_runtime = ToolRuntime(
        registry=registry,
        state_store=SessionToolStateStore(tmp_path, "session-security-v2"),
    )
    response = ApprovalResponse(
        approval_id=waiting.approval.approval_id,
        request_fingerprint=waiting.approval.request_fingerprint,
        decision="approve",
        scope="once",
    )
    approved = asyncio.run(resumed_runtime.resume(response))
    duplicate = asyncio.run(resumed_runtime.resume(response))

    assert approved.status == "success"
    assert (tmp_path / "approved.txt").read_text(encoding="utf-8") == "approved\n"
    assert duplicate.status == "error"
    assert duplicate.error is not None
    assert duplicate.error.code == "tool.approval.consumed"


def test_canonical_approval_is_atomically_consumed_across_runtimes(tmp_path: Path) -> None:
    from dataclasses import replace

    from codepilot.sessions.tool_state_store import SessionToolStateStore
    from codepilot.tools import create_builtin_registrations
    from codepilot.tools.registry import ToolRegistry
    from codepilot.tools.runtime import ToolRuntime
    from codepilot.tools.security import ApprovalResponse

    calls = 0
    registration = create_builtin_registrations(tmp_path, enabled_names=["write"])[0]
    original_handler = registration.handler

    async def counted_handler(input, context):
        nonlocal calls
        calls += 1
        return await original_handler(input, context)

    registry = ToolRegistry()
    registration_id = registry.register(replace(registration, handler=counted_handler))
    waiting = asyncio.run(
        ToolRuntime(
            registry=registry,
            state_store=SessionToolStateStore(tmp_path, "session-security-v2"),
        ).execute(
            _request(
                "write",
                registration_id,
                {"path": "atomic.txt", "content": "once\n"},
                tool_call_id="call-atomic-approval",
            )
        )
    )
    response = ApprovalResponse(
        approval_id=waiting.approval.approval_id,
        request_fingerprint=waiting.approval.request_fingerprint,
        decision="approve",
        scope="once",
    )

    async def consume_twice():
        first = ToolRuntime(
            registry=registry,
            state_store=SessionToolStateStore(tmp_path, "session-security-v2"),
        )
        second = ToolRuntime(
            registry=registry,
            state_store=SessionToolStateStore(tmp_path, "session-security-v2"),
        )
        return await asyncio.gather(first.resume(response), second.resume(response))

    results = asyncio.run(consume_twice())

    assert sorted(result.status for result in results) == ["error", "success"]
    assert calls == 1
    assert (tmp_path / "atomic.txt").read_text(encoding="utf-8") == "once\n"


def test_canonical_approval_rejects_fingerprint_mismatch(tmp_path: Path) -> None:
    from codepilot.tools import create_builtin_registrations
    from codepilot.tools.registry import ToolRegistry
    from codepilot.tools.runtime import ToolRuntime
    from codepilot.tools.security import ApprovalResponse

    registry = ToolRegistry()
    registration = create_builtin_registrations(tmp_path, enabled_names=["write"])[0]
    registration_id = registry.register(registration)
    runtime = ToolRuntime(registry=registry)
    waiting = asyncio.run(
        runtime.execute(
            _request(
                "write",
                registration_id,
                {"path": "blocked.txt", "content": "blocked\n"},
            )
        )
    )

    result = asyncio.run(
        runtime.resume(
            ApprovalResponse(
                approval_id=waiting.approval.approval_id,
                request_fingerprint="sha256:wrong",
                decision="approve",
                scope="once",
            )
        )
    )

    assert result.status == "denied"
    assert result.error is not None
    assert result.error.code == "tool.approval.fingerprint_mismatch"
    assert not (tmp_path / "blocked.txt").exists()


def test_session_approval_grant_reuses_same_resource_without_second_prompt(
    tmp_path: Path,
) -> None:
    from codepilot.tools import create_builtin_registrations
    from codepilot.tools.registry import ToolRegistry
    from codepilot.tools.runtime import ToolRuntime
    from codepilot.tools.security import ApprovalResponse

    registry = ToolRegistry()
    registration = create_builtin_registrations(tmp_path, enabled_names=["write"])[0]
    registration_id = registry.register(registration)
    runtime = ToolRuntime(registry=registry)
    first_request = _request(
        "write",
        registration_id,
        {"path": "session.txt", "content": "first\n"},
        tool_call_id="call-session-first",
    )
    waiting = asyncio.run(runtime.execute(first_request))
    approved = asyncio.run(
        runtime.resume(
            ApprovalResponse(
                approval_id=waiting.approval.approval_id,
                request_fingerprint=waiting.approval.request_fingerprint,
                decision="approve",
                scope="session",
            )
        )
    )
    second = asyncio.run(
        runtime.execute(
            _request(
                "write",
                registration_id,
                {"path": "session.txt", "content": "second\n"},
                tool_call_id="call-session-second",
            )
        )
    )

    assert approved.status == "success"
    assert second.status == "success"
    assert (tmp_path / "session.txt").read_text(encoding="utf-8") == "second\n"


def test_gateway_recovers_pending_approval_from_canonical_tool_state(tmp_path: Path) -> None:
    from codepilot.llm.ports import LLMCompleted
    from codepilot.protocols import AssistantMessage, TextContent, ToolCall
    from codepilot.runtime import SessionOpenIntent
    from codepilot.runtime.actions import (
        ApprovalDecided,
        ApprovalRequiredFrame,
        PromptSubmitted,
        RunFinishedFrame,
    )
    from codepilot.runtime.gateway import RuntimeGateway

    class Model:
        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request):
            _ = request
            self.calls += 1
            if self.calls == 1:
                yield LLMCompleted(
                    message=AssistantMessage(
                        content=[
                            ToolCall(
                                id="call-persistent-write",
                                name="write",
                                arguments={
                                    "path": "gateway-approved.txt",
                                    "content": "approved through gateway\n",
                                },
                            )
                        ],
                        stop_reason="toolUse",
                    )
                )
                return
            yield LLMCompleted(
                message=AssistantMessage(content=[TextContent(text="gateway resume complete")])
            )

    async def run_case() -> None:
        model = Model()
        first = RuntimeGateway(model_port=model)
        ref = first.open_session(
            SessionOpenIntent(
                workspace_dir=tmp_path,
                provider="deepseek",
                model_id="deepseek-v4-pro",
                session_id="session-gateway-persistent",
                memory_enabled=False,
                load_workspace_resources=False,
                tool_permission_mode="ask",
            )
        )
        paused = [
            frame
            async for frame in first.dispatch(
                ref.session_id,
                PromptSubmitted(text="write the approved file"),
            )
        ]
        approval = next(
            frame.approval for frame in paused if isinstance(frame, ApprovalRequiredFrame)
        )
        first.close(ref.session_id)

        second = RuntimeGateway(model_port=model)
        second.open_session(
            SessionOpenIntent(
                workspace_dir=tmp_path,
                provider="deepseek",
                model_id="deepseek-v4-pro",
                session_id=ref.session_id,
                memory_enabled=False,
                load_workspace_resources=False,
                tool_permission_mode="ask",
            )
        )
        assert [item.approval_id for item in second.describe(ref.session_id).pending_approvals] == [
            approval.approval_id
        ]
        resumed = [
            frame
            async for frame in second.dispatch(
                ref.session_id,
                ApprovalDecided(
                    approval_id=approval.approval_id,
                    decision="approve",
                    reason="approved",
                ),
            )
        ]
        assert any(isinstance(frame, RunFinishedFrame) for frame in resumed), resumed
        second.close(ref.session_id)

    asyncio.run(run_case())

    assert (tmp_path / "gateway-approved.txt").read_text(encoding="utf-8") == (
        "approved through gateway\n"
    )


def _request(
    name: str,
    registration_id: str,
    arguments: dict[str, object],
    *,
    tool_call_id: str | None = None,
):
    from codepilot.tools.contracts import ToolExecutionRequest

    return ToolExecutionRequest(
        run_id="run-security-v2",
        session_id="session-security-v2",
        tool_call_id=tool_call_id or f"call-{name}",
        tool_name=name,
        arguments=arguments,
        mode="execute",
        registration_id=registration_id,
    )
