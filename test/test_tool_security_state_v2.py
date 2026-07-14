from pathlib import Path

import pytest
def _policy(*, effects, permissions):
    from codepilot.tools.security import (
        ConcurrencyPolicy,
        OutputLimits,
        OutputTrustPolicy,
        TimeoutPolicy,
        ToolPolicy,
    )

    return ToolPolicy(
        allowed_modes=frozenset({"plan", "execute"}),
        declared_effects=frozenset(effects),
        required_permissions=frozenset(permissions),
        base_risk="low",
        approval="never",
        timeout=TimeoutPolicy(1_000, 1_000),
        concurrency=ConcurrencyPolicy("parallel"),
        output_limits=OutputLimits(),
        output_trust=OutputTrustPolicy(),
    )


def _access(*, effects, risk="low"):
    from codepilot.tools.security import ToolAccessRequest, ToolResource

    return ToolAccessRequest(
        actions=("mcp.call",),
        resources=(ToolResource("mcp://demo/read"),),
        effects=frozenset(effects),
        risk=risk,
        reason="test access",
    )


def _request(*, tool_call_id="call_1"):
    from codepilot.tools.contracts import ToolExecutionRequest

    return ToolExecutionRequest(
        run_id="run_1",
        session_id="session_1",
        tool_call_id=tool_call_id,
        tool_name="demo",
        arguments={},
        mode="execute",
        registration_id="reg_1",
    )


def test_tool_attempt_state_machine_rejects_invalid_transition() -> None:
    from codepilot.tools.state import ToolAttemptRecord, transition

    record = ToolAttemptRecord(
        attempt_id="session_1:run_1:call_1",
        request=_request(),
    )

    with pytest.raises(ValueError, match="received -> running"):
        transition(record, "running")


def test_permission_engine_enforces_required_permissions_and_sensitive_reads() -> None:
    from codepilot.runtime.builder import _permission_engine
    from codepilot.tools.security import PermissionEngine

    access = _access(effects={"network_access", "external_state_read"})
    policy = _policy(
        effects={"network_access", "external_state_read"},
        permissions={"mcp.call"},
    )
    request = _request()

    assert _permission_engine("read-only").decide(request, policy, access).effect == "ask"
    assert _permission_engine("ask").decide(request, policy, access).effect == "ask"
    assert _permission_engine("workspace-write").decide(request, policy, access).effect == "ask"

    missing = PermissionEngine(granted_permissions=frozenset({"workspace.read"})).decide(
        request,
        policy,
        access,
    )
    assert missing.effect == "deny"
    assert missing.reason == "required_permission_missing"
    assert missing.details["missing_permissions"] == ("mcp.call",)

    credential_policy = _policy(
        effects={"credential_access"},
        permissions={"mcp.call", "credential:demo"},
    )
    credential_access = _access(effects={"credential_access"})
    assert (
        _permission_engine("read-only")
        .decide(request, credential_policy, credential_access)
        .effect
        == "deny"
    )


def test_approval_grant_cannot_be_reused_for_higher_risk_or_new_effect() -> None:
    from codepilot.tools.security import ApprovalResponse, build_approval_challenge, issue_approval_grant
    from codepilot.tools.state import InMemoryToolStateStore, ToolAttemptRecord

    store = InMemoryToolStateStore()
    request = _request()
    approved_access = _access(effects={"external_state_read"}, risk="low")
    challenge = build_approval_challenge(request, approved_access, reason="test")
    grant = issue_approval_grant(
        challenge,
        ApprovalResponse(
            approval_id=challenge.approval_id,
            request_fingerprint=challenge.request_fingerprint,
            decision="approve",
            scope="session",
        ),
    )
    store.create(
        ToolAttemptRecord(
            attempt_id="attempt_1",
            request=request,
            state="succeeded",
            grant=grant,
        )
    )

    assert store.find_reusable_grant(request, approved_access) == grant
    assert store.find_reusable_grant(
        request,
        _access(effects={"external_state_read"}, risk="high"),
    ) is None
    assert store.find_reusable_grant(
        request,
        _access(effects={"external_state_read", "network_access"}, risk="low"),
    ) is None


def test_checkpoint_tool_state_store_round_trips_pending_approval_strictly() -> None:
    from codepilot.tools.state_store import CheckpointToolStateStore
    from codepilot.tools.security import build_approval_challenge
    from codepilot.tools.state import ToolAttemptRecord

    request = _request()
    challenge = build_approval_challenge(
        request,
        _access(effects={"external_state_read"}),
        reason="persist approval",
    )
    store = CheckpointToolStateStore(session_id=request.session_id)
    store.create(
        ToolAttemptRecord(
            attempt_id="attempt_pending",
            request=request,
            state="awaiting_approval",
            challenge=challenge,
        )
    )

    snapshot = store.checkpoint_state()
    assert snapshot is not None
    store.restore_checkpoint_state(snapshot)
    assert store.pending_challenges() == (challenge,)
    reopened = CheckpointToolStateStore(session_id=request.session_id)
    reopened.restore_checkpoint_state(snapshot)
    assert reopened.pending_challenges() == (challenge,)
    assert reopened.find_by_approval_id(challenge.approval_id).attempt_id == "attempt_pending"

    invalid = dict(snapshot)
    invalid.pop("schema_version")
    try:
        CheckpointToolStateStore(session_id=request.session_id).restore_checkpoint_state(invalid)
    except ValueError as exc:
        assert "unknown or missing" in str(exc)
    else:
        raise AssertionError("schema-less tool checkpoint must be rejected")


def test_checkpoint_tool_state_store_excludes_terminal_attempts() -> None:
    from codepilot.tools.state_store import CheckpointToolStateStore
    from codepilot.tools.results import TextContent, ToolResult
    from codepilot.tools.state import ToolAttemptRecord

    request = _request(tool_call_id="terminal")
    result = ToolResult(
        tool_call_id=request.tool_call_id,
        tool_name=request.tool_name,
        status="success",
        content=(TextContent("done"),),
        data={"value": 1},
        registration_id=request.registration_id,
    )
    store = CheckpointToolStateStore(session_id=request.session_id)
    store.create(
        ToolAttemptRecord(
            attempt_id="attempt_terminal",
            request=request,
            state="succeeded",
            result=result,
        )
    )

    assert store.get("attempt_terminal").result == result
    assert store.checkpoint_state() is None


def test_project_approval_grant_is_reusable_across_sessions(tmp_path: Path) -> None:
    from dataclasses import replace

    from codepilot.tools.security import ApprovalResponse, build_approval_challenge, issue_approval_grant
    from codepilot.tools.state import ToolAttemptRecord
    from codepilot.tools.state_store import CheckpointToolStateStore, FileToolGrantStore

    first_request = _request()
    access = _access(effects={"external_state_read"})
    challenge = build_approval_challenge(first_request, access, reason="project grant")
    grant = issue_approval_grant(
        challenge,
        ApprovalResponse(
            approval_id=challenge.approval_id,
            request_fingerprint=challenge.request_fingerprint,
            decision="approve",
            scope="project",
        ),
    )
    grant_path = tmp_path / ".codepilot" / "tool_grants.json"
    first = CheckpointToolStateStore(
        session_id=first_request.session_id,
        grant_store=FileToolGrantStore(grant_path),
    )
    first.create(
        ToolAttemptRecord(
            attempt_id="project_grant_attempt",
            request=first_request,
            state="awaiting_approval",
            challenge=challenge,
        )
    )
    first.compare_and_set(
        "project_grant_attempt",
        "awaiting_approval",
        ToolAttemptRecord(
            attempt_id="project_grant_attempt",
            request=first_request,
            state="succeeded",
            challenge=challenge,
            grant=grant,
        ),
    )

    second_request = replace(
        first_request,
        session_id="session_2",
        run_id="run_2",
        tool_call_id="call_2",
    )
    second = CheckpointToolStateStore(
        session_id="session_2",
        grant_store=FileToolGrantStore(grant_path),
    )
    assert second.find_reusable_grant(second_request, access) == grant


def test_session_approval_grant_is_reusable_only_in_own_session(tmp_path: Path) -> None:
    from dataclasses import replace

    from codepilot.tools.security import ApprovalResponse, build_approval_challenge, issue_approval_grant
    from codepilot.tools.state import ToolAttemptRecord
    from codepilot.tools.state_store import CheckpointToolStateStore, FileToolGrantStore

    request = _request()
    access = _access(effects={"external_state_read"})
    challenge = build_approval_challenge(request, access, reason="session grant")
    grant = issue_approval_grant(
        challenge,
        ApprovalResponse(
            approval_id=challenge.approval_id,
            request_fingerprint=challenge.request_fingerprint,
            decision="approve",
            scope="session",
        ),
    )
    grant_path = tmp_path / ".codepilot" / "tool_grants.json"
    first = CheckpointToolStateStore(
        session_id=request.session_id,
        grant_store=FileToolGrantStore(grant_path),
    )
    first.create(
        ToolAttemptRecord(
            attempt_id="session_grant_attempt",
            request=request,
            state="awaiting_approval",
            challenge=challenge,
        )
    )
    first.compare_and_set(
        "session_grant_attempt",
        "awaiting_approval",
        ToolAttemptRecord(
            attempt_id="session_grant_attempt",
            request=request,
            state="succeeded",
            challenge=challenge,
            grant=grant,
        ),
    )

    reopened = CheckpointToolStateStore(
        session_id=request.session_id,
        grant_store=FileToolGrantStore(grant_path),
    )
    assert reopened.find_reusable_grant(request, access) == grant

    other_request = replace(request, session_id="session_2", run_id="run_2")
    other = CheckpointToolStateStore(
        session_id="session_2",
        grant_store=FileToolGrantStore(grant_path),
    )
    assert other.find_reusable_grant(other_request, access) is None
