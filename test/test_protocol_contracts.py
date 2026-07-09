from __future__ import annotations

import pytest


def test_error_info_normalizes_and_validates_cross_layer_fields() -> None:
    from codepilot.protocols import ErrorInfo, LLMErrorInfo

    error = ErrorInfo(
        code=" runtime.session_busy ",
        message=" Session is already running ",
        source=" runtime ",
        details={"session_id": "s1"},
    )

    assert error.code == "runtime.session_busy"
    assert error.message == "Session is already running"
    assert error.source == "runtime"
    assert error.details == {"session_id": "s1"}

    with pytest.raises(ValueError, match="error code"):
        ErrorInfo(code=" ", message="missing code")

    with pytest.raises(ValueError, match="error message"):
        ErrorInfo(code="runtime.error", message="")

    with pytest.raises(ValueError, match="error source"):
        ErrorInfo(code="runtime.error", message="bad source", source="database")

    llm_error = LLMErrorInfo(
        code=" llm.timeout ",
        message=" Request timed out ",
        kind=" timeout ",
        provider=" deepseek ",
        model=" deepseek-chat ",
    )

    assert llm_error.code == "llm.timeout"
    assert llm_error.message == "Request timed out"
    assert llm_error.source == "llm"
    assert llm_error.kind == "timeout"
    assert llm_error.provider == "deepseek"
    assert llm_error.model == "deepseek-chat"

    with pytest.raises(ValueError, match="LLM error kind"):
        LLMErrorInfo(code="llm.bad", message="bad kind", kind="quota")

    with pytest.raises(ValueError, match="LLM error source"):
        LLMErrorInfo(
            code="llm.timeout",
            message="wrong source",
            kind="timeout",
            source="runtime",
        )


def test_tool_spec_normalizes_and_validates_provider_visible_fields() -> None:
    from codepilot.protocols import Tool

    parameters = {"type": "object", "properties": {"path": {"type": "string"}}}
    tool = Tool(
        name=" read ",
        description=" Read a workspace file. ",
        parameters=parameters,
    )
    parameters["properties"] = {}

    assert tool.name == "read"
    assert tool.description == "Read a workspace file."
    assert tool.parameters == {"type": "object", "properties": {"path": {"type": "string"}}}

    with pytest.raises(ValueError, match="tool name"):
        Tool(name="", description="Read", parameters={})

    with pytest.raises(ValueError, match="description"):
        Tool(name="read", description="", parameters={})

    with pytest.raises(TypeError, match="parameters"):
        Tool(name="read", description="Read", parameters=[])  # type: ignore[arg-type]


def test_tool_metadata_normalizes_and_validates_permission_facts() -> None:
    from codepilot.tools.contracts import ToolMetadata

    extra = {"capabilities": ["filesystem.read"]}
    metadata = ToolMetadata(
        name=" read ",
        category=" filesystem ",
        read_only=True,
        concurrency_safe=True,
        exclusive=False,
        requires_approval=False,
        risk_level=" low ",
        scopes=(" read ", "build", "read"),
        network_access=False,
        credential_required=False,
        extra=extra,
    )
    extra["capabilities"] = ["mutated"]

    assert metadata.name == "read"
    assert metadata.category == "filesystem"
    assert metadata.risk_level == "low"
    assert metadata.scopes == ("read", "build")
    assert metadata.extra == {"capabilities": ["filesystem.read"]}

    with pytest.raises(ValueError, match="metadata.name"):
        ToolMetadata(
            name="",
            category="filesystem",
            read_only=True,
            concurrency_safe=True,
            exclusive=False,
            requires_approval=False,
            risk_level="low",
            scopes=("read",),
        )

    with pytest.raises(ValueError, match="risk level"):
        ToolMetadata(
            name="read",
            category="filesystem",
            read_only=True,
            concurrency_safe=True,
            exclusive=False,
            requires_approval=False,
            risk_level="critical",
            scopes=("read",),
        )

    with pytest.raises(TypeError, match="read_only"):
        ToolMetadata(
            name="read",
            category="filesystem",
            read_only="yes",  # type: ignore[arg-type]
            concurrency_safe=True,
            exclusive=False,
            requires_approval=False,
            risk_level="low",
            scopes=("read",),
        )

    with pytest.raises(ValueError, match="scopes"):
        ToolMetadata(
            name="read",
            category="filesystem",
            read_only=True,
            concurrency_safe=True,
            exclusive=False,
            requires_approval=False,
            risk_level="low",
            scopes=(),
        )


def test_model_contract_normalizes_and_validates_runtime_fields() -> None:
    from codepilot.protocols import Model, ModelCapabilities

    headers = {" X-Test ": " value "}
    model = Model(
        id=" demo-model ",
        name=" Demo Model ",
        api=" openai-compatible ",
        provider=" demo ",
        base_url=" https://api.example.test ",
        reasoning=True,
        input=[" text ", "image"],
        context_window=128_000,
        max_tokens=8192,
        headers=headers,
    )
    headers["X-Test"] = "mutated"

    assert model.id == "demo-model"
    assert model.name == "Demo Model"
    assert model.api == "openai-compatible"
    assert model.provider == "demo"
    assert model.base_url == "https://api.example.test"
    assert model.input == ["text", "image"]
    assert model.headers == {"X-Test": "value"}
    assert model.capabilities is not None
    assert model.capabilities.vision is True
    assert model.capabilities.reasoning is True

    provider_default_model = Model(
        id="provider-default",
        name="Provider Default",
        api="openai-compatible",
        provider="demo",
        base_url=" ",
        reasoning=False,
        input=["text"],
        context_window=128_000,
        max_tokens=8192,
    )

    assert provider_default_model.base_url == ""

    with pytest.raises(TypeError, match="tools"):
        ModelCapabilities(tools="yes")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="capabilities"):
        Model(
            id="demo",
            name="Demo",
            api="openai-compatible",
            provider="demo",
            base_url="https://api.example.test",
            reasoning=False,
            input=["text"],
            context_window=128_000,
            max_tokens=8192,
            capabilities="tools",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="model id"):
        Model(
            id="",
            name="Demo",
            api="openai-compatible",
            provider="demo",
            base_url="https://api.example.test",
            reasoning=False,
            input=["text"],
            context_window=128_000,
            max_tokens=8192,
        )

    with pytest.raises(ValueError, match="input"):
        Model(
            id="demo",
            name="Demo",
            api="openai-compatible",
            provider="demo",
            base_url="https://api.example.test",
            reasoning=False,
            input=["audio"],  # type: ignore[list-item]
            context_window=128_000,
            max_tokens=8192,
        )

    with pytest.raises(ValueError, match="context_window"):
        Model(
            id="demo",
            name="Demo",
            api="openai-compatible",
            provider="demo",
            base_url="https://api.example.test",
            reasoning=False,
            input=["text"],
            context_window=0,
            max_tokens=8192,
        )

    with pytest.raises(TypeError, match="headers"):
        Model(
            id="demo",
            name="Demo",
            api="openai-compatible",
            provider="demo",
            base_url="https://api.example.test",
            reasoning=False,
            input=["text"],
            context_window=128_000,
            max_tokens=8192,
            headers=[],
        )  # type: ignore[arg-type]


def test_usage_and_cost_own_non_negative_accounting_invariants() -> None:
    from codepilot.protocols import Cost, Usage

    cost = Cost(input=0.5, output=0.25)

    assert cost.total == 0.75

    provider_total_only = Cost(total=0.01)

    assert provider_total_only.total == 0.01

    usage = Usage(input=3, output=4, cache_read=1, cache_write=2)

    assert usage.total_tokens == 10
    assert isinstance(usage.cost, Cost)

    provider_total_usage = Usage(total_tokens=9, cost=Cost(total=0.02))

    assert provider_total_usage.total_tokens == 9
    assert provider_total_usage.cost.total == 0.02

    with pytest.raises(ValueError, match="input"):
        Usage(input=-1)

    with pytest.raises(TypeError, match="output"):
        Usage(output=1.5)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="total_tokens"):
        Usage(total_tokens=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="cost input"):
        Cost(input=-0.1)

    with pytest.raises(TypeError, match="cost"):
        Usage(cost={"total": 0.1})  # type: ignore[arg-type]


def test_run_result_models_normalize_and_validate_run_facts() -> None:
    from codepilot.protocols import (
        AgentRunCounters,
        AgentRunResult,
        AssistantMessage,
        ErrorInfo,
        PlanSummary,
        RunSignalsSummary,
        RunVerification,
        TextContent,
        UserMessage,
    )

    counters = AgentRunCounters(model_attempts=1, tool_iterations=2, tool_calls=3)

    assert counters.model_attempts == 1
    assert counters.tool_iterations == 2
    assert counters.tool_calls == 3

    with pytest.raises(ValueError, match="model_attempts"):
        AgentRunCounters(model_attempts=-1)

    with pytest.raises(TypeError, match="tool_calls"):
        AgentRunCounters(tool_calls=True)  # type: ignore[arg-type]

    verification = RunVerification(
        tool_call_id=" call-1 ",
        tool_name=" shell ",
        status=" passed ",
        command=" pytest -q ",
        exit_code=0,
        summary=" Tests passed. ",
    )

    assert verification.tool_call_id == "call-1"
    assert verification.tool_name == "shell"
    assert verification.status == "passed"
    assert verification.command == "pytest -q"
    assert verification.summary == "Tests passed."

    with pytest.raises(ValueError, match="tool_call_id"):
        RunVerification(tool_call_id="", tool_name="shell", status="passed")

    with pytest.raises(ValueError, match="verification status"):
        RunVerification(tool_call_id="call-1", tool_name="shell", status="skipped")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="exit_code"):
        RunVerification(tool_call_id="call-1", tool_name="shell", status="passed", exit_code=False)  # type: ignore[arg-type]

        plan = PlanSummary(
            schema_version=2,
            plan_id=" plan-1 ",
            owner_run_id=" run-1 ",
            status=" active ",
            approval_state=" approved ",
            origin_mode=" build ",
            objective=" Refactor run model. ",
            summary=" Read the code and implement the refactor. ",
            items=[
                {
                    "id": " item-1 ",
                    "step": " Read code. ",
                    "details": " Inspect the implementation. ",
                    "verification": " Confirm the call path. ",
                    "status": " completed ",
                }
            ],
            revision=1,
        explanation=" Write tests. ",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    signals = RunSignalsSummary(
        workspace_changed=True,
        affected_paths=[" src/a.py ", "src/a.py"],
        verification_status="stale",
        last_error={"error_code": "verification_failed"},
        counters=counters,
    )

    assert plan.plan_id == "plan-1"
    assert plan.status == "active"
    assert plan.items == [{"id": "item-1", "step": "Read code.", "status": "completed"}]
    assert signals.affected_paths == ["src/a.py"]
    assert signals.verification_status == "stale"

    with pytest.raises(ValueError, match="plan_id"):
        PlanSummary(
            schema_version=1,
            plan_id="",
            status="active",
            approval_state="approved",
            origin_mode="build",
            objective="Goal",
        )

    with pytest.raises(ValueError, match="run signal verification"):
        RunSignalsSummary(verification_status="skipped")  # type: ignore[arg-type]

    messages = [UserMessage(content="hello")]
    affected_paths = [" src/a.py ", "", "src/a.py", "src/b.py"]
    final_message = AssistantMessage(content=[TextContent(text="done")])
    error = ErrorInfo(code="runtime.error", message="failed")
    result = AgentRunResult(
        run_id=" run-1 ",
        session_id=" session-1 ",
        status=" completed ",
        stop_reason=" final_answer ",
        counters=counters,
        messages=messages,
        final_message=final_message,
        error=error,
        affected_paths=affected_paths,
        workspace_changed=True,
        verification=[verification],
        plan=plan,
        signals=signals,
    )
    messages.append(UserMessage(content="mutated"))
    affected_paths.append("src/c.py")

    assert result.run_id == "run-1"
    assert result.session_id == "session-1"
    assert result.status == "completed"
    assert result.stop_reason == "final_answer"
    assert result.messages == [UserMessage(content="hello")]
    assert result.affected_paths == ["src/a.py", "src/b.py"]
    assert result.workspace_changed is True
    assert result.final_message is final_message
    assert result.error is error
    assert result.plan is plan
    assert result.signals is signals

    with pytest.raises(ValueError, match="run_id"):
        AgentRunResult(run_id="", session_id=None, status="completed", stop_reason="final_answer")

    with pytest.raises(ValueError, match="run status"):
        AgentRunResult(run_id="run-1", session_id=None, status="done", stop_reason="final_answer")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="counters"):
        AgentRunResult(
            run_id="run-1",
            session_id=None,
            status="completed",
            stop_reason="final_answer",
            counters={"tool_calls": 1},  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="workspace_changed"):
        AgentRunResult(
            run_id="run-1",
            session_id=None,
            status="completed",
            stop_reason="final_answer",
            workspace_changed="yes",  # type: ignore[arg-type]
        )


def test_runtime_event_type_contract_rejects_unknown_events() -> None:
    from codepilot.protocols import ensure_runtime_event_type

    assert ensure_runtime_event_type(" tool_completed ") == "tool_completed"
    assert ensure_runtime_event_type("context_projected") == "context_projected"
    assert ensure_runtime_event_type("context_compacted") == "context_compacted"
    assert ensure_runtime_event_type("memory_record_approved") == "memory_record_approved"
    assert ensure_runtime_event_type("tool_interrupted") == "tool_interrupted"

    with pytest.raises(ValueError, match="runtime event type"):
        ensure_runtime_event_type("context_prepared")

    with pytest.raises(ValueError, match="runtime event type"):
        ensure_runtime_event_type("memory_updated")

    with pytest.raises(ValueError, match="runtime event type"):
        ensure_runtime_event_type("tool_finished")
