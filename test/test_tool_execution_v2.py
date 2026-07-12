from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest


def test_execution_timeout_preserves_effects_and_runs_cleanup_lifo() -> None:
    from codepilot.tools.execution import ExecutionController
    from codepilot.tools.state import InMemoryToolStateStore, attempt_id_for

    cleanup_order: list[str] = []

    def broken_cleanup() -> None:
        cleanup_order.append("broken")
        raise RuntimeError("cleanup failed")

    async def handler(input, context):
        from codepilot.tools.security import ToolEffect, ToolResource

        _ = input
        context.cleanup.push(lambda: cleanup_order.append("first"))
        context.cleanup.push(broken_cleanup)
        context.cleanup.push(lambda: cleanup_order.append("second"))
        context.effects.report(
            ToolEffect(
                kind="session_state_write",
                resource=ToolResource("session://execution-test"),
                operation="start timeout test",
                status="started",
                certainty="observed",
            )
        )
        await asyncio.sleep(0.2)
        return SampleOutput("late")

    store = InMemoryToolStateStore()
    runtime, registration_id = _runtime(
        _registration("slow", handler, timeout_ms=30),
        store=store,
        controller=ExecutionController(),
    )
    request = _request("slow", registration_id, "timeout")

    result = asyncio.run(runtime.execute(request))

    assert result.status == "timed_out"
    assert result.error is not None
    assert result.error.code == "tool.execution.timeout"
    assert {effect.kind for effect in result.effects} == {"session_state_write"}
    assert cleanup_order == ["second", "broken", "first"]
    record = store.get(attempt_id_for(request))
    assert record is not None
    assert record.state == "timed_out"
    assert record.cleanup_errors == ("RuntimeError",)


def test_runtime_cancel_sets_token_and_runs_cleanup() -> None:
    from codepilot.tools.execution import ExecutionController
    from codepilot.tools.state import attempt_id_for

    started = asyncio.Event()
    cleaned: list[str] = []

    async def handler(input, context):
        _ = input
        context.cleanup.push(lambda: cleaned.append("done"))
        started.set()
        while True:
            context.cancellation.raise_if_cancelled()
            await asyncio.sleep(0.005)

    async def run_case():
        runtime, registration_id = _runtime(
            _registration("cancel_me", handler, timeout_ms=1_000),
            controller=ExecutionController(),
        )
        request = _request("cancel_me", registration_id, "cancel")
        task = asyncio.create_task(runtime.execute(request))
        await started.wait()
        assert await runtime.cancel(attempt_id_for(request)) is True
        return await task

    result = asyncio.run(run_case())

    assert result.status == "cancelled"
    assert result.error is not None
    assert result.error.code == "tool.execution.cancelled"
    assert cleaned == ["done"]


def test_requested_timeout_can_extend_default_up_to_policy_maximum() -> None:
    from codepilot.tools.execution import ExecutionController

    async def handler(input, context):
        _ = input, context
        await asyncio.sleep(0.08)
        return SampleOutput("finished")

    runtime, registration_id = _runtime(
        _registration(
            "requested_timeout",
            handler,
            timeout_ms=20,
            max_timeout_ms=200,
            requested_timeout_ms=120,
        ),
        controller=ExecutionController(),
    )

    result = asyncio.run(
        runtime.execute(_request("requested_timeout", registration_id, "run"))
    )

    assert result.status == "success"


def test_execute_batch_runs_parallel_tools_and_preserves_input_order() -> None:
    from codepilot.tools.execution import ExecutionController, ToolRuntimeLimits

    active = 0
    max_active = 0

    async def handler(input, context):
        nonlocal active, max_active
        _ = context
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(input.delay_ms / 1_000)
        active -= 1
        return SampleOutput(input.label)

    runtime, ids = _multi_runtime(
        (
            _registration("parallel_a", handler, concurrency="parallel"),
            _registration("parallel_b", handler, concurrency="parallel"),
        ),
        controller=ExecutionController(ToolRuntimeLimits(max_parallel_per_session=2)),
    )
    requests = (
        _request("parallel_a", ids["parallel_a"], "first", delay_ms=40),
        _request("parallel_b", ids["parallel_b"], "second", delay_ms=5),
    )

    results = asyncio.run(runtime.execute_batch(requests))

    assert max_active == 2
    assert [result.data["label"] for result in results] == ["first", "second"]


def test_execute_batch_serializes_same_group_and_stops_at_approval_barrier() -> None:
    from codepilot.tools.execution import ExecutionController

    active = 0
    max_active = 0
    after_barrier_called = False

    async def serial_handler(input, context):
        nonlocal active, max_active
        _ = context
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return SampleOutput(input.label)

    async def after_barrier(input, context):
        nonlocal after_barrier_called
        _ = input, context
        after_barrier_called = True
        return SampleOutput("unexpected")

    runtime, ids = _multi_runtime(
        (
            _registration("serial_a", serial_handler, concurrency="serial", group="workspace"),
            _registration("serial_b", serial_handler, concurrency="serial", group="workspace"),
        ),
        controller=ExecutionController(),
    )
    serial_results = asyncio.run(
        runtime.execute_batch(
            (
                _request("serial_a", ids["serial_a"], "a"),
                _request("serial_b", ids["serial_b"], "b"),
            )
        )
    )
    assert [result.data["label"] for result in serial_results] == ["a", "b"]
    assert max_active == 1

    barrier_runtime, barrier_ids = _multi_runtime(
        (
            _registration("needs_approval", serial_handler, approval="always"),
            _registration("after_barrier", after_barrier),
        ),
        controller=ExecutionController(),
    )
    barrier_results = asyncio.run(
        barrier_runtime.execute_batch(
            (
                _request("needs_approval", barrier_ids["needs_approval"], "wait"),
                _request("after_barrier", barrier_ids["after_barrier"], "later"),
            )
        )
    )

    assert [result.status for result in barrier_results] == [
        "approval_required",
        "interrupted",
    ]
    assert barrier_results[1].error.code == "tool.batch.interrupted"
    assert after_barrier_called is False


def test_core_delegates_tool_turn_batch_to_tool_port() -> None:
    from codepilot.core.tool_step import execute_tool_turn
    from codepilot.protocols import ToolCall
    from codepilot.tools.contracts import ToolSpec
    from codepilot.tools.registry import ToolCatalogEntry, ToolCatalogSnapshot
    from codepilot.tools.results import TextContent, ToolResult
    from codepilot.tools.security import (
        ConcurrencyPolicy,
        OutputLimits,
        OutputTrustPolicy,
        TimeoutPolicy,
        ToolPolicy,
    )

    policy = ToolPolicy(
        allowed_modes=frozenset({"execute"}),
        declared_effects=frozenset({"session_state_read"}),
        required_permissions=frozenset(),
        base_risk="low",
        approval="never",
        timeout=TimeoutPolicy(1_000, 1_000),
        concurrency=ConcurrencyPolicy("parallel"),
        output_limits=OutputLimits(),
        output_trust=OutputTrustPolicy(),
    )
    entries = tuple(
        ToolCatalogEntry(
            spec=ToolSpec(name, name, {"type": "object", "additionalProperties": False}),
            category="external",
            source="builtin",
            policy=policy,
            registration_id=f"reg-{name}",
            version="1.0.0",
            owner="test",
        )
        for name in ("a", "b")
    )
    snapshot = ToolCatalogSnapshot("catalog-test", entries, 1)

    class BatchPort:
        def __init__(self) -> None:
            self.batch_calls = 0

        def catalog_snapshot(self, *, mode=None):
            _ = mode
            return snapshot

        async def execute(self, request):
            raise AssertionError("Core must not schedule individual tool calls")

        async def execute_batch(self, requests):
            self.batch_calls += 1
            return [
                ToolResult(
                    tool_call_id=item.tool_call_id,
                    tool_name=item.tool_name,
                    status="success",
                    content=(TextContent(text=item.tool_name),),
                    registration_id=item.registration_id,
                )
                for item in requests
            ]

    port = BatchPort()
    observations = asyncio.run(
        execute_tool_turn(
            run_id="run-core-batch",
            session_id="session-core-batch",
            current_mode="build",
            tools=port,
            tool_calls=[
                ToolCall(id="call-a", name="a", arguments={}),
                ToolCall(id="call-b", name="b", arguments={}),
            ],
            catalog_snapshot=snapshot,
        )
    )

    assert port.batch_calls == 1
    assert [item.tool_name for item in observations] == ["a", "b"]

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        asyncio.run(
            execute_tool_turn(
                run_id="run-old-call-shape",
                session_id="session-core-batch",
                current_mode="build",
                tools=port,
                tool_calls=[],
                catalog_snapshot=snapshot,
                assistant_message=None,
            )
        )


@dataclass(frozen=True)
class SampleInput:
    label: str
    delay_ms: int = 0


@dataclass(frozen=True)
class SampleOutput:
    label: str


def _registration(
    name,
    handler,
    *,
    timeout_ms=500,
    concurrency="parallel",
    group=None,
    approval="never",
    max_timeout_ms=None,
    requested_timeout_ms=None,
):
    from codepilot.tools.codecs import DataclassCodec
    from codepilot.tools.contracts import ToolRegistration, ToolSpec
    from codepilot.tools.results import TextContent
    from codepilot.tools.security import (
        ConcurrencyPolicy,
        OutputLimits,
        OutputTrustPolicy,
        TimeoutPolicy,
        ToolAccessRequest,
        ToolAccessResolution,
        ToolPolicy,
    )

    input_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "delay_ms": {"type": "integer", "minimum": 0},
        },
        "required": ["label"],
        "additionalProperties": False,
    }
    output_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
        "additionalProperties": False,
    }

    class Resolver:
        def resolve(self, input, context):
            _ = context
            return ToolAccessResolution(
                input=input,
                access=ToolAccessRequest(
                    actions=(name,),
                    resources=(),
                    effects=frozenset({"session_state_write"}),
                    risk="low",
                    reason="execution test",
                ),
                execution_timeout_ms=requested_timeout_ms,
            )

    class Renderer:
        def render(self, data):
            return (TextContent(text=data["label"]),)

    input_codec = DataclassCodec(SampleInput, input_schema)
    output_codec = DataclassCodec(SampleOutput, output_schema)
    return ToolRegistration(
        version="1.0.0",
        implementation_version="1",
        spec=ToolSpec(name, f"Execution test tool {name}.", input_codec.json_schema, output_codec.json_schema),
        category="external",
        source="builtin",
        owner="test",
        policy=ToolPolicy(
            allowed_modes=frozenset({"execute"}),
            declared_effects=frozenset({"session_state_write"}),
            required_permissions=frozenset(),
            base_risk="low",
            approval=approval,
            timeout=TimeoutPolicy(
                default_execution_ms=timeout_ms,
                max_execution_ms=max_timeout_ms or timeout_ms,
            ),
            concurrency=ConcurrencyPolicy(mode=concurrency, group=group),
            output_limits=OutputLimits(),
            output_trust=OutputTrustPolicy(),
        ),
        input_codec=input_codec,
        output_codec=output_codec,
        handler=handler,
        renderer=Renderer(),
        access_resolver=Resolver(),
    )


def _runtime(registration, *, store=None, controller=None):
    runtime, ids = _multi_runtime((registration,), store=store, controller=controller)
    return runtime, ids[registration.spec.name]


def _multi_runtime(registrations, *, store=None, controller=None):
    from codepilot.tools.registry import ToolRegistry
    from codepilot.tools.runtime import ToolRuntime

    registry = ToolRegistry()
    ids = {item.spec.name: registry.register(item) for item in registrations}
    kwargs = {}
    if store is not None:
        kwargs["state_store"] = store
    if controller is not None:
        kwargs["execution_controller"] = controller
    return ToolRuntime(registry, **kwargs), ids


def _request(name, registration_id, label, *, delay_ms=0):
    from codepilot.tools.contracts import ToolExecutionRequest

    return ToolExecutionRequest(
        run_id="run-execution-v2",
        session_id="session-execution-v2",
        tool_call_id=f"call-{name}",
        tool_name=name,
        arguments={"label": label, "delay_ms": delay_ms},
        mode="execute",
        registration_id=registration_id,
    )
