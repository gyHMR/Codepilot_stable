from __future__ import annotations

import asyncio
from dataclasses import dataclass


def test_plan_adapter_returns_core_command_without_writing_plan_state() -> None:
    from codepilot.core.tool_adapters.plan import create_plan_registrations
    from codepilot.tools.registry import ToolRegistry
    from codepilot.tools.runtime import ToolRuntime

    registration = {
        item.spec.name: item for item in create_plan_registrations()
    }["create_build_plan"]
    assert registration.category == "plan"
    assert registration.policy.approval == "never"
    assert registration.policy.concurrency.mode == "serial"
    assert registration.policy.concurrency.group == "task_plan"

    registry = ToolRegistry()
    registration_id = registry.register(registration)
    runtime = ToolRuntime(registry)
    request = _request(
        "create_build_plan",
        registration_id,
        mode="execute",
        arguments=_build_plan_snapshot(),
    )

    result = asyncio.run(runtime.execute(request))

    assert result.status == "success"
    assert result.approval is None
    assert result.data["plan_operation"] == "create_build_plan"
    command = result.data["core_command"]
    assert command["kind"] == "submit_plan"
    assert command["command_id"] == "call-create_build_plan"
    assert command["definition"]["summary"] == "Migrate the remaining tool adapters."
    assert command["definition"]["completion_criteria"] == (
        "Special tools use canonical registrations.",
    )
    assert command["steps"][0]["step"] == "Migrate special tools"
    assert result.effects == ()
    assert registration.policy.declared_effects == frozenset()


def test_interaction_pauses_and_resumes_same_attempt_once_without_rerunning_handler() -> None:
    from codepilot.core.tool_adapters.interaction import create_interaction_registration
    from codepilot.tools.registry import ToolRegistry
    from codepilot.tools.runtime import ToolRuntime
    from codepilot.tools.state import (
        InMemoryToolStateStore,
        InteractionResponse,
        attempt_id_for,
    )

    store = InMemoryToolStateStore()
    registry = ToolRegistry()
    registration_id = registry.register(create_interaction_registration())
    runtime = ToolRuntime(registry, state_store=store)
    request = _request(
        "request_user_input",
        registration_id,
        mode="execute",
        arguments={
            "prompt": "Choose the migration strategy",
            "options": ["incremental", "single cutover"],
            "allow_free_text": False,
        },
    )

    suspended = asyncio.run(runtime.execute(request))

    assert suspended.status == "user_input_required"
    assert suspended.interaction is not None
    record = store.get(attempt_id_for(request))
    assert record is not None
    assert record.state == "awaiting_input"
    assert record.result is None

    interaction = suspended.interaction
    bad_response = InteractionResponse(
        interaction_id=str(interaction["interaction_id"]),
        request_fingerprint="wrong-fingerprint",
        session_id=request.session_id,
        tool_call_id=request.tool_call_id,
        tool_name=request.tool_name,
        registration_id=request.registration_id,
        answers={"answer": "incremental"},
    )
    rejected = asyncio.run(runtime.resume(bad_response))
    assert rejected.status == "error"
    assert rejected.error is not None
    assert rejected.error.code == "tool.interaction.fingerprint_mismatch"
    assert store.get(attempt_id_for(request)).state == "awaiting_input"

    response = InteractionResponse(
        interaction_id=str(interaction["interaction_id"]),
        request_fingerprint=str(interaction["request_fingerprint"]),
        session_id=request.session_id,
        tool_call_id=request.tool_call_id,
        tool_name=request.tool_name,
        registration_id=request.registration_id,
        answers={"answer": "incremental"},
    )
    async def resume_twice():
        return await asyncio.gather(runtime.resume(response), runtime.resume(response))

    resumed, duplicate = asyncio.run(resume_twice())
    if resumed.status == "error":
        resumed, duplicate = duplicate, resumed

    assert resumed.status == "success"
    assert resumed.data == {"answers": {"answer": "incremental"}}
    assert store.get(attempt_id_for(request)).state == "succeeded"
    assert duplicate.status == "error"
    assert duplicate.error is not None
    assert duplicate.error.code == "tool.interaction.already_consumed"


def test_interaction_is_an_execute_batch_admission_barrier() -> None:
    from codepilot.core.tool_adapters.interaction import create_interaction_registration
    from codepilot.tools.execution import ExecutionController

    called = False

    async def after_handler(input, context):
        nonlocal called
        _ = input, context
        called = True
        return _Output("unexpected")

    interaction = create_interaction_registration()
    after = _sample_registration("after_interaction", after_handler)
    runtime, ids = _runtime((interaction, after), controller=ExecutionController())

    results = asyncio.run(
        runtime.execute_batch(
            (
                _request(
                    "request_user_input",
                    ids["request_user_input"],
                    mode="execute",
                    arguments={"prompt": "Continue?", "options": ["yes", "no"]},
                ),
                _request(
                    "after_interaction",
                    ids["after_interaction"],
                    mode="execute",
                    arguments={"label": "later"},
                ),
            )
        )
    )

    assert [item.status for item in results] == ["user_input_required", "interrupted"]
    assert called is False


def test_subagent_adapter_registers_runtime_owned_tools_and_executes_through_runtime(tmp_path) -> None:
    from codepilot.runtime.subagents.tools import create_subagent_registrations

    class Session:
        session_id = "session-stage-7"

    registrations = create_subagent_registrations(
        workspace=tmp_path,
        session_provider=Session,
    )
    by_name = {item.spec.name: item for item in registrations}
    assert set(by_name) == {"dispatch_exploration", "list_exploration_agents"}
    assert all(item.category == "delegation" for item in registrations)
    assert all(item.source == "builtin" for item in registrations)
    assert by_name["dispatch_exploration"].policy.concurrency.mode == "parallel"

    runtime, ids = _runtime(registrations)
    result = asyncio.run(
        runtime.execute(
            _request(
                "list_exploration_agents",
                ids["list_exploration_agents"],
                mode="plan",
                arguments={},
            )
        )
    )

    assert result.status == "success"
    assert result.data["agents"] == ()
    assert result.data["has_reports"] is False


def test_runtime_composition_registers_all_canonical_special_adapters(tmp_path) -> None:
    from codepilot.runtime import SessionOpenIntent
    from codepilot.runtime.builder import build_runtime_session

    session = build_runtime_session(
        SessionOpenIntent(
            workspace_dir=tmp_path,
            provider="deepseek",
            model_id="deepseek-v4-pro",
            load_workspace_resources=False,
            memory_enabled=False,
        )
    )
    try:
        entries = {
            item.spec.name: item
            for item in session.tool_port.catalog_snapshot().entries
        }
    finally:
        session.controller.close()

    assert {
        "propose_plan",
        "create_build_plan",
        "update_plan_progress",
        "close_plan",
        "request_user_input",
        "list_exploration_agents",
        "dispatch_exploration",
    } <= set(entries)
    assert entries["propose_plan"].category == "plan"
    assert entries["request_user_input"].category == "interaction"
    assert entries["dispatch_exploration"].category == "delegation"


def _build_plan_snapshot() -> dict[str, object]:
    return {
        "summary": "Migrate the remaining tool adapters.",
        "completion_criteria": ["Special tools use canonical registrations."],
        "items": [
            {
                "step": "Migrate special tools",
                "details": "Register Plan, Interaction, and Subagent adapters.",
                "verification": "Run focused tool tests.",
            }
        ],
    }


def _request(name, registration_id, *, mode, arguments):
    from codepilot.tools.contracts import ToolExecutionRequest

    return ToolExecutionRequest(
        run_id="run-stage-7",
        session_id="session-stage-7",
        tool_call_id=f"call-{name}",
        tool_name=name,
        arguments=arguments,
        mode=mode,
        registration_id=registration_id,
    )


@dataclass(frozen=True)
class _Input:
    label: str


@dataclass(frozen=True)
class _Output:
    label: str


def _sample_registration(name, handler):
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

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
        "additionalProperties": False,
    }

    class Resolver:
        def resolve(self, input, request):
            _ = request
            return ToolAccessResolution(
                input=input,
                access=ToolAccessRequest(
                    actions=(name,),
                    resources=(),
                    effects=frozenset(),
                    risk="low",
                    reason="stage 7 batch test",
                ),
            )

    class Renderer:
        def render(self, data):
            return (TextContent(text=data["label"]),)

    codec_in = DataclassCodec(_Input, schema)
    codec_out = DataclassCodec(_Output, schema)
    return ToolRegistration(
        version="1.0.0",
        implementation_version="1",
        spec=ToolSpec(name, "Stage 7 batch sentinel.", schema, schema),
        category="external",
        source="builtin",
        owner="test",
        policy=ToolPolicy(
            allowed_modes=frozenset({"execute"}),
            declared_effects=frozenset(),
            required_permissions=frozenset(),
            base_risk="low",
            approval="never",
            timeout=TimeoutPolicy(500, 500),
            concurrency=ConcurrencyPolicy(mode="parallel"),
            output_limits=OutputLimits(),
            output_trust=OutputTrustPolicy(),
        ),
        input_codec=codec_in,
        output_codec=codec_out,
        handler=handler,
        renderer=Renderer(),
        access_resolver=Resolver(),
    )


def _runtime(registrations, *, controller=None):
    from codepilot.tools.registry import ToolRegistry
    from codepilot.tools.runtime import ToolRuntime

    registry = ToolRegistry()
    ids = {item.spec.name: registry.register(item) for item in registrations}
    kwargs = {"execution_controller": controller} if controller is not None else {}
    return ToolRuntime(registry, **kwargs), ids
