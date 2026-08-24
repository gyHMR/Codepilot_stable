from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

from codepilot.core.contracts import ContextPrepareRequest, CoreContextView
from codepilot.core.state import CoreState
from codepilot.llm.ports import ModelDescriptor
from codepilot.protocols import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from codepilot.sessions.context import ContextBudgetConfig, ContextService
from codepilot.sessions.memory import MemoryRecallResult, RecalledMemory


class _MemoryRecall:
    def recall(self, _query) -> MemoryRecallResult:
        return MemoryRecallResult(
            retrieved=(
                RecalledMemory(
                    memory_id="mem_1",
                    scope="project",
                    type="project",
                    key="project.verification.command",
                    content="Use focused pytest commands.",
                    source="user_explicit",
                    rank_reasons=("term:pytest",),
                ),
            )
        )


class _FailingMemoryRecall:
    def recall(self, _query):
        raise OSError("memory store unavailable")


def _request() -> ContextPrepareRequest:
    state = CoreState.new("Refactor context governance")
    message = UserMessage(
        content="Continue the context refactor.",
        metadata={"session_message_id": "msg_current"},
    )
    return ContextPrepareRequest(
        session_id="session_1",
        run_id="run_1",
        purpose="reasoning",
        directive="core.reasoning",
        messages=(message,),
        core_view=CoreContextView.from_state(state, "build"),
        model=ModelDescriptor(provider="unit", model_id="unit"),
        tool_catalog=None,
        seed={
            "system_prompt": "L0 immutable rules.",
            "permission_mode": "default",
            "checkpoint_phase": "running",
        },
    )


def test_service_materializes_five_layers_without_putting_dynamic_state_in_l0(
    tmp_path: Path,
) -> None:
    service = ContextService(
        workspace_dir=tmp_path,
        session_id="session_1",
        memory_recall=_MemoryRecall(),
        budget_config=ContextBudgetConfig(
            context_window=8000,
            max_output_tokens=500,
            safety_margin_tokens=0,
        ),
    )

    prepared = asyncio.run(service.prepare(_request()))

    assert prepared.system_prompt.startswith("L0 immutable rules.")
    assert "# Codepilot Runtime Control" in prepared.system_prompt
    assert "Core directive for this model call:" in prepared.system_prompt
    assert "core.reasoning" in prepared.system_prompt
    assert prepared.messages[0].metadata["context_attachment"] is True
    attachment = str(prepared.messages[0].content)
    assert "L1 Runtime And Task State" in attachment
    assert "Refactor context governance" in attachment
    assert "L2 Working Set And Evidence" in attachment
    assert "L3 Recalled Memory" in attachment
    assert "project.verification.command" in attachment
    assert prepared.messages[-1].metadata["session_message_id"] == "msg_current"
    assert service.latest_report["layers"]["l0"] == [prepared.system_prompt]
    assert "Core directive:" not in attachment
    assert "l1:directive" not in service.latest_report["selected_items"]


def test_memory_recall_failure_degrades_to_empty_l3(tmp_path: Path) -> None:
    service = ContextService(
        workspace_dir=tmp_path,
        session_id="session_1",
        memory_recall=_FailingMemoryRecall(),
        budget_config=ContextBudgetConfig(
            context_window=8000,
            max_output_tokens=500,
            safety_margin_tokens=0,
        ),
    )

    prepared = asyncio.run(service.prepare(_request()))

    assert "## L3 Recalled Memory\n- (none)" in str(prepared.messages[0].content)
    assert service.latest_report["memory_error"] == "memory store unavailable"


def test_l2_does_not_duplicate_read_source_body_from_l4(tmp_path: Path) -> None:
    assistant = AssistantMessage(
        content=[ToolCall(id="read_1", name="read", arguments={"path": "src/app.py"})],
        metadata={"session_message_id": "msg_assistant"},
    )
    result = ToolResultMessage(
        tool_call_id="read_1",
        tool_name="read",
        content=[TextContent(text="UNIQUE_LOGIN_SOURCE = True")],
        details={
            "path": "src/app.py",
            "sha256": "hash-a",
            "offset": 1,
            "returned_lines": 1,
        },
        metadata={"session_message_id": "msg_result"},
    )
    base = _request()
    request = replace(base, messages=(base.messages[0], assistant, result))
    service = ContextService(
        workspace_dir=tmp_path,
        session_id="session_1",
        budget_config=ContextBudgetConfig(
            context_window=8000,
            max_output_tokens=500,
            safety_margin_tokens=0,
        ),
    )

    prepared = asyncio.run(service.prepare(request))

    attachment = str(prepared.messages[0].content)
    assert "UNIQUE_LOGIN_SOURCE" not in attachment
    projected_result = next(
        message for message in prepared.messages if isinstance(message, ToolResultMessage)
    )
    assert "UNIQUE_LOGIN_SOURCE" in str(projected_result.content[0].text)


def test_runtime_control_is_compiled_into_system_prompt_not_user_attachment(
    tmp_path: Path,
) -> None:
    request = replace(
        _request(),
        seed={
            "system_prompt": "L0 immutable rules.",
            "mode_policy": "当前 mode=plan。只允许只读调查。",
            "synthetic_control": {
                "kind": "plan_feedback",
                "scope": "plan_revision_only",
                "instruction": "本轮必须调用 propose_plan。",
            },
        },
    )
    service = ContextService(
        workspace_dir=tmp_path,
        session_id="session_1",
        budget_config=ContextBudgetConfig(
            context_window=8000,
            max_output_tokens=500,
            safety_margin_tokens=0,
        ),
    )

    prepared = asyncio.run(service.prepare(request))
    attachment = str(prepared.messages[0].content)

    assert prepared.system_prompt.startswith("L0 immutable rules.")
    assert "# Codepilot Runtime Control" in prepared.system_prompt
    assert "当前 mode=plan" in prepared.system_prompt
    assert "Continuation event: plan_feedback" in prepared.system_prompt
    assert "Required scope: plan_revision_only" in prepared.system_prompt
    assert "本轮必须调用 propose_plan" in prepared.system_prompt
    assert "Core directive for this model call:" in prepared.system_prompt
    assert "core.reasoning" in prepared.system_prompt
    assert "当前 mode=plan" not in attachment
    assert "本轮必须调用 propose_plan" not in attachment
    assert "core.reasoning" not in attachment


def test_canonical_plan_is_required_runtime_state_with_full_step_details(
    tmp_path: Path,
) -> None:
    from codepilot.core.contracts import TaskPlanView, TaskStepView

    step = TaskStepView(
        step_id="plan_1:step:1",
        step="实现登录 API",
        details="修改认证服务和路由，并保留现有错误协议。",
        verification="运行登录集成测试。",
        status="pending",
    )
    plan = TaskPlanView(
        plan_id="plan_1",
        origin="plan_mode",
        status="proposed",
        revision=2,
        definition={
            "summary": "完善登录流程",
            "completion_criteria": ["登录集成测试通过"],
            "target_design": "使用现有服务边界实现认证。",
            "impact_scope": "认证服务、路由和测试。",
        },
        steps=(step,),
    )
    base = _request()
    request = replace(
        base,
        core_view=replace(base.core_view, mode="plan", plan=plan, current_step=step),
    )
    service = ContextService(
        workspace_dir=tmp_path,
        session_id="session_1",
        budget_config=ContextBudgetConfig(
            context_window=8000,
            max_output_tokens=500,
            safety_margin_tokens=0,
        ),
    )

    prepared = asyncio.run(service.prepare(request))
    attachment = str(prepared.messages[0].content)

    assert "Canonical Task Plan (runtime-authoritative state)" in attachment
    assert '"plan_id":"plan_1"' in attachment
    assert '"revision":2' in attachment
    assert '"target_design":"使用现有服务边界实现认证。"' in attachment
    assert '"details":"修改认证服务和路由，并保留现有错误协议。"' in attachment
    assert '"verification":"运行登录集成测试。"' in attachment
    assert "l1:plan" in service.latest_report["selected_items"]
    assert "l1:plan" not in service.latest_report["dropped_items"]
