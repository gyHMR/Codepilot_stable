from __future__ import annotations

import json
from pathlib import Path
import asyncio

from codepilot.evaluation.executor import EvaluationExecutor
from codepilot.evaluation.artifacts import EvalArtifactStore
from codepilot.evaluation.loader import load_eval_definition
from codepilot.evaluation.metrics import calculate_case_metrics
from codepilot.evaluation.report import build_suite_summary, render_suite_markdown
from codepilot.evaluation.types import (
    AssertionResult,
    EvalBudgets,
    EvalCase,
    EvalEvidence,
    EvalResult,
    EvalRuntimeProfile,
)
from codepilot.llm import AssistantMessageEventStream
from codepilot.observability import AuditBundle
from codepilot.protocols import (
    AssistantMessage,
    ContextReport,
    Model,
    TextContent,
)
from codepilot.runtime import RuntimeService
from codepilot.runtime.types import CreateAgentSessionOptions, UserInput


def _bundle(
    tmp_path: Path,
    *,
    report: dict,
    events: list[dict] | None = None,
    result: dict | None = None,
) -> AuditBundle:
    return AuditBundle(
        run_id="run_1",
        session_id="session_1",
        events=events or [],
        state={},
        result=result or {},
        report=report,
        workspace=tmp_path,
    )


def _evidence(tmp_path: Path, bundle: AuditBundle) -> EvalEvidence:
    return EvalEvidence(
        workspace=tmp_path,
        baseline={},
        session_id="session_1",
        audit_bundles=[bundle],
    )


def _model() -> Model:
    return Model(
        id="eval-model",
        name="Eval Model",
        api="unit-test",
        provider="unit-test",
        base_url="",
        reasoning=False,
        input=["text"],
        context_window=1000,
        max_tokens=100,
    )


async def _direct_final_stream(**_: object) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    message = AssistantMessage(
        content=[TextContent(text="done")],
        stop_reason="stop",
        model="eval-model",
        provider="unit-test",
        api="unit-test",
    )
    stream.end(message)
    return stream


def test_loader_parses_metrics_and_expected(tmp_path: Path) -> None:
    path = tmp_path / "case.json"
    path.write_text(
        json.dumps(
            {
                "id": "context-key-hit",
                "domain": "context",
                "fixture": "context_heavy",
                "prompt": "find the important files",
                "metrics": [
                    "context.key_context_hit_rate",
                    "context.token_efficiency",
                ],
                "expected": {
                    "key_context": ["src/sample.py", "docs/architecture.md"],
                },
                "assertions": [{"type": "run", "expect_status": "completed"}],
            }
        ),
        encoding="utf-8",
    )

    definition = load_eval_definition(path)

    assert isinstance(definition, EvalCase)
    assert definition.metrics == [
        "context.key_context_hit_rate",
        "context.token_efficiency",
    ]
    assert definition.expected["key_context"] == [
        "src/sample.py",
        "docs/architecture.md",
    ]


def test_context_metrics_use_selected_items(tmp_path: Path) -> None:
    evidence = _evidence(
        tmp_path,
        _bundle(
            tmp_path,
            report={
                "context": {
                    "reports": [
                        {
                            "estimated_tokens_after": 200,
                            "selected_items": [
                                {
                                    "id": "active:src/sample.py",
                                    "path": "src/sample.py",
                                    "estimated_tokens": 30,
                                    "freshness": "fresh",
                                },
                                {
                                    "id": "active:docs/legacy-notes.md",
                                    "path": "docs/legacy-notes.md",
                                    "estimated_tokens": 70,
                                    "freshness": "stale",
                                },
                            ]
                        }
                    ]
                }
            },
        ),
    )

    metrics = calculate_case_metrics(
        [
            "context.key_context_hit_rate",
            "context.token_efficiency",
            "context.stale_context_rate",
        ],
        {"key_context": ["src/sample.py", "docs/architecture.md"]},
        evidence,
        [],
    )

    assert metrics["context.key_context_hit_rate"]["value"] == 0.5
    assert metrics["context.token_efficiency"]["value"] == 0.15
    assert metrics["context.stale_context_rate"]["value"] == 0.5


def test_memory_metrics_count_hits_reads_and_repeated_failures(
    tmp_path: Path,
) -> None:
    events = [
        {
            "type": "tool_execution_start",
            "toolCallId": "read_1",
            "toolName": "read",
            "args": {"path": "calculator.py"},
        },
        {
            "type": "tool_execution_start",
            "toolCallId": "read_2",
            "toolName": "read",
            "args": {"path": "calculator.py"},
        },
        {
            "type": "tool_execution_start",
            "toolCallId": "edit_1",
            "toolName": "edit",
            "args": {"path": "calculator.py"},
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "edit_1",
            "toolName": "edit",
            "status": "error",
            "errorReason": "old_text_not_found",
        },
        {
            "type": "tool_execution_start",
            "toolCallId": "edit_2",
            "toolName": "edit",
            "args": {"path": "calculator.py"},
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "edit_2",
            "toolName": "edit",
            "status": "error",
            "errorReason": "old_text_not_found",
        },
    ]
    evidence = _evidence(
        tmp_path,
        _bundle(
            tmp_path,
            report={
                "memory": {
                    "retrieved_memory_ids": ["memory:file-summary"],
                }
            },
            events=events,
        ),
    )

    metrics = calculate_case_metrics(
        [
            "memory.memory_retrieval_hit_rate",
            "memory.redundant_read_count",
            "memory.failed_attempt_recurrence_rate",
        ],
        {"memory_ids": ["memory:file-summary", "memory:failed-fix"]},
        evidence,
        [],
    )

    assert metrics["memory.memory_retrieval_hit_rate"]["value"] == 0.5
    assert metrics["memory.redundant_read_count"]["value"] == 1
    assert metrics["memory.failed_attempt_recurrence_rate"]["value"] == 0.5


def test_memory_hit_rate_can_use_simple_retrieval_expectation(
    tmp_path: Path,
) -> None:
    evidence = _evidence(
        tmp_path,
        _bundle(
            tmp_path,
            report={"memory": {"retrieved_memory_ids": ["mem_dynamic"]}},
        ),
    )

    metrics = calculate_case_metrics(
        ["memory.memory_retrieval_hit_rate"],
        {"expect_memory_retrieval": True},
        evidence,
        [],
    )

    assert metrics["memory.memory_retrieval_hit_rate"]["value"] == 1.0


def test_planning_metrics_use_task_events_and_outcome(tmp_path: Path) -> None:
    task = {
        "steps": [
            {"id": "step_1", "status": "completed", "evidence_refs": ["tool:1"]},
            {"id": "step_2", "status": "completed", "evidence_refs": []},
        ],
        "completion_satisfied": True,
    }
    events = [
        {"type": "task_step_updated", "task": task},
        {"type": "task_decision", "decision": {"action": "repair"}},
    ]
    evidence = _evidence(
        tmp_path,
        _bundle(
            tmp_path,
            report={"task": {"completion_satisfied": True}},
            events=events,
        ),
    )
    failed_assertion = AssertionResult(
        name="command",
        dimension="coding_outcome",
        status="failed",
        summary="tests failed",
        required=True,
    )

    metrics = calculate_case_metrics(
        [
            "planning.evidence_coverage_rate",
            "planning.false_completion_rate",
            "planning.repair_replan_success_rate",
        ],
        {},
        evidence,
        [failed_assertion],
    )

    assert metrics["planning.evidence_coverage_rate"]["value"] == 0.5
    assert metrics["planning.false_completion_rate"]["value"] == 1.0
    assert metrics["planning.repair_replan_success_rate"]["value"] == 0.0


def test_security_metrics_distinguish_dangerous_and_benign_tools(
    tmp_path: Path,
) -> None:
    events = [
        {
            "type": "tool_execution_end",
            "toolCallId": "write_1",
            "toolName": "write",
            "status": "denied",
            "workspaceChanged": False,
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "read_1",
            "toolName": "read",
            "status": "success",
            "workspaceChanged": False,
        },
    ]
    evidence = _evidence(
        tmp_path,
        _bundle(
            tmp_path,
            report={
                "security": {
                    "denied_or_approval_count": 1,
                    "mutation_after_denial_count": 0,
                }
            },
            events=events,
        ),
    )

    metrics = calculate_case_metrics(
        [
            "security.dangerous_tool_block_rate",
            "security.mutation_after_denial_rate",
            "security.benign_tool_pass_rate",
        ],
        {
            "dangerous_tools": ["write"],
            "benign_tools": ["read"],
        },
        evidence,
        [],
    )

    assert metrics["security.dangerous_tool_block_rate"]["value"] == 1.0
    assert metrics["security.mutation_after_denial_rate"]["value"] == 0.0
    assert metrics["security.benign_tool_pass_rate"]["value"] == 1.0


def test_security_block_rate_can_count_safe_refusal_without_tool_call(
    tmp_path: Path,
) -> None:
    evidence = _evidence(
        tmp_path,
        _bundle(
            tmp_path,
            report={},
            result={
                "final_message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "这是危险操作，我不能执行。",
                        }
                    ],
                }
            },
        ),
    )

    metrics = calculate_case_metrics(
        ["security.dangerous_tool_block_rate"],
        {
            "dangerous_tools": ["bash"],
            "allow_safe_refusal": True,
        },
        evidence,
        [],
    )

    assert metrics["security.dangerous_tool_block_rate"]["value"] == 1.0


def test_ratio_metric_is_not_applicable_when_denominator_is_zero(
    tmp_path: Path,
) -> None:
    evidence = _evidence(
        tmp_path,
        _bundle(tmp_path, report={"context": {"reports": []}}),
    )

    metrics = calculate_case_metrics(
        ["context.key_context_hit_rate"],
        {"key_context": []},
        evidence,
        [],
    )

    assert metrics["context.key_context_hit_rate"]["value"] is None
    assert metrics["context.key_context_hit_rate"]["display"] == "N/A"


def test_context_report_exposes_selected_item_summaries() -> None:
    report = ContextReport(
        context_id="ctx_1",
        repository_fingerprint="repo",
        total_budget_tokens=100,
        estimated_tokens_before=80,
        estimated_tokens_after=40,
        selected_items=[
            {
                "id": "active:calculator.py",
                "path": "calculator.py",
                "estimated_tokens": 20,
                "freshness": "fresh",
            }
        ],
    )

    assert report.to_dict()["selected_items"] == [
        {
            "id": "active:calculator.py",
            "path": "calculator.py",
            "estimated_tokens": 20,
            "freshness": "fresh",
        }
    ]


def test_runtime_profile_can_disable_context_memory_and_task_control(
    tmp_path: Path,
) -> None:
    base = CreateAgentSessionOptions(
        workspace_dir=tmp_path,
        model=_model(),
        stream_fn=_direct_final_stream,
    )
    applied = EvaluationExecutor._apply_runtime_profile(
        base,
        EvalRuntimeProfile(
            context_governance_enabled=False,
            memory_enabled=False,
            task_control_enabled=False,
        ),
    )
    runtime = RuntimeService()
    handle = runtime.create_session(applied)
    events: list[dict] = []
    handle.session.subscribe(lambda event: events.append(dict(event)))

    async def run() -> object:
        result = await runtime.run_message(
            handle.session_id,
            UserInput(text="answer directly"),
        )
        await runtime.aclose_all()
        return result

    result = asyncio.run(run())

    assert applied.context_governance_enabled is False
    assert applied.memory_enabled is False
    assert applied.task_control_enabled is False
    assert handle.assembly.session_options.prepare_context is None
    assert handle.session.memory_enabled is False
    assert result.task is None
    assert not any(
        event.get("type") in {
            "context_prepared",
            "memory_retrieved",
            "memory_updated",
            "task_plan_created",
            "task_step_updated",
            "task_decision",
            "completion_checked",
        }
        for event in events
    )


def test_executor_adds_declared_metrics_to_eval_result(tmp_path: Path) -> None:
    evidence = _evidence(
        tmp_path,
        _bundle(
            tmp_path,
            report={
                "memory": {
                    "retrieved_memory_ids": ["memory:known-file"],
                }
            },
        ),
    )
    passed = AssertionResult(
        name="run",
        dimension="runtime_contract",
        status="passed",
        summary="completed",
        required=True,
    )
    artifacts = EvalArtifactStore(tmp_path / "evals", "eval_1")

    result = EvaluationExecutor()._evaluate(
        "memory-hit",
        [],
        EvalBudgets(),
        evidence,
        artifacts,
        metric_names=["memory.memory_retrieval_hit_rate"],
        expected={"memory_ids": ["memory:known-file"]},
        error=None,
        started=0.0,
        precomputed=[passed],
    )

    assert result.metrics["memory.memory_retrieval_hit_rate"]["value"] == 1.0
    assert result.metrics["runtime.run_count"] == 1


def test_suite_summary_and_markdown_include_metric_averages() -> None:
    results = [
        EvalResult(
            case_id="case-1",
            overall="passed",
            session_id="session-1",
            run_ids=["run-1"],
            dimensions=[],
            failure_categories=[],
            metrics={
                "context.key_context_hit_rate": {
                    "value": 1.0,
                    "display": "100.0%",
                }
            },
            artifact_dir="case-1",
        ),
        EvalResult(
            case_id="case-2",
            overall="passed",
            session_id="session-2",
            run_ids=["run-2"],
            dimensions=[],
            failure_categories=[],
            metrics={
                "context.key_context_hit_rate": {
                    "value": 0.5,
                    "display": "50.0%",
                }
            },
            artifact_dir="case-2",
        ),
    ]

    summary = build_suite_summary(results)
    markdown = render_suite_markdown(results, summary)

    assert summary["metric_averages"]["context.key_context_hit_rate"] == 0.75
    assert "Key Context Hit Rate" in markdown
    assert "75.0%" in markdown
