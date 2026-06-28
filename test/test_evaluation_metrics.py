from __future__ import annotations

import json
from pathlib import Path
import asyncio

from codepilot.evaluation.executor import EvaluationExecutor
from codepilot.evaluation.artifacts import EvalArtifactStore
from codepilot.evaluation.loader import load_eval_definition
from codepilot.evaluation.metrics import calculate_case_metrics
from codepilot.evaluation.report import build_suite_summary, render_suite_markdown
from codepilot.evaluation.service import filter_definitions_by_tags
from codepilot.evaluation.types import (
    AssertionResult,
    AssertionSpec,
    DimensionResult,
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


def test_filter_definitions_by_tags_requires_all_requested_tags() -> None:
    graded_hard = EvalCase(
        id="graded-hard",
        domain="planning",
        fixture="issue_tracker",
        prompt="fix a representative bug",
        assertions=[],
        tags=["suite:graded", "difficulty:hard"],
    )
    graded_medium = EvalCase(
        id="graded-medium",
        domain="planning",
        fixture="issue_tracker",
        prompt="inspect representative files",
        assertions=[],
        tags=["suite:graded", "difficulty:medium"],
    )
    contract_hard = EvalCase(
        id="contract-hard",
        domain="planning",
        fixture="calculator",
        prompt="repair a small fixture",
        assertions=[],
        tags=["suite:contract", "difficulty:hard"],
    )

    filtered = filter_definitions_by_tags(
        [graded_hard, graded_medium, contract_hard],
        ["suite:graded", "difficulty:hard"],
    )

    assert [item.id for item in filtered] == ["graded-hard"]


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


def test_token_efficiency_uses_latest_context_report_and_caps_at_one(
    tmp_path: Path,
) -> None:
    evidence = _evidence(
        tmp_path,
        _bundle(
            tmp_path,
            report={
                "context": {
                    "reports": [
                        {
                            "estimated_tokens_after": 100,
                            "selected_items": [
                                {
                                    "path": "old.py",
                                    "estimated_tokens": 100,
                                }
                            ],
                        },
                        {
                            "estimated_tokens_after": 50,
                            "selected_items": [
                                {
                                    "path": "src/current.py",
                                    "estimated_tokens": 80,
                                }
                            ],
                        },
                    ]
                }
            },
        ),
    )

    metrics = calculate_case_metrics(
        ["context.token_efficiency"],
        {"key_context": ["src/current.py"]},
        evidence,
        [],
    )

    assert metrics["context.token_efficiency"]["value"] == 1.0
    assert metrics["context.token_efficiency"]["numerator"] == 80
    assert metrics["context.token_efficiency"]["denominator"] == 80


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


def test_planning_step_completion_and_replan_metrics(tmp_path: Path) -> None:
    task = {
        "steps": [
            {"id": "step_1", "status": "completed", "evidence_refs": ["tool:1"]},
            {"id": "step_2", "status": "completed", "evidence_refs": ["tool:2"]},
            {"id": "step_3", "status": "pending", "evidence_refs": []},
        ],
        "completion_satisfied": True,
    }
    evidence = _evidence(
        tmp_path,
        _bundle(
            tmp_path,
            report={"task": {"completion_satisfied": True}},
            events=[
                {"type": "task_step_updated", "task": task},
                {"type": "task_decision", "decision": {"action": "replan"}},
            ],
        ),
    )

    metrics = calculate_case_metrics(
        [
            "planning.step_completion_rate",
            "planning.replan_success_rate",
        ],
        {},
        evidence,
        [],
    )

    assert metrics["planning.step_completion_rate"]["value"] == 2 / 3
    assert metrics["planning.step_completion_rate"]["numerator"] == 2
    assert metrics["planning.step_completion_rate"]["denominator"] == 3
    assert metrics["planning.replan_success_rate"]["value"] == 1.0


def test_planning_invalid_tool_call_count_uses_narrow_failure_reasons(
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
            "type": "tool_execution_end",
            "toolCallId": "read_1",
            "toolName": "read",
            "status": "success",
        },
        {
            "type": "tool_execution_start",
            "toolCallId": "shell_1",
            "toolName": "shell",
            "args": {"command": "python -m"},
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "shell_1",
            "toolName": "shell",
            "status": "error",
            "errorReason": "command_parse_error",
        },
        {
            "type": "tool_execution_start",
            "toolCallId": "edit_1",
            "toolName": "edit",
            "args": {"path": "calculator.py", "old_text": "return x"},
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
            "args": {"path": "calculator.py", "old_text": "return x"},
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "edit_2",
            "toolName": "edit",
            "status": "error",
            "errorReason": "old_text_not_found",
        },
        {
            "type": "tool_execution_start",
            "toolCallId": "write_1",
            "toolName": "write",
            "args": {"path": "state.txt"},
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "write_1",
            "toolName": "write",
            "status": "denied",
            "errorReason": "user_denied",
        },
    ]
    evidence = _evidence(tmp_path, _bundle(tmp_path, report={}, events=events))

    metrics = calculate_case_metrics(
        [
            "planning.invalid_tool_call_count",
            "planning.invalid_tool_call_rate",
        ],
        {},
        evidence,
        [],
    )

    assert metrics["planning.invalid_tool_call_count"]["value"] == 2
    assert metrics["planning.invalid_tool_call_count"]["numerator"] == 2
    assert metrics["planning.invalid_tool_call_count"]["denominator"] == 5
    assert metrics["planning.invalid_tool_call_rate"]["value"] == 0.4
    assert metrics["planning.invalid_tool_call_rate"]["numerator"] == 2
    assert metrics["planning.invalid_tool_call_rate"]["denominator"] == 5


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


def test_security_block_rate_counts_workspace_boundary_errors(
    tmp_path: Path,
) -> None:
    events = [
        {
            "type": "tool_execution_end",
            "toolCallId": "write_1",
            "toolName": "write",
            "status": "error",
            "errorReason": "path_escapes_workspace",
            "workspaceChanged": False,
        }
    ]
    evidence = _evidence(tmp_path, _bundle(tmp_path, report={}, events=events))

    metrics = calculate_case_metrics(
        ["security.dangerous_tool_block_rate"],
        {"dangerous_tools": ["write"]},
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


def test_runtime_profile_can_enable_eval_auto_approval(tmp_path: Path) -> None:
    base = CreateAgentSessionOptions(
        workspace_dir=tmp_path,
        model=_model(),
        stream_fn=_direct_final_stream,
    )

    applied = EvaluationExecutor._apply_runtime_profile(
        base,
        EvalRuntimeProfile(auto_approve=True),
    )

    assert applied.approval_provider is not None


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


def test_budget_assertions_are_diagnostic_and_do_not_fail_case(
    tmp_path: Path,
) -> None:
    evidence = _evidence(
        tmp_path,
        _bundle(
            tmp_path,
            report={
                "run": {
                    "summary": {
                        "tool_calls": 3,
                        "model_attempts": 1,
                    }
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
        role="guardrail",
    )
    artifacts = EvalArtifactStore(tmp_path / "evals", "eval_1")

    result = EvaluationExecutor()._evaluate(
        "budget-diagnostic",
        [],
        EvalBudgets(max_tool_calls=1),
        evidence,
        artifacts,
        error=None,
        started=0.0,
        precomputed=[passed],
    )

    budget_assertion = next(
        assertion
        for dimension in result.dimensions
        for assertion in dimension.assertion_results
        if assertion.name == "budget_tool_calls"
    )
    assert result.overall == "passed"
    assert budget_assertion.status == "failed"
    assert budget_assertion.required is False
    assert budget_assertion.role == "diagnostic"


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


def test_suite_summary_separates_primary_metrics_and_diagnostics() -> None:
    primary_pass = AssertionResult(
        name="metric",
        dimension="context_governance",
        status="passed",
        summary="context.key_context_hit_rate >= 1",
        actual={
            "metric": "context.key_context_hit_rate",
            "value": 1.0,
            "display": "100.0%",
        },
        role="primary",
    )
    diagnostic_fail = AssertionResult(
        name="budget_tool_calls",
        dimension="efficiency",
        status="failed",
        summary="tool_calls exceeded budget",
        actual={"value": 6},
        required=False,
        role="diagnostic",
    )
    diagnostic_metric = AssertionResult(
        name="metric",
        dimension="memory",
        status="failed",
        summary="memory.memory_retrieval_hit_rate expected >= 1, got 0",
        actual={
            "metric": "memory.memory_retrieval_hit_rate",
            "value": 0.0,
            "display": "0.0%",
        },
        required=False,
        role="diagnostic",
    )
    results = [
        EvalResult(
            case_id="context-case",
            overall="passed",
            session_id="session-1",
            run_ids=["run-1"],
            dimensions=[],
            failure_categories=[],
            metrics={
                "context.key_context_hit_rate": {
                    "value": 1.0,
                    "display": "100.0%",
                },
                "runtime.tool_calls": 6,
            },
            artifact_dir="context-case",
        ),
        EvalResult(
            case_id="security-case",
            overall="passed",
            session_id="session-2",
            run_ids=["run-2"],
            dimensions=[
                DimensionResult(
                    dimension="context_governance",
                    status="passed",
                    summary="1/1 assertions passed",
                    assertion_results=[primary_pass],
                ),
                DimensionResult(
                    dimension="efficiency",
                    status="failed",
                    summary="0/1 assertions passed",
                    assertion_results=[diagnostic_fail],
                ),
                DimensionResult(
                    dimension="memory",
                    status="failed",
                    summary="0/1 assertions passed",
                    assertion_results=[diagnostic_metric],
                ),
            ],
            failure_categories=["efficiency.budget_tool_calls_failed"],
            metrics={
                "context.key_context_hit_rate": {
                    "value": 0.5,
                    "display": "50.0%",
                },
                "memory.memory_retrieval_hit_rate": {
                    "value": 0.0,
                    "display": "0.0%",
                },
                "runtime.tool_calls": 6,
            },
            artifact_dir="security-case",
        ),
    ]

    summary = build_suite_summary(results)

    assert summary["domains"] == {
        "context": {"failed": 0, "passed": 1, "pass_rate": 1.0, "total": 1},
        "security": {"failed": 0, "passed": 1, "pass_rate": 1.0, "total": 1},
    }
    assert summary["primary_metric_averages"] == {
        "context.key_context_hit_rate": 1.0
    }
    assert "memory.memory_retrieval_hit_rate" not in summary["primary_metric_averages"]
    assert summary["assertion_roles"]["diagnostic"]["failed"] == 2
    assert summary["diagnostic_failures"] == {
        "efficiency.budget_tool_calls_failed": 1,
        "memory.metric_failed": 1,
    }
