from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from codepilot.evaluation import (
    EvalCase,
    EvalCaseValidationError,
    EvalRunOptions,
    EvalScenario,
    EvaluationService,
    ScenarioStep,
    VerifierSpec,
    load_eval_definition,
    load_eval_suite,
)
from codepilot.evaluation.types import EvaluationEvidence
from codepilot.evaluation.verifier import (
    capture_workspace_baseline,
    run_verifiers,
)
from codepilot.llm import AssistantMessageEventStream
from codepilot.protocols import AssistantMessage, Model, TextContent
from codepilot.runtime import CreateAgentSessionOptions


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


def test_loader_parses_case_and_scenario(tmp_path: Path) -> None:
    case_file = tmp_path / "case.json"
    case_file.write_text(
        json.dumps(
            {
                "id": "case-1",
                "category": "coding",
                "fixture": "fixture",
                "prompt": "fix it",
                "verifiers": [
                    {
                        "type": "command",
                        "command": "python -m pytest -q",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    scenario_file = tmp_path / "scenario.json"
    scenario_file.write_text(
        json.dumps(
            {
                "id": "scenario-1",
                "fixture": "fixture",
                "steps": [
                    {"type": "prompt", "text": "inspect"},
                    {
                        "type": "modify_file",
                        "path": "app.py",
                        "content": "changed",
                    },
                ],
                "verifiers": [{"type": "run", "expect_freshness": "stale"}],
            }
        ),
        encoding="utf-8",
    )

    case = load_eval_definition(case_file)
    scenario = load_eval_definition(scenario_file)

    assert isinstance(case, EvalCase)
    assert case.verifiers[0].type == "command"
    assert isinstance(scenario, EvalScenario)
    assert scenario.steps[1].type == "modify_file"


def test_loader_rejects_invalid_verifier(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps(
            {
                "id": "bad",
                "category": "coding",
                "fixture": "fixture",
                "prompt": "fix",
                "verifiers": [{"type": "command"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvalCaseValidationError):
        load_eval_definition(path)


def test_example_coding_benchmarks_are_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    definitions = load_eval_suite(root / "benchmarks" / "coding")

    assert len(definitions) == 3
    assert all(isinstance(item, EvalCase) for item in definitions)


def test_outcome_verifiers_include_evidence(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("before\n", encoding="utf-8")
    baseline = capture_workspace_baseline(workspace)
    (workspace / "app.py").write_text("after\n", encoding="utf-8")
    evidence = EvaluationEvidence(workspace=workspace, baseline=baseline)

    results = run_verifiers(
        [
            VerifierSpec(
                type="command",
                options={
                    "command": (
                        f'"{sys.executable}" -c "import pathlib; '
                        "assert pathlib.Path('app.py').exists()\""
                    )
                },
            ),
            VerifierSpec(
                type="file",
                options={"path": "app.py", "contains": "after"},
            ),
            VerifierSpec(
                type="diff",
                options={"allowed_paths": ["app.py"]},
            ),
        ],
        evidence,
    )

    assert [result.status for result in results] == [
        "passed",
        "passed",
        "passed",
    ]
    assert results[0].evidence["command"]
    assert results[2].actual == {"changed_paths": ["app.py"]}


def test_run_and_trace_verifiers_detect_contract_mismatch(
    tmp_path: Path,
) -> None:
    evidence = EvaluationEvidence(
        workspace=tmp_path,
        baseline={},
        session_id="s1",
        run_ids=["r1"],
        run_results={
            "r1": {
                "run_id": "r1",
                "session_id": "s1",
                "status": "completed",
                "stop_reason": "final_answer",
                "counters": {
                    "model_attempts": 1,
                    "tool_iterations": 1,
                    "tool_calls": 2,
                },
                "messages": [],
                "affected_paths": [],
                "workspace_changed": False,
                "verification": [],
            }
        },
        run_events={
            "r1": [
                {"type": "agent_start", "runId": "r1", "timestamp": 1},
                {
                    "type": "tool_execution_start",
                    "runId": "r1",
                    "toolCallId": "call-1",
                    "timestamp": 2,
                },
                {
                    "type": "tool_execution_end",
                    "runId": "r1",
                    "toolCallId": "call-1",
                    "timestamp": 3,
                },
                {"type": "agent_end", "runId": "r1", "timestamp": 4},
            ]
        },
    )

    results = run_verifiers(
        [
            VerifierSpec(
                type="run",
                options={
                    "expect_status": "completed",
                    "expect_stop_reason": "final_answer",
                },
            ),
            VerifierSpec(type="trace"),
        ],
        evidence,
    )

    assert results[0].status == "passed"
    assert results[1].status == "failed"
    assert "tool_calls counter=2" in results[1].summary


def test_evaluation_service_runs_case_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    fixtures = tmp_path / "fixtures"
    fixture = fixtures / "tiny"
    fixture.mkdir(parents=True)
    (fixture / "app.py").write_text("BUG\n", encoding="utf-8")
    artifacts = tmp_path / "artifacts"

    service = EvaluationService(runtime_factory=_FakeRuntime)
    case = EvalCase(
        id="fix-bug",
        category="coding",
        fixture="tiny",
        prompt="fix",
        verifiers=[
            VerifierSpec(
                type="file",
                options={"path": "app.py", "contains": "fixed"},
            ),
            VerifierSpec(
                type="diff",
                options={"allowed_paths": ["app.py"]},
            ),
            VerifierSpec(
                type="run",
                options={
                    "expect_status": "completed",
                    "expect_stop_reason": "final_answer",
                },
            ),
            VerifierSpec(type="trace"),
        ],
    )
    options = EvalRunOptions(
        fixtures_root=fixtures,
        artifact_root=artifacts,
        eval_id="eval-test",
        session_options=CreateAgentSessionOptions(
            workspace_dir=".",
            model=_model(),
            load_workspace_resources=False,
        ),
    )

    result = asyncio.run(service.run_case(case, options))

    assert result.verdict == "passed"
    assert result.run_ids == ["run-1"]
    case_dir = artifacts / "eval-test" / "cases" / "fix-bug"
    assert (case_dir / "result.json").is_file()
    assert (case_dir / "verifier-results.json").is_file()
    assert (case_dir / "workspace.diff").read_text(
        encoding="utf-8"
    ) == "M app.py"


def test_evaluation_service_runs_restart_scenario(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    fixture = fixtures / "tiny"
    fixture.mkdir(parents=True)
    (fixture / "app.py").write_text("one\n", encoding="utf-8")
    factory = _SharedFakeRuntimeFactory()
    service = EvaluationService(runtime_factory=factory)
    scenario = EvalScenario(
        id="restart",
        fixture="tiny",
        steps=[
            ScenarioStep(type="prompt", options={"text": "inspect"}),
            ScenarioStep(type="restart"),
            ScenarioStep(
                type="modify_file",
                options={"path": "external.txt", "content": "changed\n"},
            ),
            ScenarioStep(type="prompt", options={"text": "finish"}),
        ],
        verifiers=[
            VerifierSpec(
                type="file",
                options={"path": "external.txt", "contains": "changed"},
            )
        ],
    )
    options = EvalRunOptions(
        fixtures_root=fixtures,
        artifact_root=tmp_path / "artifacts",
        eval_id="eval-scenario",
        session_options=CreateAgentSessionOptions(
            workspace_dir=".",
            model=_model(),
            load_workspace_resources=False,
        ),
    )

    result = asyncio.run(service.run_scenario(scenario, options))

    assert result.verdict == "passed"
    assert result.session_id == "session-1"
    assert result.run_ids == ["run-1", "run-2"]


def test_evaluation_service_runs_with_session_scripted_stream(
    tmp_path: Path,
) -> None:
    fixtures = tmp_path / "fixtures"
    fixture = fixtures / "tiny"
    fixture.mkdir(parents=True)
    (fixture / "app.py").write_text("unchanged\n", encoding="utf-8")

    def scripted_stream(*_args):
        stream = AssistantMessageEventStream()
        stream.end(
            AssistantMessage(
                content=[TextContent(text="done")],
                stop_reason="stop",
            )
        )
        return stream

    service = EvaluationService()
    case = EvalCase(
        id="scripted-runtime",
        category="harness",
        fixture="tiny",
        prompt="respond deterministically",
        verifiers=[
            VerifierSpec(
                type="run",
                options={
                    "expect_status": "completed",
                    "expect_stop_reason": "final_answer",
                    "expect_tool_calls": 0,
                },
            ),
            VerifierSpec(type="trace"),
        ],
    )
    options = EvalRunOptions(
        fixtures_root=fixtures,
        artifact_root=tmp_path / "artifacts",
        eval_id="eval-scripted",
        session_options=CreateAgentSessionOptions(
            workspace_dir=".",
            model=_model(),
            stream_fn=scripted_stream,
            load_workspace_resources=False,
        ),
    )

    result = asyncio.run(service.run_case(case, options))

    assert result.verdict == "passed"
    assert len(result.run_ids) == 1
    run_refs = json.loads(
        (
            tmp_path
            / "artifacts"
            / "eval-scripted"
            / "cases"
            / "scripted-runtime"
            / "run-refs.json"
        ).read_text(encoding="utf-8")
    )
    assert run_refs["runs"][0]["run_id"] == result.run_ids[0]


def test_missing_fixture_is_reported_as_invalid_case(tmp_path: Path) -> None:
    service = EvaluationService()
    case = EvalCase(
        id="missing-fixture",
        category="coding",
        fixture="missing",
        prompt="fix",
        verifiers=[VerifierSpec(type="trace")],
    )
    options = EvalRunOptions(
        fixtures_root=tmp_path / "fixtures",
        artifact_root=tmp_path / "artifacts",
        eval_id="eval-invalid",
        session_options=CreateAgentSessionOptions(
            workspace_dir=".",
            model=_model(),
            load_workspace_resources=False,
        ),
    )

    result = asyncio.run(service.run_case(case, options))

    assert result.verdict == "invalid_case"
    assert "Fixture directory not found" in (result.error or "")
    assert (
        tmp_path
        / "artifacts"
        / "eval-invalid"
        / "cases"
        / "missing-fixture"
        / "result.json"
    ).is_file()


class _FakeRuntime:
    def __init__(self, shared: dict | None = None) -> None:
        self.shared = shared or {"next_run": 1}
        self.workspace = Path()
        self.session_id = "session-1"
        self.results: list[dict] = []
        self.events: dict[str, list[dict]] = {}

    def create_session(self, options):
        self.workspace = Path(options.workspace_dir)
        self.session_id = options.session_id or "session-1"
        return SimpleNamespace(session_id=self.session_id)

    async def send_message(self, session_id, message):
        _ = session_id, message
        run_id = f"run-{self.shared['next_run']}"
        self.shared["next_run"] += 1
        app = self.workspace / "app.py"
        if app.exists():
            app.write_text(
                app.read_text(encoding="utf-8").replace("BUG", "fixed"),
                encoding="utf-8",
            )
        events = [
            {"type": "agent_start", "runId": run_id, "timestamp": 1},
            {"type": "agent_end", "runId": run_id, "timestamp": 2},
        ]
        result = {
            "run_id": run_id,
            "session_id": self.session_id,
            "status": "completed",
            "stop_reason": "final_answer",
            "counters": {
                "model_attempts": 1,
                "tool_iterations": 0,
                "tool_calls": 0,
            },
            "messages": [],
            "affected_paths": ["app.py"] if app.exists() else [],
            "workspace_changed": app.exists(),
            "verification": [],
        }
        self.results.append(result)
        self.events[run_id] = events
        for event in events:
            yield event

    async def continue_session(self, session_id):
        async for event in self.send_message(
            session_id,
            SimpleNamespace(text="continue"),
        ):
            yield event

    def list_runs(self, session_id, limit=None):
        _ = session_id
        values = self.results
        return values[-limit:] if limit is not None else values

    def get_run_events(self, session_id, run_id, limit=None):
        _ = session_id
        values = self.events.get(run_id, [])
        return values[-limit:] if limit is not None else values

    def get_session_freshness(self, session_id):
        _ = session_id
        return {
            "status": "valid",
            "checked_paths": [],
            "changed_paths": [],
            "missing_paths": [],
            "workspace_path": str(self.workspace),
        }

    def get_session_status(self, session_id):
        _ = session_id
        return SimpleNamespace(is_running=False)

    async def cancel_run(self, session_id):
        _ = session_id
        return False

    async def aclose_all(self):
        return None


class _SharedFakeRuntimeFactory:
    def __init__(self) -> None:
        self.shared = {"next_run": 1}

    def __call__(self):
        return _FakeRuntime(self.shared)
