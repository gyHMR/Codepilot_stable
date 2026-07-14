from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from codepilot.core.contracts import ModelEntry, ToolResultEntry
from codepilot.runtime.environment import RunEnvironmentFactory, RunResourceScope
from codepilot.runtime.registry import ActiveRunRegistry
from codepilot.tools.results import ToolResult


def _prepared(*, entry=None, continuation: bool = False):
    return SimpleNamespace(
        run_id="run_1",
        session_id="session_1",
        loop_input=SimpleNamespace(entry=entry or ModelEntry()),
        context_port=_Context(),
        state_port=_Boundary(),
        context_refs={"continuation": "automatic"} if continuation else {},
    )


class _Model:
    async def stream(self, _request):
        if False:
            yield None


class _Tools:
    def __init__(self) -> None:
        self.requests = []

    def catalog_snapshot(self, *, mode=None):
        del mode

    def prepare_batch(self, requests):
        self.requests.extend(requests)
        return None

    async def execute_prepared(self, _batch_id):
        return ()


class _Context:
    def prepare(self, _request):
        return None


class _Boundary:
    def commit(self, _boundary):
        return None


def test_environment_factory_separates_session_and_run_resources() -> None:
    model = _Model()
    tools = _Tools()
    deadline = int(time.time() * 1000) + 10_000
    environment = RunEnvironmentFactory().create(
        _prepared(),
        model=model,
        tools=tools,
        deadline_at_ms=deadline,
    )

    ports = environment.ports()
    assert environment.trigger == "prompt"
    assert environment.model is model
    assert environment.tools is tools
    assert environment.resources.cancellation is environment.cancellation
    assert ports.cancellation is environment.cancellation
    assert ports.model is model
    assert ports.tools is not tools

    resumed = RunEnvironmentFactory().create(
        _prepared(
                entry=ToolResultEntry(
                    (
                        ToolResult(
                            "call_1",
                            "read",
                            "success",
                            registration_id="registration_1",
                        ),
                    )
                )
        ),
        model=None,
        tools=None,
    )
    continued = RunEnvironmentFactory().create(
        _prepared(continuation=True),
        model=None,
        tools=None,
    )
    assert resumed.trigger == "resume"
    assert continued.trigger == "continuation"


def test_resource_scope_tracks_tasks_and_releases_once() -> None:
    async def run_case() -> None:
        scope = RunResourceScope()
        started = asyncio.Event()
        order: list[str] = []

        async def worker() -> None:
            started.set()
            await asyncio.Event().wait()

        async def async_cleanup() -> None:
            order.append("async")

        scope.add_cleanup(lambda: order.append("sync"))
        scope.add_cleanup(async_cleanup)
        task = scope.create_task(worker(), name="worker")
        await started.wait()

        assert scope.has_active_tasks
        assert scope.cancel("user_cancelled") is True
        assert scope.cancel("ignored") is False
        await asyncio.gather(task, return_exceptions=True)
        await scope.release()
        await scope.release()

        assert scope.cancellation.reason == "user_cancelled"
        assert scope.released is True
        assert scope.has_active_tasks is False
        assert order == ["async", "sync"]

    asyncio.run(run_case())


def test_resource_scope_deadline_cancels_tracked_task() -> None:
    async def run_case() -> None:
        scope = RunResourceScope(deadline_at_ms=int(time.time() * 1000) + 20)

        async def worker() -> None:
            await asyncio.Event().wait()

        task = scope.create_task(worker())
        result = await asyncio.gather(task, return_exceptions=True)

        assert isinstance(result[0], asyncio.CancelledError)
        assert scope.cancellation.cancelled is True
        assert scope.cancellation.reason == "deadline_exceeded"
        await scope.release()

    asyncio.run(run_case())


def test_deadline_cannot_be_won_by_task_finishing_inside_cancel_grace() -> None:
    async def run_case() -> None:
        scope = RunResourceScope(
            deadline_at_ms=int(time.time() * 1000) + 10,
            cancel_grace_ms=250,
        )

        async def worker() -> str:
            await asyncio.sleep(0.05)
            return "too late"

        result = await asyncio.gather(scope.create_task(worker()), return_exceptions=True)

        assert isinstance(result[0], asyncio.CancelledError)
        assert scope.cancellation.reason == "deadline_exceeded"
        await scope.release()

    asyncio.run(run_case())


def test_active_run_registry_delegates_task_ownership_to_scope() -> None:
    async def run_case() -> None:
        scope = RunResourceScope()
        task = scope.create_task(asyncio.Event().wait())
        registry = ActiveRunRegistry()
        registry.start("session_1", "run_1", scope)

        assert registry.is_running("session_1")
        assert registry.has_active_tasks("session_1")
        assert registry.cancel("session_1", "user_cancelled") == "run_1"
        await asyncio.gather(task, return_exceptions=True)
        assert scope.cancellation.reason == "user_cancelled"
        assert registry.finish("session_1", run_id="run_1") == "run_1"
        await scope.release()

    asyncio.run(run_case())


def test_closing_active_session_waits_for_run_resources_before_controller_close() -> None:
    from codepilot.runtime.gateway import RuntimeGateway
    from codepilot.runtime.registry import RuntimeSession

    async def run_case() -> None:
        closed = False

        class Controller:
            session_id = "session_close_active"

            def close(self) -> None:
                nonlocal closed
                closed = True

        gateway = RuntimeGateway()
        gateway._sessions.add(RuntimeSession(controller=Controller()))  # noqa: SLF001
        scope = RunResourceScope(cancel_grace_ms=0)
        scope.create_task(asyncio.Event().wait())
        gateway._active_runs.start(  # noqa: SLF001
            "session_close_active",
            "run_close_active",
            scope,
        )

        gateway.close("session_close_active")

        assert closed is False
        await scope.release()
        await asyncio.gather(*tuple(gateway._background_tasks))  # noqa: SLF001
        assert closed is True

    asyncio.run(run_case())


def test_run_deadline_reaches_tool_execution_request() -> None:
    from codepilot.tools.contracts import ToolExecutionRequest

    tools = _Tools()
    deadline = int(time.time() * 1000) + 10_000
    environment = RunEnvironmentFactory().create(
        _prepared(),
        model=_Model(),
        tools=tools,
        deadline_at_ms=deadline,
    )
    ports = environment.ports()
    assert ports.tools is not None
    ports.tools.prepare_batch(
        (
            ToolExecutionRequest(
                run_id="run_1",
                session_id="session_1",
                tool_call_id="call_1",
                tool_name="read",
                arguments={},
                mode="execute",
                registration_id="registration_1",
            ),
        )
    )

    assert tools.requests[0].deadline_at_ms == deadline
