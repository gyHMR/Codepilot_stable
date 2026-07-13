from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from codepilot.runtime.environment import RunEnvironmentFactory, RunResourceScope
from codepilot.runtime.sessions import ActiveRunRegistry


def _prepared(*, entry: str = "prompt", continuation: bool = False):
    return SimpleNamespace(
        run_id="run_1",
        session_id="session_1",
        loop_input=SimpleNamespace(entry=entry),
        context_port=object(),
        state_port=object(),
        context_refs={"continuation": "automatic"} if continuation else {},
    )


def test_environment_factory_separates_session_and_run_resources() -> None:
    model = object()
    tools = object()
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
    assert ports.deadline_at_ms == deadline

    resumed = RunEnvironmentFactory().create(_prepared(entry="resume"), model=None, tools=None)
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


def test_run_deadline_reaches_tool_execution_request() -> None:
    async def run_case() -> None:
        from codepilot.core.tool_step import execute_tool_turn
        from codepilot.protocols import ToolCall
        from codepilot.tools.registry import ToolCatalogSnapshot

        class Tools:
            requests = []

            async def execute_batch(self, requests):
                self.requests.extend(requests)
                return []

        tools = Tools()
        deadline = int(time.time() * 1000) + 10_000
        await execute_tool_turn(
            run_id="run_1",
            session_id="session_1",
            current_mode="build",
            tools=tools,  # type: ignore[arg-type]
            tool_calls=[ToolCall(id="call_1", name="read", arguments={})],
            catalog_snapshot=ToolCatalogSnapshot("catalog_1", (), 0),
            deadline_at_ms=deadline,
        )

        assert tools.requests[0].deadline_at_ms == deadline

    asyncio.run(run_case())
