from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from codepilot.protocols import Model
from codepilot.runtime.service import (
    RuntimeService,
    SessionBusyError,
    UserInput,
)
from codepilot.runtime.types import CreateAgentSessionOptions


def _model() -> Model:
    return Model(
        id="test-model",
        name="Test Model",
        api="unit-test",
        provider="unit-test",
        base_url="",
        reasoning=False,
        input=["text"],
        context_window=1000,
        max_tokens=100,
    )


def _options(tmp_path: Path, **overrides) -> CreateAgentSessionOptions:
    values = {
        "workspace_dir": tmp_path,
        "model": _model(),
        "load_workspace_resources": False,
    }
    values.update(overrides)
    return CreateAgentSessionOptions(**values)


def test_runtime_assembly_is_registered_with_session(tmp_path: Path) -> None:
    runtime = RuntimeService()
    handle = runtime.create_session(
        _options(tmp_path, read_only_mode=True)
    )
    try:
        assert runtime.get_assembly(handle.session_id) is handle.assembly
        assert handle.assembly.session_options.session_id == handle.session_id
        status = runtime.get_session_status(handle.session_id)
        assert status.permission_mode == "read-only"
        assert status.workspace == str(tmp_path.resolve())
    finally:
        runtime.close_all()


def test_runtime_command_registers_replacement_session(
    tmp_path: Path,
) -> None:
    async def run_case() -> None:
        runtime = RuntimeService()
        handle = runtime.create_session(_options(tmp_path))
        try:
            result = await runtime.execute_command(handle.session_id, "/clear")
            assert result.switched_session is None
            assert result.switched_session_id is not None
            replacement = result.switched_session_id
            assert runtime.get_assembly(replacement).profile.model.id == "test-model"
        finally:
            runtime.close_all()

    asyncio.run(run_case())


def test_runtime_does_not_execute_prompt_hooks_twice(
    tmp_path: Path,
) -> None:
    calls = 0

    def stop_before_model(_context) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("stop before model")

    async def run_case() -> None:
        runtime = RuntimeService()
        handle = runtime.create_session(
            _options(tmp_path, before_prompt_hooks=[stop_before_model])
        )
        try:
            with pytest.raises(RuntimeError, match="stop before model"):
                await runtime.run_message(
                    handle.session_id,
                    UserInput(text="hello"),
                )
            assert calls == 1
            assert runtime.get_session_status(handle.session_id).is_running is False
        finally:
            runtime.close_all()

    asyncio.run(run_case())


def test_runtime_rejects_second_run_for_same_session() -> None:
    async def run_case() -> None:
        class FakeSession:
            session_id = "s1"

            def __init__(self) -> None:
                self.listeners = []
                self.release = asyncio.Event()

            def subscribe(self, listener):
                self.listeners.append(listener)

                def unsubscribe():
                    self.listeners.remove(listener)

                return unsubscribe

            async def run(self, text, *, images=None):
                _ = text, images
                for listener in list(self.listeners):
                    listener({"type": "turn_start"})
                await self.release.wait()

        runtime = RuntimeService()
        fake = FakeSession()
        runtime._sessions[fake.session_id] = fake  # type: ignore[assignment]

        first = runtime.send_message("s1", UserInput(text="first"))
        assert (await anext(first))["type"] == "turn_start"

        second = runtime.send_message("s1", UserInput(text="second"))
        with pytest.raises(SessionBusyError):
            await anext(second)

        fake.release.set()
        with pytest.raises(StopAsyncIteration):
            await anext(first)

    asyncio.run(run_case())


def test_cancel_run_cancels_stream_task() -> None:
    async def run_case() -> None:
        class FakeSession:
            session_id = "s1"

            def __init__(self) -> None:
                self.listeners = []
                self.started = asyncio.Event()

            def subscribe(self, listener):
                self.listeners.append(listener)

                def unsubscribe():
                    self.listeners.remove(listener)

                return unsubscribe

            async def run(self, text, *, images=None):
                _ = text, images
                self.started.set()
                await asyncio.Event().wait()

        runtime = RuntimeService()
        fake = FakeSession()
        runtime._sessions[fake.session_id] = fake  # type: ignore[assignment]

        async def consume() -> None:
            async for _event in runtime.send_message(
                "s1",
                UserInput(text="hello"),
            ):
                pass

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(fake.started.wait(), timeout=1)
        assert await runtime.cancel_run("s1") is True
        with pytest.raises(asyncio.CancelledError):
            await consumer
        assert "s1" not in runtime._active_runs

    asyncio.run(run_case())
