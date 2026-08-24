from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


class FakeGateway:
    def __init__(self) -> None:
        self.opened = []
        self.dispatch_calls = []
        self.closed = []
        self.close_all_called = False
        self.pending = ()

    def open_session(self, intent):
        self.opened.append(intent)
        return SimpleNamespace(session_id=intent.session_id or "new-session")

    def describe(self, session_id):
        return SimpleNamespace(
            status=SimpleNamespace(
                session_id=session_id,
                workspace=str(self.opened[-1].workspace_dir),
                model_id="unit/model",
                permission_mode="workspace-write",
                current_mode="build",
                is_running=False,
                message_count=0,
            ),
            pending_approvals=self.pending,
            session=SimpleNamespace(messages=()),
        )

    async def dispatch(self, session_id, action):
        from codepilot.runtime.actions import ProgressFrame, RunFinishedFrame

        self.dispatch_calls.append((session_id, action))
        yield ProgressFrame(event={"type": "text_delta", "delta": "hello"})
        yield RunFinishedFrame(record=SimpleNamespace(run_id="r1", final_text="hello"))

    def close(self, session_id):
        self.closed.append(session_id)

    async def close_all(self):
        self.close_all_called = True


def test_sessions_use_fixed_workspace_and_open_once(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.interfaces.web.service import WebService

        gateway = FakeGateway()
        service = WebService(runtime=gateway, workspace=tmp_path)

        await service.ensure_open("s1")
        await service.ensure_open("s1")

        assert gateway.opened[0].workspace_dir == tmp_path.resolve()
        assert len(gateway.opened) == 1

    asyncio.run(run_case())


def test_one_prompt_dispatch_broadcasts_to_two_subscribers(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.interfaces.web.service import WebService

        gateway = FakeGateway()
        service = WebService(runtime=gateway, workspace=tmp_path)
        await service.ensure_open("s1")
        hub = service.events_for("s1")
        first, second = hub.subscribe(), hub.subscribe()

        accepted = await service.submit_prompt("s1", "hello")
        await service.wait_for_idle("s1")

        assert accepted.accepted is True
        assert len(gateway.dispatch_calls) == 1
        assert (await first.get()).data["delta"] == "hello"
        assert (await second.get()).data["delta"] == "hello"

    asyncio.run(run_case())


def test_active_run_rejects_second_prompt(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.interfaces.web.service import WebConflict, WebService

        gate = asyncio.Event()

        class BlockingGateway(FakeGateway):
            async def dispatch(self, session_id, action):
                self.dispatch_calls.append((session_id, action))
                await gate.wait()
                if False:
                    yield None

        service = WebService(runtime=BlockingGateway(), workspace=tmp_path)
        await service.ensure_open("s1")
        await service.submit_prompt("s1", "first")
        with pytest.raises(WebConflict, match="runtime.run_active"):
            await service.submit_prompt("s1", "second")
        gate.set()
        await service.wait_for_idle("s1")

    asyncio.run(run_case())


def test_approval_cancel_and_shutdown_map_to_runtime_actions(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.interfaces.web.service import WebService
        from codepilot.runtime.actions import ApprovalDecided, RunCancelled

        gateway = FakeGateway()
        gateway.pending = (SimpleNamespace(approval_id="a1"),)
        service = WebService(runtime=gateway, workspace=tmp_path)
        await service.ensure_open("s1")

        await service.decide_approval("s1", "a1", "approve", "ok")
        await service.wait_for_idle("s1")
        await service.cancel("s1")
        await service.wait_for_idle("s1")
        await service.shutdown()

        assert isinstance(gateway.dispatch_calls[0][1], ApprovalDecided)
        assert isinstance(gateway.dispatch_calls[1][1], RunCancelled)
        assert gateway.close_all_called is True

    asyncio.run(run_case())


def test_background_dispatch_failure_publishes_error_snapshot_and_cleans_task(
    tmp_path,
) -> None:
    async def run_case() -> None:
        from codepilot.interfaces.web.service import WebService

        class BrokenGateway(FakeGateway):
            async def dispatch(self, session_id, action):
                self.dispatch_calls.append((session_id, action))
                raise RuntimeError("dispatch exploded")
                if False:
                    yield None

        service = WebService(runtime=BrokenGateway(), workspace=tmp_path)
        await service.ensure_open("s1")
        service._approval_resolutions["s1"] = "approval-stale"  # noqa: SLF001

        await service.submit_prompt("s1", "hello")
        await service.wait_for_idle("s1")

        events = tuple(service.events_for("s1")._events)  # noqa: SLF001
        assert [event.type for event in events[-2:]] == ["error", "session.snapshot"]
        assert events[-2].data["code"] == "runtime.dispatch_failed"
        assert events[-1].data["execution"]["status"] == "idle"
        assert "s1" not in service._tasks  # noqa: SLF001
        assert "s1" not in service._approval_resolutions  # noqa: SLF001

    asyncio.run(run_case())


def test_recovery_continuation_request_is_forwarded_from_synthetic_wait_state(
    tmp_path,
    monkeypatch,
) -> None:
    async def run_case() -> None:
        from codepilot.interfaces.web.service import WebService
        from codepilot.runtime.actions import ContinuationRequested

        gateway = FakeGateway()
        service = WebService(runtime=gateway, workspace=tmp_path)
        await service.ensure_open("s1")
        monkeypatch.setattr(
            service,
            "_recovery_wait_state",
            lambda _session_id: {
                "run_id": "run-crashed",
                "kind": "continuation",
                "request_id": "recovery:run-crashed",
                "payload": {"reason": "runtime.recovery_required"},
            },
        )

        await service.continue_run("s1", "recovery:run-crashed")
        await service.wait_for_idle("s1")

        assert isinstance(gateway.dispatch_calls[0][1], ContinuationRequested)
        assert gateway.dispatch_calls[0][1].request_id == "recovery:run-crashed"

    asyncio.run(run_case())


def test_projection_exposes_nonwaiting_checkpoint_as_recovery_continuation(
    tmp_path,
    monkeypatch,
) -> None:
    async def run_case() -> None:
        from codepilot.interfaces.web.service import WebService

        service = WebService(runtime=FakeGateway(), workspace=tmp_path)
        await service.ensure_open("s1")
        state = SimpleNamespace(
            current_run_id="run-crashed",
            revision=7,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:01+00:00",
        )
        run = SimpleNamespace(checkpoint=SimpleNamespace(waiting=None))
        monkeypatch.setattr(service, "_state_for", lambda _session_id: state)
        monkeypatch.setattr(
            service._session_states,  # noqa: SLF001
            "get_run",
            lambda _run_id: run,
        )

        projection = service.projection("s1")

        assert projection["execution"] == {
            "run_id": "run-crashed",
            "status": "waiting_continuation",
        }
        assert projection["wait"] == {
            "run_id": "run-crashed",
            "kind": "continuation",
            "request_id": "recovery:run-crashed",
            "payload": {"reason": "runtime.recovery_required"},
        }

    asyncio.run(run_case())
