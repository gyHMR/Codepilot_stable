from __future__ import annotations

import asyncio


class EchoModelPort:
    async def stream(self, _request):
        from codepilot.llm.ports import LLMCompleted
        from codepilot.protocols import AssistantMessage, TextContent

        yield LLMCompleted(message=AssistantMessage(content=[TextContent(text="web echo")]))


def unit_model():
    from codepilot.protocols import Model

    return Model(
        id="web-runtime",
        name="Web Runtime",
        api="unit-test",
        provider="unit-test",
        base_url="",
        reasoning=False,
        input=["text"],
        context_window=4000,
        max_tokens=500,
    )


def test_real_runtime_prompt_reaches_web_events_and_persistence(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.interfaces.web.service import WebService
        from codepilot.runtime.gateway import RuntimeGateway

        runtime = RuntimeGateway(model_port=EchoModelPort())
        service = WebService(
            runtime=runtime,
            workspace=tmp_path,
            session_open_options={"model": unit_model(), "memory_enabled": False},
        )
        created = await service.create_session()
        session_id = created["session_id"]

        await service.submit_prompt(session_id, "hello from web")
        await service.wait_for_idle(session_id)

        replay = service.events_for(session_id).replay_after(None)
        assert replay.events == ()
        all_events = tuple(service.events_for(session_id)._events)  # noqa: SLF001
        assert any(event.type == "run_finished" for event in all_events)
        messages = service.messages(session_id)
        assert any("web echo" in str(message) for message in messages)
        await service.shutdown()

    asyncio.run(run_case())
