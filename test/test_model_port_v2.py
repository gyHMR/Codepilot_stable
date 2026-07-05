from __future__ import annotations

import asyncio


def test_provider_model_port_streams_completed_message_from_injected_stream() -> None:
    async def run_case() -> None:
        from codepilot.llm.event_stream import AssistantMessageEventStream
        from codepilot.llm.ports import LLMCompleted, LLMRequest, ModelDescriptor, ProviderModelPort
        from codepilot.protocols import AssistantMessage, Model, TextContent, UserMessage

        seen = {}

        async def fake_stream(model, context, options):
            seen["model"] = model.id
            seen["messages"] = context.messages
            seen["session_id"] = options.session_id
            stream = AssistantMessageEventStream()
            stream.end(AssistantMessage(content=[TextContent(text="done")]))
            return stream

        port = ProviderModelPort(
            model=Model(
                id="unit",
                name="Unit",
                api="unit-test",
                provider="unit-test",
                base_url="",
                reasoning=False,
                input=["text"],
                context_window=4000,
                max_tokens=500,
            ),
            stream_fn=fake_stream,
        )

        events = [
            event
            async for event in port.stream(
                LLMRequest(
                    model=ModelDescriptor(provider="unit-test", model_id="unit"),
                    messages=(UserMessage(content="hello"),),
                    system_prompt="rules",
                    correlation={"session_id": "s1"},
                )
            )
        ]

        assert isinstance(events[-1], LLMCompleted)
        assert events[-1].message.content[0].text == "done"
        assert seen["model"] == "unit"
        assert seen["messages"][-1].content == "hello"
        assert seen["session_id"] == "s1"

    asyncio.run(run_case())


def test_provider_model_port_reports_stream_errors_as_llm_failed() -> None:
    async def run_case() -> None:
        from codepilot.llm.ports import LLMFailed, LLMRequest, ModelDescriptor, ProviderModelPort
        from codepilot.protocols import Model, UserMessage

        async def broken_stream(_model, _context, _options):
            raise RuntimeError("offline")

        port = ProviderModelPort(
            model=Model(
                id="unit",
                name="Unit",
                api="unit-test",
                provider="unit-test",
                base_url="",
                reasoning=False,
                input=["text"],
                context_window=4000,
                max_tokens=500,
            ),
            stream_fn=broken_stream,
        )

        events = [
            event
            async for event in port.stream(
                LLMRequest(
                    model=ModelDescriptor(provider="unit-test", model_id="unit"),
                    messages=(UserMessage(content="hello"),),
                )
            )
        ]

        assert isinstance(events[-1], LLMFailed)
        assert str(events[-1].error) == "offline"

    asyncio.run(run_case())


def test_provider_model_port_applies_model_capabilities_before_provider_call() -> None:
    async def run_case() -> None:
        from codepilot.llm.ports import LLMFailed, LLMOptions, LLMRequest, ModelDescriptor, ProviderModelPort
        from codepilot.protocols import ImageContent, Model, ModelCapabilities, UserMessage

        called = False

        async def should_not_stream(*_args):
            nonlocal called
            called = True
            raise AssertionError("provider should not be called")

        port = ProviderModelPort(
            model=Model(
                id="text-only",
                name="Text Only",
                api="unit-test",
                provider="unit-test",
                base_url="",
                reasoning=False,
                input=["text"],
                context_window=4000,
                max_tokens=500,
                capabilities=ModelCapabilities(vision=False),
            ),
            stream_fn=should_not_stream,
        )

        events = [
            event
            async for event in port.stream(
                LLMRequest(
                    model=ModelDescriptor(provider="unit-test", model_id="text-only"),
                    messages=(UserMessage(content=[ImageContent(data="abc")]),),
                    options=LLMOptions(reasoning="high"),
                )
            )
        ]

        assert called is False
        assert isinstance(events[-1], LLMFailed)
        assert events[-1].error.code == "llm.unsupported_capability"

    asyncio.run(run_case())


def test_provider_model_port_filters_prompt_tools_and_reasoning_by_capability() -> None:
    async def run_case() -> None:
        from codepilot.llm.event_stream import AssistantMessageEventStream
        from codepilot.llm.ports import LLMCompleted, LLMOptions, LLMRequest, ModelDescriptor, ProviderModelPort
        from codepilot.protocols import AssistantMessage, Model, ModelCapabilities, TextContent, UserMessage

        captured = {}

        async def fake_stream(_model, context, options):
            captured["system_prompt"] = context.system_prompt
            captured["tools"] = context.tools
            captured["reasoning"] = options.reasoning
            stream = AssistantMessageEventStream()
            stream.end(AssistantMessage(content=[TextContent(text="ok")]))
            return stream

        port = ProviderModelPort(
            model=Model(
                id="limited",
                name="Limited",
                api="unit-test",
                provider="unit-test",
                base_url="",
                reasoning=False,
                input=["text"],
                context_window=4000,
                max_tokens=500,
                capabilities=ModelCapabilities(
                    tools=False,
                    system_prompt=False,
                    reasoning=False,
                ),
            ),
            stream_fn=fake_stream,
        )

        events = [
            event
            async for event in port.stream(
                LLMRequest(
                    model=ModelDescriptor(provider="unit-test", model_id="limited"),
                    messages=(UserMessage(content="hello"),),
                    system_prompt="hidden",
                    tools=({"name": "read"},),
                    options=LLMOptions(reasoning="high"),
                )
            )
        ]

        assert isinstance(events[-1], LLMCompleted)
        assert captured == {
            "system_prompt": None,
            "tools": [],
            "reasoning": None,
        }

    asyncio.run(run_case())


def test_provider_model_port_uses_complete_path_for_non_streaming_models() -> None:
    async def run_case() -> None:
        from codepilot.llm.ports import LLMCompleted, LLMRequest, ModelDescriptor, ProviderModelPort
        from codepilot.protocols import AssistantMessage, Model, ModelCapabilities, TextContent, UserMessage

        called = False

        async def fake_complete(_model, _context, _options):
            nonlocal called
            called = True
            return AssistantMessage(content=[TextContent(text="complete")])

        async def should_not_stream(*_args):
            raise AssertionError("stream path should not be used")

        port = ProviderModelPort(
            model=Model(
                id="complete-only",
                name="Complete Only",
                api="unit-test",
                provider="unit-test",
                base_url="",
                reasoning=False,
                input=["text"],
                context_window=4000,
                max_tokens=500,
                capabilities=ModelCapabilities(streaming=False),
            ),
            stream_fn=should_not_stream,
            complete_fn=fake_complete,
        )

        events = [
            event
            async for event in port.stream(
                LLMRequest(
                    model=ModelDescriptor(provider="unit-test", model_id="complete-only"),
                    messages=(UserMessage(content="hello"),),
                )
            )
        ]

        assert called is True
        assert isinstance(events[-1], LLMCompleted)
        assert events[-1].message.content[0].text == "complete"

    asyncio.run(run_case())
