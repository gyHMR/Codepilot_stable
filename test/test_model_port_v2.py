from __future__ import annotations

import asyncio


def test_provider_model_port_streams_completed_message_from_injected_stream() -> None:
    async def run_case() -> None:
        from codepilot.llm.stream import AssistantMessageEventStream
        from codepilot.llm.adapter import ProviderModelPort
        from codepilot.llm.ports import LLMCompleted, LLMCorrelation, LLMRequest, ModelDescriptor
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
                    correlation=LLMCorrelation(session_id="s1"),
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
        from codepilot.llm.adapter import ProviderModelPort
        from codepilot.llm.ports import LLMFailed, LLMRequest, ModelDescriptor
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
        assert events[-1].error.message == "offline"
        assert events[-1].error.code == "llm.unknown"

    asyncio.run(run_case())


def test_provider_model_port_converts_provider_error_message_to_llm_failed() -> None:
    async def run_case() -> None:
        from codepilot.llm.adapter import ProviderModelPort
        from codepilot.llm.ports import LLMFailed, LLMRequest, ModelDescriptor
        from codepilot.llm.stream import AssistantMessageEventStream
        from codepilot.protocols import AssistantMessage, LLMErrorInfo, Model, TextContent, UserMessage

        async def provider_error(_model, _context, _options):
            stream = AssistantMessageEventStream()
            stream.end(
                AssistantMessage(
                    content=[TextContent(text="")],
                    error_info=LLMErrorInfo(
                        code="llm.rate_limit",
                        message="too many requests",
                        kind="rate_limit",
                        provider="unit",
                        model="unit",
                    ),
                )
            )
            return stream

        port = ProviderModelPort(
            model=Model(
                id="unit", name="Unit", api="unit-test", provider="unit",
                base_url="", reasoning=False, input=["text"],
                context_window=4000, max_tokens=500,
            ),
            stream_fn=provider_error,
        )
        events = [event async for event in port.stream(LLMRequest(
            model=ModelDescriptor(provider="unit", model_id="unit"),
            messages=(UserMessage(content="hello"),),
        ))]

        assert isinstance(events[-1], LLMFailed)
        assert events[-1].error.code == "llm.rate_limit"

    asyncio.run(run_case())


def test_provider_model_port_applies_model_capabilities_before_provider_call() -> None:
    async def run_case() -> None:
        from codepilot.llm.adapter import ProviderModelPort
        from codepilot.llm.ports import LLMFailed, LLMOptions, LLMRequest, ModelDescriptor
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
        from codepilot.llm.stream import AssistantMessageEventStream
        from codepilot.llm.adapter import ProviderModelPort
        from codepilot.llm.ports import LLMCompleted, LLMOptions, LLMRequest, ModelDescriptor
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
        from codepilot.llm.adapter import ProviderModelPort
        from codepilot.llm.ports import LLMCompleted, LLMRequest, ModelDescriptor
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


def test_provider_model_port_maps_non_streaming_provider_error_to_llm_failed() -> None:
    async def run_case() -> None:
        from codepilot.llm.adapter import ProviderModelPort
        from codepilot.llm.ports import LLMFailed, LLMRequest, ModelDescriptor
        from codepilot.protocols import AssistantMessage, LLMErrorInfo, Model, ModelCapabilities, UserMessage

        async def complete_error(_model, _context, _options):
            return AssistantMessage(
                error_info=LLMErrorInfo(
                    code="llm.auth", message="bad key", kind="auth",
                    provider="unit", model="unit",
                )
            )

        port = ProviderModelPort(model=Model(
            id="unit", name="Unit", api="unit-test", provider="unit",
            base_url="", reasoning=False, input=["text"], context_window=4000,
            max_tokens=500,
            capabilities=ModelCapabilities(streaming=False),
        ), complete_fn=complete_error)
        events = [event async for event in port.stream(LLMRequest(
            model=ModelDescriptor(provider="unit", model_id="unit"),
            messages=(UserMessage(content="hello"),),
        ))]

        assert isinstance(events[-1], LLMFailed)
        assert events[-1].error.code == "llm.auth"

    asyncio.run(run_case())


def test_provider_model_port_passes_explicit_timeout_to_provider() -> None:
    async def run_case() -> None:
        from codepilot.llm.adapter import ProviderModelPort
        from codepilot.llm.ports import LLMOptions, LLMRequest, ModelDescriptor
        from codepilot.llm.stream import AssistantMessageEventStream
        from codepilot.protocols import AssistantMessage, Model, TextContent, UserMessage

        captured = {}

        async def fake_stream(_model, _context, options):
            captured["timeout"] = options.timeout_seconds
            captured["proxy_url"] = options.proxy_url
            stream = AssistantMessageEventStream()
            stream.end(AssistantMessage(content=[TextContent(text="ok")]))
            return stream

        port = ProviderModelPort(model=Model(
            id="unit", name="Unit", api="unit-test", provider="unit",
            base_url="", reasoning=False, input=["text"], context_window=4000,
            max_tokens=500,
        ), stream_fn=fake_stream, proxy_url="http://127.0.0.1:7897")
        _ = [event async for event in port.stream(LLMRequest(
            model=ModelDescriptor(provider="unit", model_id="unit"),
            messages=(UserMessage(content="hello"),),
            options=LLMOptions(timeout_seconds=12.5),
        ))]

        assert captured["timeout"] == 12.5
        assert captured["proxy_url"] == "http://127.0.0.1:7897"

    asyncio.run(run_case())


def test_cancelling_model_port_stream_cancels_provider_background_task() -> None:
    async def run_case() -> None:
        from codepilot.llm.adapter import ProviderModelPort
        from codepilot.llm.ports import LLMRequest, ModelDescriptor
        from codepilot.llm.stream import AssistantMessageEventStream
        from codepilot.protocols import Model, UserMessage

        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def fake_stream(_model, _context, _options):
            stream = AssistantMessageEventStream()

            async def worker():
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

            stream.start_background(worker())
            return stream

        port = ProviderModelPort(model=Model(
            id="unit", name="Unit", api="unit-test", provider="unit",
            base_url="", reasoning=False, input=["text"], context_window=4000,
            max_tokens=500,
        ), stream_fn=fake_stream)
        iterator = port.stream(LLMRequest(
            model=ModelDescriptor(provider="unit", model_id="unit"),
            messages=(UserMessage(content="hello"),),
        ))
        consumer = asyncio.create_task(anext(iterator))
        await started.wait()
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        await iterator.aclose()

        assert cancelled.is_set()

    asyncio.run(run_case())


def test_provider_model_port_maps_started_reasoning_tool_call_and_usage() -> None:
    async def run_case() -> None:
        from codepilot.llm.adapter import ProviderModelPort
        from codepilot.llm.ports import (
            LLMCompleted, LLMReasoningDelta, LLMRequest, LLMStarted,
            LLMToolCallDelta, ModelDescriptor,
        )
        from codepilot.llm.stream import AssistantMessageEventStream, llm_event
        from codepilot.protocols import AssistantMessage, Model, ThinkingContent, ToolCall, Usage, UserMessage

        tool_call = ToolCall(id="call-1", name="read", arguments={"path": "a.py"})

        async def fake_stream(_model, _context, _options):
            stream = AssistantMessageEventStream()
            stream.push(llm_event("start"))
            stream.push(llm_event("thinking_delta", delta="inspect"))
            stream.push(llm_event("toolcall_end", toolCall=tool_call))
            stream.end(AssistantMessage(
                content=[ThinkingContent(thinking="inspect"), tool_call],
                usage=Usage(input=10, output=5),
            ))
            return stream

        port = ProviderModelPort(model=Model(
            id="unit", name="Unit", api="unit-test", provider="unit",
            base_url="", reasoning=True, input=["text"], context_window=4000,
            max_tokens=500,
        ), stream_fn=fake_stream)
        events = [event async for event in port.stream(LLMRequest(
            model=ModelDescriptor(provider="unit", model_id="unit"),
            messages=(UserMessage(content="hello"),),
        ))]

        assert any(isinstance(event, LLMStarted) for event in events)
        assert any(isinstance(event, LLMReasoningDelta) and event.text == "inspect" for event in events)
        assert any(isinstance(event, LLMToolCallDelta) and event.tool_call == tool_call for event in events)
        completed = next(event for event in events if isinstance(event, LLMCompleted))
        assert completed.usage is not None
        assert completed.usage.total_tokens == 15

    asyncio.run(run_case())
