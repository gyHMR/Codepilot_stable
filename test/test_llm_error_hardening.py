from __future__ import annotations


def test_http_error_details_are_redacted_and_keep_provider_request_id() -> None:
    import httpx

    from codepilot.llm.stream import classify_llm_error
    from codepilot.protocols import Model

    request = httpx.Request("POST", "https://api.example.test/chat")
    response = httpx.Response(
        429,
        request=request,
        headers={"x-request-id": "req-123", "retry-after": "2"},
        text='{"authorization":"Bearer secret-token","api_key":"sk-secret"}',
    )
    error = httpx.HTTPStatusError("429 for Bearer secret-token", request=request, response=response)
    model = Model(
        id="unit", name="Unit", api="openai-compatible", provider="openai",
        base_url="https://api.example.test", reasoning=False, input=["text"],
        context_window=4000, max_tokens=500,
    )

    info = classify_llm_error(error, model)
    serialized = str(info)

    assert "secret-token" not in serialized
    assert "sk-secret" not in serialized
    assert info.details["provider_request_id"] == "req-123"
    assert info.details["retry_after"] == "2"
    assert info.details["response_excerpt"]


def test_adapter_normalizes_unexpected_errors_with_run_correlation() -> None:
    import asyncio

    async def run_case() -> None:
        from codepilot.llm.adapter import ProviderModelPort
        from codepilot.llm.ports import LLMCorrelation, LLMFailed, LLMRequest, ModelDescriptor
        from codepilot.protocols import LLMErrorInfo, Model, UserMessage

        async def broken_stream(*_args):
            raise RuntimeError("provider offline")

        port = ProviderModelPort(model=Model(
            id="unit", name="Unit", api="unit-test", provider="unit",
            base_url="", reasoning=False, input=["text"], context_window=4000,
            max_tokens=500,
        ), stream_fn=broken_stream)
        events = [event async for event in port.stream(LLMRequest(
            model=ModelDescriptor(provider="unit", model_id="unit"),
            messages=(UserMessage(content="hello"),),
            correlation=LLMCorrelation(run_id="run-1", session_id="session-1"),
        ))]

        assert isinstance(events[-1], LLMFailed)
        assert isinstance(events[-1].error, LLMErrorInfo)
        assert events[-1].error.details["run_id"] == "run-1"
        assert events[-1].error.details["session_id"] == "session-1"

    asyncio.run(run_case())


def test_empty_timeout_error_keeps_retryable_timeout_classification() -> None:
    import httpx

    from codepilot.llm.stream import classify_llm_error
    from codepilot.protocols import Model

    request = httpx.Request("POST", "https://api.example.test/chat")
    error = httpx.ConnectTimeout("", request=request)
    model = Model(
        id="unit", name="Unit", api="openai-compatible", provider="openai",
        base_url="https://api.example.test", reasoning=False, input=["text"],
        context_window=4000, max_tokens=500,
    )

    info = classify_llm_error(error, model)

    assert info.code == "llm.timeout"
    assert info.kind == "timeout"
    assert info.retryable is True
    assert info.message
    assert info.details["exception_type"] == "ConnectTimeout"
