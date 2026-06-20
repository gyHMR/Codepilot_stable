from __future__ import annotations

import asyncio

import httpx
import pytest

from codepilot.llm.errors import classify_llm_error
from codepilot.llm.providers.openai_compatible import stream_openai_compatible
from codepilot.protocols import Context, Model


class _ErrorBodyStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'{"error":{"message":"Invalid tool message sequence"}}'

    async def aclose(self) -> None:
        return None


def _model() -> Model:
    return Model(
        id="deepseek-chat",
        name="DeepSeek Chat",
        api="openai-compatible",
        provider="deepseek",
        base_url="https://api.deepseek.com/v1",
        reasoning=False,
        input=["text"],
        context_window=64_000,
        max_tokens=8_192,
    )


def _unread_error_response() -> httpx.Response:
    request = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
    return httpx.Response(400, request=request, stream=_ErrorBodyStream())


def test_classify_llm_error_does_not_mask_unread_streaming_response() -> None:
    response = _unread_error_response()
    error = httpx.HTTPStatusError(
        "400 Bad Request",
        request=response.request,
        response=response,
    )

    info = classify_llm_error(error, _model())

    assert info.kind == "provider_response"
    assert info.status_code == 400
    assert "response_text" not in info.details


def test_openai_stream_reads_error_body_before_classification(monkeypatch) -> None:
    async def run_case() -> None:
        response = _unread_error_response()

        class _StreamContext:
            async def __aenter__(self) -> httpx.Response:
                return response

            async def __aexit__(self, *_args) -> None:
                await response.aclose()

        class _AsyncClient:
            def __init__(self, **_kwargs) -> None:
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args) -> None:
                return None

            def stream(self, *_args, **_kwargs):
                return _StreamContext()

        monkeypatch.setattr(
            "codepilot.llm.providers.openai_compatible.httpx.AsyncClient",
            _AsyncClient,
        )

        event_stream = stream_openai_compatible(_model(), Context(messages=[]))
        message = await asyncio.wait_for(event_stream.result(), timeout=0.5)

        assert message.stop_reason == "error"
        assert message.error_info is not None
        assert message.error_info.details["response_text"] == (
            '{"error":{"message":"Invalid tool message sequence"}}'
        )

    asyncio.run(run_case())


def test_background_provider_failure_is_attached_to_stream() -> None:
    async def run_case() -> None:
        from codepilot.llm.event_stream import AssistantMessageEventStream

        event_stream = AssistantMessageEventStream()

        async def fail() -> None:
            raise RuntimeError("provider task crashed")

        event_stream.start_background(fail())

        with pytest.raises(RuntimeError, match="provider task crashed"):
            await asyncio.wait_for(event_stream.result(), timeout=0.5)

    asyncio.run(run_case())
