"""OpenAI 标准 Chat Completions 流式 provider。

适配 OpenAI 风格的 Chat Completions 接口（也适用于 DeepSeek 等兼容 API）。
它展示了如何把统一 ToolCall/Message 协议映射到兼容 API。

特点：
1) 使用 SSE data 行消费增量（openai-compatible 协议）；
2) 统一输出 text/thinking/toolcall 事件；
3) 最终组装成 AssistantMessage。

不同网关的推理字段名适配：
- DeepSeek: delta["reasoning_content"]
- OpenAI: delta["reasoning"]
"""

import json
from typing import Any

import httpx

from ..catalog import get_env_api_key
from ..stream import (
    AssistantMessageEventStream,
    SimpleStreamOptions,
    StreamOptions,
    classify_llm_error,
    llm_event,
)
from codepilot.protocols import (
    Context,
    Model,
    TextContent,
    ThinkingContent,
    ToolCall,
)
from .common import empty_assistant_message, finalize_tool_arguments, normalize_usage, parse_partial_json, to_openai_messages, to_openai_tools


def _map_stop_reason(finish_reason: str | None) -> str:
    """将 OpenAI 的 finish_reason 映射为统一的格式。

    OpenAI 原始值 → 统一值：
    - "tool_calls" → "toolUse"
    - "length" → "length"
    - "stop" 或其他 → "stop"
    """
    if finish_reason == "tool_calls":
        return "toolUse"
    if finish_reason == "length":
        return "length"
    return "stop"


def stream_openai_compatible(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    """创建 OpenAI-compatible API 的流式调用。

    发起 SSE 流式请求，解析 OpenAI Chat Completions 的数据行格式，
    输出统一的事件流（text_start/delta/end、thinking_start/delta/end、
    toolcall_start/delta/end、done/error）。

    支持的工具调用格式：
    - OpenAI 标准 (tool_calls 数组)
    - DeepSeek 等兼容 API

    参数:
        model: 模型配置
        context: 上下文（消息 + 系统提示 + 工具）
        options: 流式调用选项

    返回:
        AssistantMessageEventStream 事件流
    """
    stream = AssistantMessageEventStream()
    resolved_options = options or StreamOptions()

    async def _run() -> None:
        out = empty_assistant_message(api=model.api, provider=model.provider, model=model.id)
        try:
            api_key = resolved_options.api_key or get_env_api_key(model.provider)
            if not api_key and model.provider not in {"deepseek"}:
                api_key = get_env_api_key("openai")
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            if model.headers:
                headers.update(model.headers)
            if resolved_options.headers:
                headers.update(resolved_options.headers)

            payload: dict[str, Any] = {
                "model": model.id,
                "messages": to_openai_messages(context),
                "stream": True,
            }
            if resolved_options.temperature is not None:
                payload["temperature"] = resolved_options.temperature
            if resolved_options.max_tokens is not None:
                payload["max_tokens"] = resolved_options.max_tokens
            apply_openai_reasoning_options(payload, model, resolved_options)
            tools = to_openai_tools(context.tools)
            if tools:
                payload["tools"] = tools

            async with httpx.AsyncClient(timeout=resolved_options.timeout_seconds) as client:
                async with client.stream(
                    "POST",
                    f"{model.base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=payload,
                ) as response:
                    if not response.is_success:
                        body = await response.aread()
                        try:
                            response.raise_for_status()
                        except httpx.HTTPStatusError as exc:
                            exc._response_text = body.decode("utf-8", errors="replace")
                            raise
                    response.raise_for_status()
                    stream.push(llm_event("start", partial=out))

                    current_text: TextContent | None = None
                    current_thinking: ThinkingContent | None = None
                    tool_call_index_map: dict[int, ToolCall] = {}
                    tool_call_partial_json: dict[int, str] = {}

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            break

                        chunk = json.loads(raw)
                        if chunk.get("id"):
                            out.response_id = chunk["id"]
                        choice = (chunk.get("choices") or [{}])[0]

                        finish_reason = choice.get("finish_reason")
                        if finish_reason:
                            out.stop_reason = _map_stop_reason(finish_reason)

                        delta = choice.get("delta") or {}

                        # 文本增量
                        text_delta = delta.get("content")
                        if text_delta:
                            if current_text is None:
                                current_text = TextContent(text="")
                                out.content.append(current_text)
                                stream.push(llm_event("text_start", contentIndex=len(out.content) - 1, partial=out))
                            current_text.text += text_delta
                            stream.push(
                                llm_event(
                                    "text_delta",
                                    contentIndex=len(out.content) - 1,
                                    delta=text_delta,
                                    partial=out,
                                )
                            )

                        # 推理增量（支持不同网关的不同字段名）
                        reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning")
                        if reasoning_delta:
                            if current_thinking is None:
                                current_thinking = ThinkingContent(thinking="")
                                out.content.append(current_thinking)
                                stream.push(llm_event("thinking_start", contentIndex=len(out.content) - 1, partial=out))
                            current_thinking.thinking += reasoning_delta
                            stream.push(
                                llm_event(
                                    "thinking_delta",
                                    contentIndex=len(out.content) - 1,
                                    delta=reasoning_delta,
                                    partial=out,
                                )
                            )

                        # 工具调用增量（按 index 聚合多个候选工具调用）
                        for tc_delta in delta.get("tool_calls") or []:
                            index = tc_delta.get("index", 0)
                            tc = tool_call_index_map.get(index)
                            if tc is None:
                                tc = ToolCall(id=tc_delta.get("id", ""), name="", arguments={})
                                tool_call_index_map[index] = tc
                                tool_call_partial_json[index] = ""
                                out.content.append(tc)
                                stream.push(llm_event("toolcall_start", contentIndex=len(out.content) - 1, partial=out))

                            if tc_delta.get("id"):
                                tc.id = tc_delta["id"]
                            fn = tc_delta.get("function") or {}
                            if fn.get("name"):
                                tc.name = fn["name"]
                            if fn.get("arguments"):
                                tool_call_partial_json[index] += fn["arguments"]
                                tc.raw_arguments = tool_call_partial_json[index]
                                tc.arguments = parse_partial_json(tool_call_partial_json[index])
                                stream.push(
                                    llm_event(
                                        "toolcall_delta",
                                        contentIndex=out.content.index(tc),
                                        delta=fn["arguments"],
                                        partial=out,
                                    )
                                )

                        usage = chunk.get("usage")
                        if usage:
                            out.usage.input = usage.get("prompt_tokens", out.usage.input)
                            out.usage.output = usage.get("completion_tokens", out.usage.output)
                            out.usage.total_tokens = usage.get("total_tokens", out.usage.total_tokens)

                    normalize_usage(out.usage)
                    # 收尾事件：发出所有正在进行中的块的 *_end
                    if current_text is not None:
                        stream.push(
                            llm_event(
                                "text_end",
                                contentIndex=out.content.index(current_text),
                                content=current_text.text,
                                partial=out,
                            )
                        )
                    if current_thinking is not None:
                        stream.push(
                            llm_event(
                                "thinking_end",
                                contentIndex=out.content.index(current_thinking),
                                content=current_thinking.thinking,
                                partial=out,
                            )
                        )
                    for index, tc in tool_call_index_map.items():
                        finalize_tool_arguments(tc, tool_call_partial_json[index])
                        stream.push(
                            llm_event(
                                "toolcall_end",
                                contentIndex=out.content.index(tc),
                                toolCall=tc,
                                partial=out,
                            )
                        )

                    stream.push(llm_event("done", reason=out.stop_reason, message=out))
                    stream.end(out)
        except Exception as exc:
            out.stop_reason = "error"
            out.error_message = str(exc)
            out.error_info = classify_llm_error(exc, model)
            stream.push(llm_event("error", reason="error", error=out, errorInfo=out.error_info))
            stream.end(out)

    stream.start_background(_run())
    return stream


def stream_simple_openai_compatible(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """简化的 OpenAI-compatible 流式调用（第一阶段复用标准 stream）。"""
    return stream_openai_compatible(model, context, options)


def apply_openai_reasoning_options(
    payload: dict[str, Any], model: Model, options: SimpleStreamOptions
) -> None:
    """应用 OpenAI/DeepSeek 的推理选项。

    DeepSeek: 使用 thinking={"type": "enabled"}
    OpenAI: 使用 reasoning_effort 参数

    参数:
        payload: API 请求体字典（会被修改）
        model: 模型配置（用于区分 DeepSeek / OpenAI）
        options: 流式选项（含 reasoning level）
    """
    level = getattr(options, "reasoning", None)
    if level is None:
        return
    if model.provider == "deepseek":
        payload["thinking"] = {"type": "enabled"}
        return
    if model.provider == "openai":
        payload["reasoning_effort"] = {
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "high",
        }[level]