from __future__ import annotations

"""Read-only planning discovery for plan mode."""

import json
import time
from typing import Any

from codepilot.protocols import AssistantMessage, TextContent, ToolCall, ToolResultMessage

from ..events import AgentEventEmitter
from ..llm_runner import LLMStreamRunner, StreamFn
from ..tool_coordinator import ToolCallCoordinator
from ..types import AgentContext, AgentLoopConfig
from .contracts import (
    PlanningBudget,
    PlanningBudgetUsage,
    PlanningDiscoveryReport,
    planning_discovery_report_from_mapping,
)


class PlanningDiscovery:
    """Run a small scratch ReAct loop that can only see read-only tools."""

    async def run(
        self,
        context: AgentContext,
        *,
        config: AgentLoopConfig,
        budget: PlanningBudget,
        emitter: AgentEventEmitter,
        stream_fn: StreamFn | None,
        signal: Any | None = None,
    ) -> PlanningDiscoveryReport:
        read_tools = [tool for tool in context.tools if _is_read_only_tool(tool)]
        if not read_tools:
            return PlanningDiscoveryReport(
                status="skipped",
                budget=PlanningBudgetUsage(stop_reason="no_read_only_tools"),
            )

        await emitter.emit(
            {
                "type": "planning_discovery_started",
                "mode": "plan",
                "planning": {
                    "phase": "discovery",
                    "source": "default",
                    "budget": budget.to_signal(),
                    "discovery": None,
                },
            }
        )
        scratch = AgentContext(
            system_prompt=_discovery_system_prompt(context.system_prompt),
            messages=list(context.messages),
            tools=read_tools,
            task_recovery_projection=context.task_recovery_projection,
            task_signal=context.task_signal,
        )
        runner = LLMStreamRunner(config=config, emitter=emitter, stream_fn=stream_fn)
        coordinator = ToolCallCoordinator(config=config, emitter=emitter)
        started_at = time.monotonic()
        model_rounds = 0
        tool_calls_used = 0
        estimated_tokens = 0
        evidence_refs: list[str] = []
        last_report: PlanningDiscoveryReport | None = None

        while model_rounds < budget.max_model_rounds:
            if time.monotonic() - started_at > budget.max_wall_seconds:
                break
            model_rounds += 1
            assistant = await runner.stream_assistant_response(scratch, signal=signal)
            estimated_tokens += _estimate_message_tokens(assistant)
            if assistant.stop_reason == "error":
                last_report = PlanningDiscoveryReport(
                    status="failed",
                    evidence_refs=evidence_refs,
                    budget=PlanningBudgetUsage(
                        model_rounds=model_rounds,
                        tool_calls=tool_calls_used,
                        estimated_tokens=estimated_tokens,
                        stop_reason="model_error",
                    ),
                )
                break
            tool_calls = [
                item for item in assistant.content if isinstance(item, ToolCall)
            ]
            if tool_calls:
                remaining = budget.max_tool_calls - tool_calls_used
                if remaining <= 0:
                    break
                if len(tool_calls) > remaining:
                    assistant = AssistantMessage(
                        content=tool_calls[:remaining],
                        stop_reason=assistant.stop_reason,
                    )
                    tool_calls = tool_calls[:remaining]
                results = await coordinator.execute_batch(
                    scratch,
                    assistant,
                    signal=signal,
                )
                tool_calls_used += len(tool_calls)
                evidence_refs.extend(
                    f"tool:{result.tool_call_id}"
                    for result in results
                    if result.tool_call_id
                )
                estimated_tokens += sum(_estimate_result_tokens(result) for result in results)
                scratch.messages.extend(results)
                await emitter.emit(
                    {
                        "type": "planning_discovery_step",
                        "mode": "plan",
                        "toolResults": results,
                        "planning": {
                            "phase": "discovery",
                            "source": "default",
                            "budget": budget.to_signal(),
                            "discovery": None,
                        },
                    }
                )
                continue
            last_report = _parse_discovery_message(
                assistant,
                evidence_refs=evidence_refs,
                usage=PlanningBudgetUsage(
                    model_rounds=model_rounds,
                    tool_calls=tool_calls_used,
                    estimated_tokens=estimated_tokens,
                    stop_reason="sufficient_evidence",
                ),
            )
            break

        if last_report is None:
            last_report = PlanningDiscoveryReport(
                status="budget_exhausted",
                evidence_refs=evidence_refs,
                budget=PlanningBudgetUsage(
                    model_rounds=model_rounds,
                    tool_calls=tool_calls_used,
                    estimated_tokens=estimated_tokens,
                    stop_reason="budget_exhausted",
                ),
            )

        await emitter.emit(
            {
                "type": "planning_discovery_completed",
                "mode": "plan",
                "planning": {
                    "phase": "discovery",
                    "source": "default",
                    "budget": budget.to_signal(),
                    "discovery": last_report.to_signal(),
                },
            }
        )
        return last_report


def _discovery_system_prompt(base_prompt: str | None) -> str:
    prompt = (base_prompt or "").rstrip()
    discovery = (
        "## Task Planning Discovery\n"
        "You are Codepilot's read-only planning discovery step. Inspect the "
        "codebase only with the tools provided. Do not modify files. When you "
        "have enough evidence, output JSON only with: status, facts, "
        "relevant_files, risks, verification_hints, open_questions."
    )
    return f"{prompt}\n\n{discovery}".strip()


def _parse_discovery_message(
    message: AssistantMessage,
    *,
    evidence_refs: list[str],
    usage: PlanningBudgetUsage,
) -> PlanningDiscoveryReport:
    data = _loads_json_object(_assistant_text(message))
    if isinstance(data, dict):
        data = dict(data)
        data["evidence_refs"] = list(evidence_refs)
        data["budget"] = usage.to_signal()
        return planning_discovery_report_from_mapping(data)
    return PlanningDiscoveryReport(
        status="failed",
        evidence_refs=tuple(evidence_refs),
        budget=PlanningBudgetUsage(
            model_rounds=usage.model_rounds,
            tool_calls=usage.tool_calls,
            estimated_tokens=usage.estimated_tokens,
            stop_reason="invalid_json",
        ),
    )


def _assistant_text(message: AssistantMessage) -> str:
    return "".join(
        block.text for block in message.content if isinstance(block, TextContent)
    ).strip()


def _loads_json_object(text: str) -> object:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _is_read_only_tool(tool: object) -> bool:
    metadata = getattr(tool, "metadata", None)
    return bool(getattr(metadata, "read_only", False))


def _estimate_message_tokens(message: AssistantMessage) -> int:
    return max(1, len(_assistant_text(message)) // 4)


def _estimate_result_tokens(result: ToolResultMessage) -> int:
    text = "".join(
        block.text for block in result.content if isinstance(block, TextContent)
    )
    return max(1, len(text) // 4)


__all__ = ["PlanningDiscovery"]
