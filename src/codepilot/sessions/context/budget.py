from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codepilot.llm.estimation import ContextUsageCalibrator, estimate_context, estimate_text_tokens
from codepilot.protocols import Message, Tool

from .contracts import ContextBudget, ContextItem, ContextPressure


class ContextBudgetExceededError(RuntimeError):
    pass


@dataclass(frozen=True)
class ContextBudgetConfig:
    context_window: int
    max_output_tokens: int
    safety_margin_tokens: int = 1024
    tight_ratio: float = 0.70
    critical_ratio: float = 0.85
    conversation_compaction_ratio: float = 0.55
    inline_tool_result_ratio: float = 0.08
    single_evidence_tokens: int = 512
    single_memory_tokens: int = 600

    def __post_init__(self) -> None:
        for name, value in (
            ("context_window", self.context_window),
            ("max_output_tokens", self.max_output_tokens),
            ("safety_margin_tokens", self.safety_margin_tokens),
            ("single_evidence_tokens", self.single_evidence_tokens),
            ("single_memory_tokens", self.single_memory_tokens),
        ):
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.context_window <= self.max_output_tokens + self.safety_margin_tokens:
            raise ValueError("context window must exceed output reserve and safety margin")
        if not 0 < self.tight_ratio < self.critical_ratio < 1:
            raise ValueError("pressure ratios must satisfy 0 < tight < critical < 1")
        if not 0 < self.conversation_compaction_ratio <= 1:
            raise ValueError("conversation_compaction_ratio must be in (0, 1]")


class ContextBudgetManager:
    def __init__(self, config: ContextBudgetConfig) -> None:
        self.config = config
        self.budget = ContextBudget(
            effective_input_tokens=(
                config.context_window
                - config.max_output_tokens
                - config.safety_margin_tokens
            ),
            output_reserve_tokens=config.max_output_tokens,
            safety_margin_tokens=config.safety_margin_tokens,
            layer_weights={"l1": 0.10, "l2": 0.35, "l3": 0.10, "l4": 0.45},
        )

    def assess(
        self,
        *,
        raw_tokens: int,
        conversation_tokens: int,
    ) -> ContextPressure:
        effective = self.budget.effective_input_tokens
        ratio = raw_tokens / effective
        conversation_ratio = conversation_tokens / effective
        reasons: list[str] = []
        if ratio > 1:
            level = "overflow"
            reasons.append("input_exceeds_effective_budget")
        elif ratio >= self.config.critical_ratio:
            level = "critical"
            reasons.append("critical_budget_pressure")
        elif ratio >= self.config.tight_ratio:
            level = "tight"
            reasons.append("tight_budget_pressure")
        else:
            level = "normal"
        if conversation_ratio >= self.config.conversation_compaction_ratio:
            reasons.append("conversation_pressure")
        return ContextPressure(
            level=level,  # type: ignore[arg-type]
            raw_tokens=max(0, raw_tokens),
            effective_budget=effective,
            conversation_tokens=max(0, conversation_tokens),
            reasons=tuple(reasons),
        )

    def estimate(
        self,
        *,
        system_prompt: str,
        messages: tuple[Message, ...],
        tools: tuple[Tool, ...],
        correction_factors: dict[str, float] | None = None,
    ) -> int:
        return estimate_context(
            list(messages),
            system_prompt,
            list(tools),
            correction_factors=correction_factors,
        ).total

    def estimate_messages(
        self,
        messages: tuple[Message, ...],
        *,
        correction_factors: dict[str, float] | None = None,
    ) -> int:
        return estimate_context(
            list(messages),
            "",
            correction_factors=correction_factors,
        ).total

    def estimate_item(self, content: str) -> int:
        return estimate_text_tokens(content).total

    def select_items(
        self,
        items: tuple[ContextItem, ...],
        *,
        available_tokens: int,
    ) -> tuple[tuple[ContextItem, ...], tuple[ContextItem, ...]]:
        selected: list[ContextItem] = []
        dropped: list[ContextItem] = []
        used = 0
        l3_used = 0
        l3_limit = int(self.budget.effective_input_tokens * 0.10)
        retention_order = {
            "required": 0,
            "protected": 1,
            "budgeted": 2,
            "discard_first": 3,
        }
        ranked = sorted(
            items,
            key=lambda item: (
                retention_order[item.retention],
                -item.relevance,
                _freshness_rank(item.freshness),
                -item.recency,
                item.item_id,
            ),
        )
        for item in ranked:
            if item.layer == "l3" and l3_used + item.estimated_tokens > l3_limit:
                dropped.append(item)
                continue
            if item.retention == "required":
                if used + item.estimated_tokens > available_tokens:
                    raise ContextBudgetExceededError(
                        f"required Context item exceeds budget: {item.item_id}"
                    )
                selected.append(item)
                used += item.estimated_tokens
                if item.layer == "l3":
                    l3_used += item.estimated_tokens
                continue
            if used + item.estimated_tokens <= available_tokens:
                selected.append(item)
                used += item.estimated_tokens
                if item.layer == "l3":
                    l3_used += item.estimated_tokens
            else:
                dropped.append(item)
        return tuple(selected), tuple(dropped)

    def ensure_final_fit(self, estimated_tokens: int) -> None:
        if estimated_tokens > self.budget.effective_input_tokens:
            raise ContextBudgetExceededError(
                "final Context exceeds effective input budget: "
                f"{estimated_tokens}>{self.budget.effective_input_tokens}"
            )


def calibrate_context_usage(
    *,
    workspace_dir: str | Path,
    provider: str | None,
    model: str | None,
    report: dict[str, object] | None,
    actual_input_tokens: int,
) -> None:
    if not report or actual_input_tokens <= 0:
        return
    raw = report.get("raw_estimate_tokens")
    by_type = report.get("estimation_by_type")
    if not isinstance(raw, int) or raw <= 0:
        return
    calibrator = ContextUsageCalibrator(workspace_dir)
    calibrator.update(
        provider=provider,
        model=model,
        raw_estimate=raw,
        actual_input_tokens=actual_input_tokens,
        breakdown=by_type if isinstance(by_type, dict) else {},
    )


def _freshness_rank(value: str) -> int:
    return {"fresh": 0, "unknown": 1, "stale": 2, "missing": 3}.get(value, 4)


__all__ = [
    "ContextBudgetConfig",
    "ContextBudgetExceededError",
    "ContextBudgetManager",
    "calibrate_context_usage",
]
