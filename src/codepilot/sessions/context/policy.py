from __future__ import annotations

# 新手导读：ContextPressurePolicy 根据 token 估算和工具输出压力判断 normal/tight/critical。
# 关注点：不同压力会影响保留消息数量、工具输出摘要和 checkpoint 行为。

"""上下文压力策略。"""

from dataclasses import dataclass

from codepilot.core import ContextPreparationRequest
from codepilot.protocols import ContextPressure


@dataclass(frozen=True)
class ContextPressurePolicy:
    """根据模型窗口和上下文体积判断本轮治理压力。"""

    safety_margin_tokens: int = 1024
    tight_ratio: float = 0.72
    critical_ratio: float = 0.90
    tool_output_ratio: float = 0.25

    def evaluate(
        self,
        request: ContextPreparationRequest,
        *,
        estimated_tokens: int,
        tool_output_tokens: int,
        history_tokens: int,
    ) -> ContextPressure:
        effective_budget = max(
            128,
            request.model_context_window
            - request.model_max_output_tokens
            - self.safety_margin_tokens,
        )
        pressure_ratio = (
            estimated_tokens / effective_budget
            if effective_budget > 0
            else 1.0
        )
        reasons: list[str] = []
        if tool_output_tokens >= int(effective_budget * self.tool_output_ratio):
            reasons.append("tool_output_pressure")
        if history_tokens >= int(effective_budget * 0.50):
            reasons.append("history_pressure")
        if pressure_ratio >= self.critical_ratio:
            level = "critical"
            reasons.append("critical_budget_pressure")
        elif pressure_ratio >= self.tight_ratio or reasons:
            level = "tight"
            if pressure_ratio >= self.tight_ratio:
                reasons.append("tight_budget_pressure")
        else:
            level = "normal"
        return ContextPressure(
            level=level,
            effective_budget=effective_budget,
            estimated_tokens=max(0, estimated_tokens),
            reasons=_dedupe(reasons),
        )


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


__all__ = ["ContextPressurePolicy"]
