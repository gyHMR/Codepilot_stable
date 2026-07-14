from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from codepilot.core.contracts import ContextPrepareRequest, CoreContextView
from codepilot.core.state import CoreState
from codepilot.llm.ports import ModelDescriptor
from codepilot.protocols import UserMessage
from codepilot.sessions.context import ContextBudgetConfig, ContextService
from codepilot.sessions.context.budget import (
    ContextBudgetExceededError,
    ContextBudgetManager,
)


def test_pressure_thresholds_include_overflow() -> None:
    manager = ContextBudgetManager(
        ContextBudgetConfig(
            context_window=1000,
            max_output_tokens=100,
            safety_margin_tokens=0,
            tight_ratio=0.70,
            critical_ratio=0.85,
        )
    )

    assert manager.assess(raw_tokens=600, conversation_tokens=100).level == "normal"
    assert manager.assess(raw_tokens=630, conversation_tokens=100).level == "tight"
    assert manager.assess(raw_tokens=765, conversation_tokens=100).level == "critical"
    assert manager.assess(raw_tokens=901, conversation_tokens=100).level == "overflow"


def test_l0_overflow_fails_instead_of_clipping_rules(tmp_path: Path) -> None:
    state = CoreState.new("Small task")
    request = ContextPrepareRequest(
        session_id="session_1",
        run_id="run_1",
        purpose="reasoning",
        directive=None,
        messages=(UserMessage(content="Do it."),),
        core_view=CoreContextView.from_state(state, "build"),
        model=ModelDescriptor(provider="unit", model_id="unit"),
        tool_catalog=None,
        seed={"system_prompt": "immutable rule " * 1200},
    )
    service = ContextService(
        workspace_dir=tmp_path,
        session_id="session_1",
        budget_config=ContextBudgetConfig(
            context_window=500,
            max_output_tokens=100,
            safety_margin_tokens=0,
        ),
    )

    with pytest.raises(ContextBudgetExceededError, match="L0"):
        asyncio.run(service.prepare(request))
