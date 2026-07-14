from __future__ import annotations

from codepilot.sessions.context.contracts import (
    CompactSummary,
    ContextItem,
    ContextSourceRef,
)


def test_context_items_separate_layer_and_retention_class() -> None:
    item = ContextItem(
        item_id="goal",
        layer="l1",
        retention="protected",
        content="Current goal: refactor context governance.",
        source=ContextSourceRef(kind="core", ref="core:goal"),
        estimated_tokens=12,
        relevance=100,
        freshness="fresh",
    )

    assert item.layer == "l1"
    assert item.retention == "protected"
    assert item.source.ref == "core:goal"


def test_compact_summary_is_structured_and_renders_source_refs() -> None:
    summary = CompactSummary(
        original_goal="Refactor context governance.",
        user_constraints=("Do not run the full test suite.",),
        decisions=("Use five context layers.",),
        completed_work=("Memory stage completed.",),
        verification_state="Focused tests passed.",
        next_actions=("Implement ContextService.",),
        source_refs=("message:msg_1", "message:msg_5"),
    )

    rendered = summary.render()

    assert "Use five context layers." in rendered
    assert "message:msg_1" in rendered
    assert CompactSummary.from_mapping(summary.to_mapping()) == summary
