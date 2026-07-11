from __future__ import annotations

import asyncio


def test_tool_runtime_defers_and_resumes_approval_directly() -> None:
    from codepilot.protocols import TextContent
    from codepilot.tools.approvals import DeferredApprovalProvider
    from codepilot.tools.contracts import ToolInvocation, ToolResumeDecision, ToolResult
    from codepilot.tools.permissions import PermissionPolicy
    from codepilot.tools.registry import ToolRegistry
    from codepilot.tools.runtime import ToolRuntime

    executed: list[str] = []

    async def execute(request, signal=None, on_update=None):
        _ = signal, on_update
        executed.append(request.tool_call_id)
        return ToolResult(content=[TextContent(text="done")])

    registry = ToolRegistry()
    registry.register(_tool_definition(execute))
    runtime = ToolRuntime(
        registry=registry,
        permission_policy=PermissionPolicy(permission_mode="ask"),
        approval_provider=DeferredApprovalProvider(),
    )

    waiting = asyncio.run(
        runtime.execute(
            ToolInvocation(
                run_id="run1",
                tool_call_id="call1",
                name="write",
                arguments={"path": "demo.txt", "content": "ok"},
            )
        )
    )
    assert waiting.status == "approval_required"
    assert waiting.interruption is not None
    assert executed == []

    approved = asyncio.run(
        runtime.resume(ToolResumeDecision(waiting.interruption.approval_id, "approve"))
    )
    assert approved.status == "success"
    assert approved.content[0].text == "done"
    assert executed == ["call1"]


def test_tool_runtime_denied_resume_keeps_tool_unexecuted() -> None:
    from codepilot.protocols import TextContent
    from codepilot.tools.approvals import DeferredApprovalProvider
    from codepilot.tools.contracts import ToolInvocation, ToolResumeDecision, ToolResult
    from codepilot.tools.permissions import PermissionPolicy
    from codepilot.tools.registry import ToolRegistry
    from codepilot.tools.runtime import ToolRuntime

    executed = False

    async def execute(request, signal=None, on_update=None):
        nonlocal executed
        _ = request, signal, on_update
        executed = True
        return ToolResult(content=[TextContent(text="should not run")])

    registry = ToolRegistry()
    registry.register(_tool_definition(execute))
    runtime = ToolRuntime(
        registry=registry,
        permission_policy=PermissionPolicy(permission_mode="ask"),
        approval_provider=DeferredApprovalProvider(),
    )

    waiting = asyncio.run(
        runtime.execute(
            ToolInvocation(
                run_id="run1",
                tool_call_id="call1",
                name="write",
                arguments={"path": "demo.txt", "content": "ok"},
            )
        )
    )
    denied = asyncio.run(
        runtime.resume(ToolResumeDecision(waiting.interruption.approval_id, "deny", "no"))  # type: ignore[union-attr]
    )

    assert denied.status == "denied"
    assert denied.metadata["approved"] is False
    assert executed is False


def test_tool_runtime_before_and_after_hooks_patch_result() -> None:
    from codepilot.protocols import AfterToolCallResult, TextContent
    from codepilot.tools.contracts import ToolInvocation, ToolResult
    from codepilot.tools.permissions import PermissionPolicy
    from codepilot.tools.registry import ToolRegistry
    from codepilot.tools.runtime import ToolRuntime

    async def execute(request, signal=None, on_update=None):
        _ = request, signal, on_update
        return ToolResult(content=[TextContent(text="raw")])

    def after(ctx, _signal):
        assert ctx.tool_call.name == "write"
        return AfterToolCallResult(
            content=[TextContent(text="patched")],
            details={"hook": "after"},
            is_error=False,
        )

    registry = ToolRegistry()
    registry.register(_tool_definition(execute, requires_approval=False))
    runtime = ToolRuntime(
        registry=registry,
        permission_policy=PermissionPolicy(permission_mode="workspace-write"),
        after_tool_call=after,
    )

    observation = asyncio.run(
        runtime.execute(
            ToolInvocation(
                run_id="run1",
                tool_call_id="call1",
                name="write",
                arguments={"path": "demo.txt", "content": "ok"},
            )
        )
    )

    assert observation.status == "success"
    assert observation.content[0].text == "patched"
    assert observation.metadata["details"] == {"hook": "after"}


def test_tool_runtime_missing_approval_id_returns_recoverable_error() -> None:
    from codepilot.tools.contracts import ToolResumeDecision
    from codepilot.tools.permissions import PermissionPolicy
    from codepilot.tools.registry import ToolRegistry
    from codepilot.tools.runtime import ToolRuntime

    runtime = ToolRuntime(ToolRegistry(), permission_policy=PermissionPolicy())

    observation = asyncio.run(
        runtime.resume(ToolResumeDecision("approval_missing", "approve"))
    )

    assert observation.status == "error"
    assert observation.metadata["error_code"] == "approval_not_found"


def _tool_definition(execute, *, requires_approval: bool = False):
    from codepilot.tools.contracts import ToolDefinition, ToolMetadata

    return ToolDefinition(
        name="write",
        label="Write",
        description="Write a file",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        metadata=ToolMetadata(
            name="write",
            category="filesystem",
            read_only=False,
            concurrency_safe=False,
            exclusive=True,
            requires_approval=requires_approval,
            risk_level="medium",
            scopes=("build",),
        ),
        execute=execute,
    )
