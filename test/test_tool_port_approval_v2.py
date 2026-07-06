from __future__ import annotations

import asyncio


def test_tool_runtime_port_defers_and_resumes_approval_through_runtime() -> None:
    async def run_case() -> None:
        from codepilot.protocols import TextContent
        from codepilot.tools.policy import DeferredApprovalProvider
        from codepilot.tools.authoring import AgentTool, AgentToolResult, ToolMetadata
        from codepilot.tools.engine import ToolRuntime
        from codepilot.tools.policy import PermissionPolicy
        from codepilot.tools.adapter import ToolRuntimePort
        from codepilot.tools.ports import (
            ToolInvocation,
            ToolResumeDecision,
        )
        from codepilot.tools.authoring import ToolRegistry

        executed: list[dict] = []

        async def execute(tool_call_id, params, signal=None, on_update=None):
            executed.append(params)
            return AgentToolResult(
                tool_call_id=tool_call_id,
                tool_name="danger",
                content=[TextContent(text=f"ran:{params['value']}")],
                affected_paths=["created.txt"],
                workspace_changed=True,
            )

        registry = ToolRegistry()
        registry.register(
            AgentTool(
                name="danger",
                label="Danger",
                description="Mutates the workspace.",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
                execute=execute,
            ),
            metadata=ToolMetadata(
                name="danger",
                category="file",
                read_only=False,
                concurrency_safe=False,
                exclusive=True,
                requires_approval=True,
                risk_level="high",
                resource_scope=(".",),
            ),
        )
        port = ToolRuntimePort(
            ToolRuntime(
                registry=registry,
                permission_policy=PermissionPolicy(mode="workspace-write"),
                approval_provider=DeferredApprovalProvider(),
            )
        )

        deferred = await port.execute(
            ToolInvocation(
                run_id="run1",
                tool_call_id="call1",
                name="danger",
                arguments={"value": "ok"},
            )
        )

        assert deferred.status == "approval_required"
        assert deferred.interruption is not None
        assert deferred.interruption.tool_name == "danger"
        assert executed == []

        resumed = await port.resume(
            ToolResumeDecision(
                approval_id=deferred.interruption.approval_id,
                decision="approve",
                reason="ok",
            )
        )

        assert resumed.status == "success"
        assert resumed.content[0].text == "ran:ok"
        assert resumed.affected_paths == ("created.txt",)
        assert resumed.workspace_changed is True
        assert executed == [{"value": "ok"}]

    asyncio.run(run_case())


def test_tool_runtime_port_denied_resume_does_not_execute_pending_tool() -> None:
    async def run_case() -> None:
        from codepilot.protocols import TextContent
        from codepilot.tools.policy import DeferredApprovalProvider
        from codepilot.tools.authoring import AgentTool, AgentToolResult, ToolMetadata
        from codepilot.tools.engine import ToolRuntime
        from codepilot.tools.policy import PermissionPolicy
        from codepilot.tools.adapter import ToolRuntimePort
        from codepilot.tools.ports import (
            ToolInvocation,
            ToolResumeDecision,
        )
        from codepilot.tools.authoring import ToolRegistry

        executed = False

        async def execute(tool_call_id, params, signal=None, on_update=None):
            nonlocal executed
            executed = True
            return AgentToolResult(content=[TextContent(text="should not run")])

        registry = ToolRegistry()
        registry.register(
            AgentTool(
                name="danger",
                label="Danger",
                description="Mutates the workspace.",
                parameters={"type": "object", "properties": {}},
                execute=execute,
            ),
            metadata=ToolMetadata(
                name="danger",
                category="file",
                read_only=False,
                concurrency_safe=False,
                exclusive=True,
                requires_approval=True,
                risk_level="high",
                resource_scope=(".",),
            ),
        )
        port = ToolRuntimePort(
            ToolRuntime(
                registry=registry,
                permission_policy=PermissionPolicy(mode="workspace-write"),
                approval_provider=DeferredApprovalProvider(),
            )
        )

        deferred = await port.execute(
            ToolInvocation(
                run_id="run1",
                tool_call_id="call1",
                name="danger",
                arguments={},
            )
        )
        assert deferred.interruption is not None

        denied = await port.resume(
            ToolResumeDecision(
                approval_id=deferred.interruption.approval_id,
                decision="deny",
                reason="not now",
            )
        )

        assert denied.status == "denied"
        assert denied.metadata["approval_id"] == deferred.interruption.approval_id
        assert executed is False

    asyncio.run(run_case())


def test_tool_runtime_port_keeps_pending_when_approved_runtime_raises() -> None:
    async def run_case() -> None:
        from codepilot.protocols import TextContent
        from codepilot.tools.adapter import ToolRuntimePort
        from codepilot.tools.authoring import AgentToolResult, ToolRuntimeResult
        from codepilot.tools.ports import ToolInvocation, ToolResumeDecision

        class Registry:
            def list(self):
                return []

        class FlakyRuntime:
            registry = Registry()

            def __init__(self) -> None:
                self.resume_attempts = 0

            async def execute(self, request, *, approval_id=None):
                if approval_id is None:
                    return ToolRuntimeResult(
                        result=AgentToolResult(
                            tool_call_id=request.tool_call_id,
                            tool_name=request.name,
                            content=[TextContent(text="approval needed")],
                            status="approval_required",
                            error_code="approval_required",
                            approval_id="approval_1",
                            details={
                                "status": "approval_required",
                                "approval_id": "approval_1",
                            },
                        ),
                        status="approval_required",
                        is_error=True,
                        approved=False,
                        approval_id="approval_1",
                    )
                self.resume_attempts += 1
                if self.resume_attempts == 1:
                    raise RuntimeError("transient runtime failure")
                return ToolRuntimeResult(
                    result=AgentToolResult(
                        tool_call_id=request.tool_call_id,
                        tool_name=request.name,
                        content=[TextContent(text="resumed")],
                    ),
                    status="success",
                    approved=True,
                    approval_id=approval_id,
                )

        runtime = FlakyRuntime()
        port = ToolRuntimePort(runtime)  # type: ignore[arg-type]
        deferred = await port.execute(
            ToolInvocation(
                run_id="run1",
                tool_call_id="call1",
                name="danger",
                arguments={},
            )
        )
        assert deferred.interruption is not None

        decision = ToolResumeDecision(
            approval_id=deferred.interruption.approval_id,
            decision="approve",
        )
        try:
            await port.resume(decision)
        except RuntimeError as exc:
            assert "transient" in str(exc)
        else:
            raise AssertionError("first resume should raise")

        resumed = await port.resume(decision)

        assert runtime.resume_attempts == 2
        assert resumed.status == "success"
        assert resumed.content[0].text == "resumed"

    asyncio.run(run_case())


def test_tool_runtime_port_runs_before_and_after_hooks_with_protocol_snapshot() -> None:
    async def run_case() -> None:
        from codepilot.protocols import AssistantMessage, TextContent, Tool, UserMessage
        from codepilot.protocols.commands import (
            AfterToolCallContext,
            AfterToolCallResult,
            BeforeToolCallContext,
            ToolHookContextSnapshot,
        )
        from codepilot.tools.authoring import AgentTool, AgentToolResult, ToolMetadata
        from codepilot.tools.engine import ToolRuntime
        from codepilot.tools.adapter import ToolRuntimePort
        from codepilot.tools.ports import ToolInvocation
        from codepilot.tools.authoring import ToolRegistry

        seen: list[tuple[str, str, str]] = []
        executed: list[dict] = []

        def before(ctx: BeforeToolCallContext, signal):
            seen.append(("before", ctx.context.run_id, ctx.context.system_prompt))
            if ctx.args.get("blocked"):
                from codepilot.protocols.commands import BeforeToolCallResult

                return BeforeToolCallResult(block=True, reason="blocked by test hook")
            return None

        def after(ctx: AfterToolCallContext, signal):
            seen.append(("after", ctx.context.session_id or "", ctx.tool_call.name))
            return AfterToolCallResult(
                content=[TextContent(text="hooked")],
                details={"hooked": True},
                is_error=False,
            )

        async def execute(tool_call_id, params, signal=None, on_update=None):
            executed.append(params)
            return AgentToolResult(
                tool_call_id=tool_call_id,
                tool_name="echo",
                content=[TextContent(text="raw")],
                details={"raw": True},
            )

        registry = ToolRegistry()
        registry.register(
            AgentTool(
                name="echo",
                label="Echo",
                description="Echoes text.",
                parameters={"type": "object", "properties": {}},
                execute=execute,
            ),
            metadata=ToolMetadata(
                name="echo",
                category="utility",
                read_only=True,
                concurrency_safe=True,
                exclusive=False,
                requires_approval=False,
                risk_level="low",
                resource_scope=(".",),
            ),
        )
        port = ToolRuntimePort(
            ToolRuntime(registry=registry),
            before_tool_call=before,
            after_tool_call=after,
        )
        snapshot = ToolHookContextSnapshot(
            run_id="run1",
            session_id="session1",
            system_prompt="rules",
            messages=(UserMessage(content="hello"),),
            tools=(Tool(name="echo", description="Echoes text.", parameters={}),),
        )

        blocked = await port.execute(
            ToolInvocation(
                run_id="run1",
                tool_call_id="call_blocked",
                name="echo",
                arguments={"blocked": True},
                assistant_message=AssistantMessage(content=[]),
                context=snapshot,
            )
        )
        assert blocked.status == "denied"
        assert blocked.content[0].text == "blocked by test hook"
        assert executed == []

        observed = await port.execute(
            ToolInvocation(
                run_id="run1",
                tool_call_id="call_ok",
                name="echo",
                arguments={"blocked": False},
                assistant_message=AssistantMessage(content=[]),
                context=snapshot,
            )
        )
        assert observed.status == "success"
        assert observed.content[0].text == "hooked"
        assert observed.metadata["details"] == {"hooked": True}
        assert executed == [{"blocked": False}]
        assert seen == [
            ("before", "run1", "rules"),
            ("before", "run1", "rules"),
            ("after", "session1", "echo"),
        ]

    asyncio.run(run_case())
