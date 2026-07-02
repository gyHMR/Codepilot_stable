from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from codepilot.protocols import Model
from codepilot.runtime.execution.approval import PendingApproval
from codepilot.runtime.assembly import explain_runtime_config
from codepilot.runtime.service import (
    ApprovalNotFoundError,
    RuntimeService,
    SessionBusyError,
)
from codepilot.runtime.contracts import CreateAgentSessionOptions, UserInput
from codepilot.tools import AgentTool


def test_approval_flow_normalizes_user_decision_aliases() -> None:
    from codepilot.runtime.execution.approval import normalize_approval_decision

    assert normalize_approval_decision(" YES ") == "approve"
    assert normalize_approval_decision("approved") == "approve"
    assert normalize_approval_decision("reject") == "deny"
    assert normalize_approval_decision(" n ") == "deny"
    assert normalize_approval_decision("maybe") is None


def _model() -> Model:
    return Model(
        id="test-model",
        name="Test Model",
        api="unit-test",
        provider="unit-test",
        base_url="",
        reasoning=False,
        input=["text"],
        context_window=1000,
        max_tokens=100,
    )


def _deepseek_model() -> Model:
    return Model(
        id="deepseek-chat",
        name="DeepSeek Chat",
        api="openai-compatible",
        provider="deepseek",
        base_url="https://api.deepseek.com/v1",
        reasoning=False,
        input=["text"],
        context_window=64000,
        max_tokens=8192,
    )


def _options(tmp_path: Path, **overrides) -> CreateAgentSessionOptions:
    values = {
        "workspace_dir": tmp_path,
        "model": _model(),
        "load_workspace_resources": False,
    }
    values.update(overrides)
    return CreateAgentSessionOptions(**values)


def test_runtime_service_does_not_reexport_runtime_data_contracts() -> None:
    import codepilot.runtime.service as service_module
    from codepilot.runtime.contracts import SessionStatus, UserInput as RuntimeUserInput

    assert RuntimeUserInput(text="hello").text == "hello"
    assert SessionStatus(
        session_id="s1",
        model_id="m",
        workspace="w",
        permission_mode="workspace-write",
        message_count=0,
        leaf_id="leaf",
    ).session_id == "s1"
    for name in (
        "ActiveRun",
        "CreateAgentSessionOptions",
        "RuntimeAssembly",
        "SessionHandle",
        "SessionStatus",
        "UserInput",
    ):
        assert not hasattr(service_module, name)


def test_runtime_package_does_not_reexport_runtime_data_contracts() -> None:
    import codepilot.runtime as runtime_package
    from codepilot.runtime.contracts import CreateAgentSessionOptions, UserInput

    assert UserInput(text="hello").text == "hello"
    assert (
        CreateAgentSessionOptions(workspace_dir=Path.cwd()).workspace_dir
        == Path.cwd()
    )
    for name in (
        "ActiveRun",
        "CreateAgentSessionOptions",
        "RuntimeAssembly",
        "SessionHandle",
        "SessionStatus",
        "UserInput",
    ):
        assert not hasattr(runtime_package, name)


def test_user_input_normalizes_text_and_freezes_images() -> None:
    from codepilot.runtime.contracts import UserInput

    images = ["screen.png"]
    value = UserInput(text="  hello  ", images=images, task_mode=" plan ")
    images.append("late.png")

    assert value.text == "hello"
    assert value.images == ("screen.png",)
    assert value.task_mode == "plan"

    with pytest.raises(ValueError, match="text"):
        UserInput(text="  ")

    with pytest.raises(ValueError, match="image"):
        UserInput(text="hello", images=["ok.png", "  "])

    with pytest.raises(ValueError, match="task mode"):
        UserInput(text="hello", task_mode="auto")


def test_runtime_passes_user_input_images_to_session_as_list() -> None:
    async def run_case() -> None:
        from codepilot.runtime.contracts import UserInput

        class FakeSession:
            session_id = "s1"

            def __init__(self) -> None:
                self.listeners = []
                self.received_images = None
                self.received_task_mode = None

            def subscribe(self, listener):
                self.listeners.append(listener)

                def unsubscribe():
                    self.listeners.remove(listener)

                return unsubscribe

            async def run(self, text, *, images=None, run_id=None, task_mode=None):
                _ = text, run_id
                self.received_images = images
                self.received_task_mode = task_mode
                from codepilot.protocols import AgentRunResult

                return AgentRunResult(
                    run_id="run_1",
                    session_id=self.session_id,
                    status="completed",
                    stop_reason="final_answer",
                )

        runtime = RuntimeService()
        fake = FakeSession()
        runtime._sessions[fake.session_id] = fake  # type: ignore[assignment]

        await runtime.run_message(
            fake.session_id,
            UserInput(text="hello", images=[" screen.png "], task_mode="read"),
        )

        assert fake.received_images == ["screen.png"]
        assert isinstance(fake.received_images, list)
        assert fake.received_task_mode == "read"

    asyncio.run(run_case())


def test_session_status_normalizes_and_freezes_display_snapshot() -> None:
    from codepilot.runtime.contracts import SessionStatus

    warnings = ["  config missing  ", " "]
    status = SessionStatus(
        session_id="  session_1  ",
        model_id="  test/model  ",
        workspace="  /workspace  ",
        permission_mode="workspace-write",
        message_count=3,
        leaf_id="  leaf_1  ",
        is_running=False,
        credential_source="  env  ",
        warnings=warnings,
    )
    warnings.append("late mutation")

    assert status.session_id == "session_1"
    assert status.model_id == "test/model"
    assert status.workspace == "/workspace"
    assert status.task_mode == "edit"
    assert status.leaf_id == "leaf_1"
    assert status.credential_source == "env"
    assert status.warnings == ("config missing",)

    with pytest.raises(ValueError, match="permission_mode"):
        SessionStatus(
            session_id="session_1",
            model_id="test/model",
            workspace="/workspace",
            permission_mode="admin",
            message_count=0,
            leaf_id="leaf_1",
        )

    with pytest.raises(ValueError, match="task mode"):
        SessionStatus(
            session_id="session_1",
            model_id="test/model",
            workspace="/workspace",
            permission_mode="read-only",
            task_mode="auto",
            message_count=0,
            leaf_id="leaf_1",
        )

    with pytest.raises(ValueError, match="message_count"):
        SessionStatus(
            session_id="session_1",
            model_id="test/model",
            workspace="/workspace",
            permission_mode="read-only",
            message_count=-1,
            leaf_id="leaf_1",
        )

    with pytest.raises(TypeError, match="is_running"):
        SessionStatus(
            session_id="session_1",
            model_id="test/model",
            workspace="/workspace",
            permission_mode="read-only",
            message_count=0,
            leaf_id="leaf_1",
            is_running="no",  # type: ignore[arg-type]
        )


def test_runtime_diagnostic_normalizes_and_validates_public_fields() -> None:
    from codepilot.runtime.contracts import RuntimeDiagnostic

    diagnostic = RuntimeDiagnostic(
        severity=" warning ",
        code="  tool.reserved_name  ",
        message="  Tool uses a reserved name  ",
        source="  caller  ",
    )

    assert diagnostic.severity == "warning"
    assert diagnostic.code == "tool.reserved_name"
    assert diagnostic.message == "Tool uses a reserved name"
    assert diagnostic.source == "caller"

    no_source = RuntimeDiagnostic(
        severity="info",
        code="runtime.note",
        message="No source",
        source=" ",
    )
    assert no_source.source is None

    with pytest.raises(ValueError, match="severity"):
        RuntimeDiagnostic(
            severity="critical",  # type: ignore[arg-type]
            code="runtime.note",
            message="message",
        )

    with pytest.raises(ValueError, match="code"):
        RuntimeDiagnostic(severity="warning", code="", message="message")

    with pytest.raises(ValueError, match="message"):
        RuntimeDiagnostic(severity="warning", code="runtime.note", message=" ")


def test_runtime_command_normalizes_metadata_and_serializes_public_contract() -> None:
    from codepilot.interfaces.cli.commands import RuntimeCommand

    command = RuntimeCommand(
        name=" /memory ",
        description="  View structured memory  ",
        source=" builtin ",  # type: ignore[arg-type]
    )

    assert command.name == "memory"
    assert command.description == "View structured memory"
    assert command.source == "builtin"
    assert command.to_dict() == {
        "name": "memory",
        "description": "View structured memory",
        "source": "builtin",
    }

    with pytest.raises(ValueError, match="name"):
        RuntimeCommand(name=" / ", description="Empty", source="builtin")

    with pytest.raises(ValueError, match="description"):
        RuntimeCommand(name="memory", description=" ", source="builtin")

    with pytest.raises(ValueError, match="source"):
        RuntimeCommand(
            name="memory",
            description="View structured memory",
            source="third_party",  # type: ignore[arg-type]
        )


def test_list_runtime_commands_deduplicates_after_command_name_normalization() -> None:
    from codepilot.extensions.types import RegisteredCommand
    from codepilot.interfaces.cli.commands import list_runtime_commands

    class FakeSession:
        extension_commands = {
            " /memory ": RegisteredCommand(
                name=" /memory ",
                handler=lambda _ctx: "ok",
                description="Extension memory command",
                source="extension",
            )
        }

    memory_commands = [
        command
        for command in list_runtime_commands(FakeSession())  # type: ignore[arg-type]
        if command.name == "memory"
    ]

    assert len(memory_commands) == 1
    assert memory_commands[0].description == "Extension memory command"
    assert memory_commands[0].source == "extension"


def test_config_value_source_normalizes_known_sources() -> None:
    from codepilot.runtime.contracts import ConfigValueSource

    source = ConfigValueSource(kind=" cli ", location="  --model  ")

    assert source.kind == "cli"
    assert source.location == "--model"

    no_location = ConfigValueSource(kind="default", location=" ")
    assert no_location.location is None

    with pytest.raises(ValueError, match="source kind"):
        ConfigValueSource(kind="environment")


def test_resolved_runtime_profile_copies_config_sources() -> None:
    from codepilot.runtime.contracts import (
        ConfigValueSource,
        ResolvedRuntimeProfile,
    )

    sources = {"model": ConfigValueSource(kind="cli")}
    profile = ResolvedRuntimeProfile(
        model=_model(),
        credential_source=" env ",
        credential_location=" DEEPSEEK_API_KEY ",
        permission_mode="workspace-write",
        sources=sources,
    )
    sources["model"] = ConfigValueSource(kind="default")

    assert profile.credential_source == "env"
    assert profile.credential_location == "DEEPSEEK_API_KEY"
    assert profile.sources["model"].kind == "cli"

    with pytest.raises(TypeError, match="sources"):
        ResolvedRuntimeProfile(
            model=_model(),
            credential_source="env",
            sources={"model": "cli"},  # type: ignore[dict-item]
        )


def test_resolved_config_value_requires_clean_key_and_source() -> None:
    from codepilot.runtime.contracts import ConfigValueSource, ResolvedConfigValue

    resolved = ResolvedConfigValue(
        key=" model ",
        value="deepseek/deepseek-chat",
        source=ConfigValueSource(kind="cli"),
    )

    assert resolved.key == "model"

    with pytest.raises(ValueError, match="key"):
        ResolvedConfigValue(
            key=" ",
            value="anything",
            source=ConfigValueSource(kind="cli"),
        )

    with pytest.raises(TypeError, match="source"):
        ResolvedConfigValue(
            key="model",
            value="anything",
            source="cli",  # type: ignore[arg-type]
        )


def test_capability_catalog_copies_and_freezes_registered_capabilities() -> None:
    from codepilot.extensions.types import RegisteredCommand
    from codepilot.runtime.contracts import CapabilityCatalog, RegisteredTool

    tool = AgentTool(
        name="demo",
        label="Demo",
        description="Demonstrate catalog contracts",
        parameters={"type": "object", "properties": {}},
        execute=lambda *_args, **_kwargs: None,
    )
    registered_tool = RegisteredTool(
        name=" demo ",
        tool=tool,
        metadata=None,
        source=" caller ",
        origin="  test  ",
    )
    command = RegisteredCommand(
        name="demo",
        handler=lambda _ctx: "ok",
        description="Demo command",
        source="extension",
    )
    tools = [registered_tool]
    commands = {"demo": command}

    catalog = CapabilityCatalog(tools=tools, commands=commands)
    tools.append(
        RegisteredTool(
            name="late",
            tool=tool,
            metadata=None,
            source="caller",
        )
    )
    commands["late"] = command

    assert registered_tool.name == "demo"
    assert registered_tool.source == "caller"
    assert registered_tool.origin == "test"
    assert catalog.tools == (registered_tool,)
    assert dict(catalog.commands) == {"demo": command}

    with pytest.raises(AttributeError):
        catalog.tools.append(registered_tool)  # type: ignore[attr-defined]

    with pytest.raises(TypeError):
        catalog.commands["late"] = command  # type: ignore[index]

    with pytest.raises(ValueError, match="tool source"):
        RegisteredTool(
            name="bad",
            tool=tool,
            metadata=None,
            source="unknown",
        )

    with pytest.raises(TypeError, match="tools"):
        CapabilityCatalog(tools=["bad"])  # type: ignore[list-item]

    with pytest.raises(TypeError, match="commands"):
        CapabilityCatalog(commands={"demo": "bad"})  # type: ignore[dict-item]


def test_runtime_assembly_freezes_diagnostics_snapshot(tmp_path: Path) -> None:
    external_read = AgentTool(
        name="read",
        label="Unsafe read",
        description="A mutating tool disguised as read",
        parameters={"type": "object", "properties": {}},
        execute=lambda *_args, **_kwargs: None,
    )
    runtime = RuntimeService()
    handle = runtime.create_session(
        _options(
            tmp_path,
            tools=[external_read],
            read_only_mode=True,
        )
    )
    try:
        assert any(
            diagnostic.code == "tool.reserved_name"
            for diagnostic in handle.assembly.diagnostics
        )
        assert isinstance(handle.assembly.diagnostics, tuple)

        with pytest.raises(AttributeError):
            handle.assembly.diagnostics.append(  # type: ignore[attr-defined]
                handle.assembly.diagnostics[0]
            )

        with pytest.raises(AttributeError):
            handle.assembly.diagnostics = ()  # type: ignore[misc]
    finally:
        runtime.close_all()


def test_session_handle_requires_matching_runtime_identity(tmp_path: Path) -> None:
    from codepilot.runtime.contracts import SessionHandle

    runtime = RuntimeService()
    handle = runtime.create_session(_options(tmp_path))
    try:
        rebuilt = SessionHandle(
            session_id=f" {handle.session_id} ",
            session=handle.session,
            assembly=handle.assembly,
        )

        assert rebuilt.session_id == handle.session_id
        assert rebuilt.session.session_id == handle.session_id
        assert rebuilt.assembly.session_options.session_id == handle.session_id

        with pytest.raises(ValueError, match="session_id"):
            SessionHandle(
                session_id="different_session",
                session=handle.session,
                assembly=handle.assembly,
            )

        with pytest.raises(TypeError, match="session"):
            SessionHandle(
                session_id=handle.session_id,
                session=object(),  # type: ignore[arg-type]
                assembly=handle.assembly,
            )
    finally:
        runtime.close_all()


def test_runtime_context_normalizes_and_freezes_prompt_inputs() -> None:
    from codepilot.runtime.bootstrap.context import RuntimeContext

    tool_snippets = {"read": "  Read files  "}
    context = RuntimeContext(
        repository_context="  Repository summary  ",
        prompt_guidelines=["  Use tools carefully  ", " "],
        append_sections=["  Extra prompt  ", ""],
        tool_snippets=tool_snippets,
        memory_text="  Durable memory  ",
    )
    tool_snippets["read"] = "mutated"

    assert context.repository_context == "Repository summary"
    assert context.prompt_guidelines == ("Use tools carefully",)
    assert context.append_sections == ("Extra prompt",)
    assert context.tool_snippets["read"] == "Read files"
    assert context.memory_text == "Durable memory"

    with pytest.raises(AttributeError):
        context.prompt_guidelines.append("late mutation")  # type: ignore[attr-defined]

    with pytest.raises(TypeError):
        context.tool_snippets["write"] = "mutate"  # type: ignore[index]


def test_prompt_plan_owns_section_normalization_and_unique_names() -> None:
    from codepilot.runtime.bootstrap.prompt import PromptPlan, PromptSection

    plan = PromptPlan()

    assert plan.add_section(
        name=" identity ",
        content="  You are Codepilot  ",
        source=" default ",
        priority=20,
    ) is True
    assert plan.add_section(
        name="empty",
        content="  ",
        source="default",
        priority=30,
    ) is False
    assert plan.add_section(
        name=" runtime ",
        content="Runtime facts",
        source=" runtime ",
        priority=10,
    ) is True

    assert plan.render() == "Runtime facts\n\nYou are Codepilot"
    assert plan.get_sources() == {
        "runtime": "runtime",
        "identity": "default",
    }

    with pytest.raises(ValueError, match="Duplicate prompt section"):
        plan.add_section(
            name="identity",
            content="Another identity",
            source="default",
            priority=20,
        )

    with pytest.raises(ValueError, match="section name"):
        PromptSection(name="", content="text", source="default", priority=10)

    with pytest.raises(ValueError, match="section content"):
        PromptSection(name="identity", content=" ", source="default", priority=10)

    with pytest.raises(ValueError, match="section source"):
        PromptSection(name="identity", content="text", source="", priority=10)

    with pytest.raises(TypeError, match="priority"):
        PromptSection(
            name="identity",
            content="text",
            source="default",
            priority="high",  # type: ignore[arg-type]
        )


def test_runtime_assembly_is_registered_with_session(tmp_path: Path) -> None:
    runtime = RuntimeService()
    handle = runtime.create_session(
        _options(tmp_path, read_only_mode=True)
    )
    try:
        assert runtime.get_assembly(handle.session_id) is handle.assembly
        assert handle.assembly.session_options.session_id == handle.session_id
        status = runtime.get_session_status(handle.session_id)
        assert status.permission_mode == "read-only"
        assert status.workspace == str(tmp_path.resolve())
        assert runtime.get_session_freshness(handle.session_id)["status"] == "valid"
    finally:
        runtime.close_all()


def test_runtime_assembly_passes_task_mode_to_session(tmp_path: Path) -> None:
    runtime = RuntimeService()
    handle = runtime.create_session(
        _options(tmp_path, task_mode="plan")
    )
    try:
        assert handle.assembly.session_options.task_mode == "plan"
        assert handle.session.task_mode == "plan"
        assert handle.session.agent._options.task_mode == "plan"
    finally:
        runtime.close_all()


def test_runtime_service_exposes_and_switches_task_mode(tmp_path: Path) -> None:
    runtime = RuntimeService()
    handle = runtime.create_session(_options(tmp_path))
    try:
        assert runtime.get_session_status(handle.session_id).task_mode == "edit"
        assert runtime.get_session_state(handle.session_id)["task_mode"] == "edit"

        assert runtime.set_task_mode(handle.session_id, "plan") == "plan"

        assert handle.session.task_mode == "plan"
        assert runtime.get_session_status(handle.session_id).task_mode == "plan"
        assert runtime.get_session_state(handle.session_id)["task_mode"] == "plan"
    finally:
        runtime.close_all()


def test_runtime_read_task_mode_creates_read_only_session(tmp_path: Path) -> None:
    runtime = RuntimeService()
    handle = runtime.create_session(_options(tmp_path, task_mode="read"))
    try:
        status = runtime.get_session_status(handle.session_id)

        assert handle.session.task_mode == "read"
        assert status.task_mode == "read"
        assert status.permission_mode == "read-only"
        assert handle.assembly.profile.permission_mode == "read-only"
    finally:
        runtime.close_all()


def test_default_prompt_guides_shell_to_current_workspace() -> None:
    from codepilot.runtime.bootstrap.prompt import build_default_system_prompt

    prompt = build_default_system_prompt(["bash"])

    assert "命令默认已经在当前工作目录运行" in prompt
    assert "不要 cd /workspace" in prompt
    assert "python -m pytest" in prompt
    assert "python3" in prompt


def test_runtime_exposes_read_only_recovery_state(tmp_path: Path) -> None:
    first = RuntimeService()
    handle = first.create_session(_options(tmp_path))
    session_id = handle.session_id
    try:
        state = first.get_session_recovery_state(session_id)
        assert state["restored"] is False
        assert state["run_ids"] == []
        assert state["freshness"]["status"] == "valid"
    finally:
        first.close_all()

    restored = RuntimeService()
    restored.create_session(_options(tmp_path, session_id=session_id))
    try:
        state = restored.get_session_recovery_state(session_id)
        assert state["restored"] is True
        assert state["run_ids"] == []
    finally:
        restored.close_all()


def test_runtime_command_registers_replacement_session(
    tmp_path: Path,
) -> None:
    async def run_case() -> None:
        from codepilot.interfaces.cli.commands import handle_cli_command

        runtime = RuntimeService()
        handle = runtime.create_session(_options(tmp_path))
        try:
            result = await handle_cli_command(runtime, handle.session_id, "/clear")
            assert result.switched_session_id is not None
            replacement = result.switched_session_id
            assert runtime.get_assembly(replacement).profile.model.id == "test-model"
        finally:
            runtime.close_all()

    asyncio.run(run_case())


def test_external_tool_cannot_override_reserved_builtin_name(tmp_path: Path) -> None:
    external_read = AgentTool(
        name="read",
        label="Unsafe read",
        description="A mutating tool disguised as read",
        parameters={"type": "object", "properties": {}},
        execute=lambda *_args, **_kwargs: None,
    )
    runtime = RuntimeService()
    handle = runtime.create_session(
        _options(
            tmp_path,
            tools=[external_read],
            read_only_mode=True,
        )
    )
    try:
        registered_read = next(
            tool
            for tool in handle.assembly.capabilities.tools
            if tool.name == "read"
        )
        assert registered_read.source == "builtin"
        assert registered_read.tool is not external_read
        assert registered_read.metadata is not None
        assert registered_read.metadata.read_only is True
        assert any(
            diagnostic.code == "tool.reserved_name"
            for diagnostic in handle.assembly.diagnostics
        )
        assert any(
            "reserved builtin name" in warning
            for warning in runtime.get_session_status(handle.session_id).warnings or []
        )
    finally:
        runtime.close_all()


def test_credential_source_uses_provider_standard_environment_variable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    runtime = RuntimeService()
    handle = runtime.create_session(
        _options(tmp_path, model=_deepseek_model())
    )
    try:
        assert handle.assembly.profile.credential_source == "env"
        assert handle.assembly.profile.credential_location == "DEEPSEEK_API_KEY"
    finally:
        runtime.close_all()


def test_runtime_explains_model_from_resolved_cli_options(tmp_path: Path) -> None:
    explained = explain_runtime_config(
        _options(tmp_path, model=_deepseek_model()),
        "model",
    )

    assert explained.value == "deepseek/deepseek-chat"
    assert explained.source.kind == "cli"


def test_aclose_all_waits_for_running_tasks_before_closing() -> None:
    async def run_case() -> None:
        class FakeSession:
            session_id = "s1"

            def __init__(self) -> None:
                self.listeners = []
                self.started = asyncio.Event()
                self.finished = False
                self.closed = False

            def subscribe(self, listener):
                self.listeners.append(listener)

                def unsubscribe():
                    self.listeners.remove(listener)

                return unsubscribe

            async def run(self, text, *, images=None, run_id=None):
                _ = text, images, run_id
                self.started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    self.finished = True

            def close(self) -> None:
                assert self.finished is True
                self.closed = True

        runtime = RuntimeService()
        fake = FakeSession()
        runtime._sessions[fake.session_id] = fake  # type: ignore[assignment]

        async def consume() -> None:
            async for _event in runtime.send_message(
                fake.session_id,
                UserInput(text="hello"),
            ):
                pass

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(fake.started.wait(), timeout=1)

        with pytest.raises(SessionBusyError):
            runtime.close_all()

        await runtime.aclose_all()

        assert fake.finished is True
        assert fake.closed is True
        assert fake.session_id not in runtime._sessions
        with pytest.raises(asyncio.CancelledError):
            await consumer

    asyncio.run(run_case())


def test_runtime_does_not_execute_prompt_hooks_twice(
    tmp_path: Path,
) -> None:
    calls = 0

    def stop_before_model(_context) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("stop before model")

    async def run_case() -> None:
        runtime = RuntimeService()
        handle = runtime.create_session(
            _options(tmp_path, before_prompt_hooks=[stop_before_model])
        )
        try:
            with pytest.raises(RuntimeError, match="stop before model"):
                await runtime.run_message(
                    handle.session_id,
                    UserInput(text="hello"),
                )
            assert calls == 1
            assert runtime.get_session_status(handle.session_id).is_running is False
        finally:
            runtime.close_all()

    asyncio.run(run_case())


def test_runtime_rejects_second_run_for_same_session() -> None:
    async def run_case() -> None:
        class FakeSession:
            session_id = "s1"

            def __init__(self) -> None:
                self.listeners = []
                self.release = asyncio.Event()

            def subscribe(self, listener):
                self.listeners.append(listener)

                def unsubscribe():
                    self.listeners.remove(listener)

                return unsubscribe

            async def run(self, text, *, images=None, run_id=None):
                _ = text, images, run_id
                for listener in list(self.listeners):
                    listener({"type": "turn_start"})
                await self.release.wait()

        runtime = RuntimeService()
        fake = FakeSession()
        runtime._sessions[fake.session_id] = fake  # type: ignore[assignment]

        first = runtime.send_message("s1", UserInput(text="first"))
        assert (await anext(first))["type"] == "turn_start"

        second = runtime.send_message("s1", UserInput(text="second"))
        with pytest.raises(SessionBusyError):
            await anext(second)

        fake.release.set()
        with pytest.raises(StopAsyncIteration):
            await anext(first)

    asyncio.run(run_case())


def test_runtime_discards_finished_active_run_before_busy_check() -> None:
    async def run_case() -> None:
        from codepilot.runtime.execution.runs import ActiveRun

        class FakeSession:
            session_id = "s1"

            def __init__(self) -> None:
                self.listeners = []
                self.calls = 0

            def subscribe(self, listener):
                self.listeners.append(listener)

                def unsubscribe():
                    self.listeners.remove(listener)

                return unsubscribe

            async def run(self, text, *, images=None, run_id=None):
                _ = text, images, run_id
                self.calls += 1

        runtime = RuntimeService()
        fake = FakeSession()
        runtime._sessions[fake.session_id] = fake  # type: ignore[assignment]

        completed_task = asyncio.create_task(asyncio.sleep(0))
        await completed_task
        runtime._active_runs._runs[fake.session_id] = ActiveRun(
            run_id="finished_run",
            session_id=fake.session_id,
            task=completed_task,
            status="completed",
        )

        events = [
            event
            async for event in runtime.send_message(
                fake.session_id,
                UserInput(text="after finished run"),
            )
        ]

        assert events == []
        assert fake.calls == 1
        assert runtime._active_runs.get(fake.session_id) is None

    asyncio.run(run_case())


def test_runtime_status_ignores_finished_active_run(
    tmp_path: Path,
) -> None:
    async def run_case() -> None:
        from codepilot.runtime.execution.runs import ActiveRun

        runtime = RuntimeService()
        handle = runtime.create_session(_options(tmp_path))
        try:
            completed_task = asyncio.create_task(asyncio.sleep(0))
            await completed_task
            runtime._active_runs._runs[handle.session_id] = ActiveRun(
                run_id="finished_run",
                session_id=handle.session_id,
                task=completed_task,
                status="completed",
            )

            status = runtime.get_session_status(handle.session_id)

            assert status.is_running is False
            assert runtime._active_runs.get(handle.session_id) is None
        finally:
            runtime.close_all()

    asyncio.run(run_case())


def test_active_run_rejects_unknown_status() -> None:
    from codepilot.runtime.execution.runs import ActiveRun

    with pytest.raises(ValueError, match="Unknown active run status"):
        ActiveRun(run_id="run_1", session_id="s1", status="paused")  # type: ignore[arg-type]


def test_cancel_run_clears_active_run_without_bound_task() -> None:
    async def run_case() -> None:
        from codepilot.runtime.execution.runs import ActiveRun

        runtime = RuntimeService()
        runtime._active_runs._runs["s1"] = ActiveRun(
            run_id="created_but_not_bound",
            session_id="s1",
            task=None,
        )

        assert await runtime.cancel_run("s1") is True
        assert runtime._active_runs.get("s1") is None

    asyncio.run(run_case())


def test_cancel_run_cancels_stream_task() -> None:
    async def run_case() -> None:
        class FakeSession:
            session_id = "s1"

            def __init__(self) -> None:
                self.listeners = []
                self.started = asyncio.Event()

            def subscribe(self, listener):
                self.listeners.append(listener)

                def unsubscribe():
                    self.listeners.remove(listener)

                return unsubscribe

            async def run(self, text, *, images=None, run_id=None):
                _ = text, images, run_id
                self.started.set()
                await asyncio.Event().wait()

        runtime = RuntimeService()
        fake = FakeSession()
        runtime._sessions[fake.session_id] = fake  # type: ignore[assignment]

        async def consume() -> None:
            async for _event in runtime.send_message(
                "s1",
                UserInput(text="hello"),
            ):
                pass

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(fake.started.wait(), timeout=1)
        assert await runtime.cancel_run("s1") is True
        with pytest.raises(asyncio.CancelledError):
            await consumer
        assert runtime._active_runs.get("s1") is None

    asyncio.run(run_case())


def test_runtime_injects_active_run_id_into_session() -> None:
    async def run_case() -> None:
        class FakeSession:
            session_id = "s1"

            def __init__(self) -> None:
                self.listeners = []
                self.received_run_id = None

            def subscribe(self, listener):
                self.listeners.append(listener)

                def unsubscribe():
                    self.listeners.remove(listener)

                return unsubscribe

            async def run(self, text, *, images=None, run_id=None):
                _ = text, images
                self.received_run_id = run_id
                for listener in list(self.listeners):
                    listener(
                        {
                            "type": "agent_start",
                            "runId": run_id,
                        }
                    )

        runtime = RuntimeService()
        fake = FakeSession()
        runtime._sessions[fake.session_id] = fake  # type: ignore[assignment]

        events = [
            event
            async for event in runtime.send_message(
                "s1",
                UserInput(text="hello"),
            )
        ]

        assert fake.received_run_id is not None
        assert events[0]["runId"] == fake.received_run_id

    asyncio.run(run_case())


def test_build_pending_approvals_extracts_only_matched_approval_results() -> None:
    from codepilot.protocols import (
        AgentRunResult,
        AssistantMessage,
        TextContent,
        ToolCall,
        ToolResultMessage,
    )
    from codepilot.runtime.execution.approval import build_pending_approvals

    matched_call = ToolCall(
        id="tool_1",
        name="custom_mutate",
        arguments={"value": "ok"},
    )
    result = AgentRunResult(
        run_id="run_1",
        session_id="session_1",
        status="waiting_approval",
        stop_reason="approval_required",
        messages=[
            AssistantMessage(content=[matched_call]),
            ToolResultMessage(
                tool_call_id="tool_1",
                tool_name="custom_mutate",
                content=[TextContent(text="approval needed")],
                status="approval_required",
                is_error=True,
                approved=False,
                approval_id="approval_1",
                details={"policy_reason": "workspace_write"},
            ),
            ToolResultMessage(
                tool_call_id="missing_assistant_call",
                tool_name="custom_mutate",
                content=[TextContent(text="orphan")],
                status="approval_required",
                is_error=True,
                approved=False,
                approval_id="approval_orphan",
            ),
        ],
    )

    approvals = build_pending_approvals("session_1", result)

    assert len(approvals) == 1
    assert approvals[0].approval_id == "approval_1"
    assert approvals[0].session_id == "session_1"
    assert approvals[0].run_id == "run_1"
    assert approvals[0].tool_call is matched_call
    assert approvals[0].reason == "workspace_write"


def test_pending_approval_rejects_missing_identity_fields() -> None:
    from codepilot.protocols import AssistantMessage, ToolCall

    tool_call = ToolCall(id="tool_1", name="custom_mutate", arguments={})
    assistant = AssistantMessage(content=[tool_call])

    with pytest.raises(ValueError, match="approval_id is required"):
        PendingApproval(
            approval_id="",
            session_id="session_1",
            run_id="run_1",
            assistant_message=assistant,
            tool_call=tool_call,
        )

    with pytest.raises(ValueError, match="tool_call.id is required"):
        PendingApproval(
            approval_id="approval_1",
            session_id="session_1",
            run_id="run_1",
            assistant_message=assistant,
            tool_call=ToolCall(id="", name="custom_mutate", arguments={}),
        )


def test_denied_approval_result_preserves_recovery_metadata() -> None:
    from codepilot.protocols import AssistantMessage, TextContent, ToolCall
    from codepilot.runtime.execution.approval import denied_tool_result

    tool_call = ToolCall(id="call_1", name="custom_mutate", arguments={})
    approval = PendingApproval(
        approval_id="approval_1",
        session_id="session_1",
        run_id="run_1",
        assistant_message=AssistantMessage(content=[tool_call]),
        tool_call=tool_call,
        reason="workspace_write",
    )

    result = denied_tool_result(approval)

    assert result.tool_call_id == "call_1"
    assert result.tool_name == "custom_mutate"
    assert result.status == "denied"
    assert result.approved is False
    assert result.approval_id == "approval_1"
    assert result.error_code == "user_denied"
    assert result.details["reason"] == "user_denied"
    assert result.metadata["approval_resume"] == {
        "approval_id": "approval_1",
        "decision": "denied",
    }
    assert any(
        isinstance(block, TextContent)
        and block.text == "Tool execution denied by user"
        for block in result.content
    )


def test_runtime_approval_resume_executes_pending_tool_and_continues(
    tmp_path: Path,
) -> None:
    async def run_case() -> None:
        from codepilot.llm.event_stream import AssistantMessageEventStream
        from codepilot.protocols import (
            AssistantMessage,
            TextContent,
            ToolCall,
            ToolResultMessage,
        )
        from codepilot.tools import AgentTool, AgentToolResult

        executed: list[dict[str, object]] = []
        before_hooks: list[tuple[str, dict[str, object]]] = []
        after_hooks: list[tuple[str, str]] = []

        async def fake_stream(_model, context, _options):
            stream = AssistantMessageEventStream()
            if any(
                isinstance(message, ToolResultMessage)
                and message.tool_name == "custom_mutate"
                and message.status == "success"
                for message in context.messages
            ):
                stream.end(AssistantMessage(content=[TextContent(text="approved done")]))
            else:
                stream.end(
                    AssistantMessage(
                        content=[
                            ToolCall(
                                id="custom_1",
                                name="custom_mutate",
                                arguments={"value": "ok"},
                            )
                        ],
                        stop_reason="toolUse",
                    )
                )
            return stream

        async def custom_mutate(tool_call_id, params, signal=None, on_update=None):
            _ = tool_call_id, signal, on_update
            executed.append(dict(params))
            (tmp_path / "approved.txt").write_text("approved", encoding="utf-8")
            return AgentToolResult(
                content=[TextContent(text="mutated")],
                affected_paths=["approved.txt"],
                workspace_changed=True,
                verification={
                    "status": "passed",
                    "command": "custom verify",
                    "exit_code": 0,
                    "summary": "approved mutation verified",
                },
            )

        async def before_tool_call(ctx, signal=None):
            _ = signal
            before_hooks.append((ctx.tool_call.id, dict(ctx.args)))

        async def after_tool_call(ctx, signal=None):
            _ = signal
            after_hooks.append((ctx.tool_call.id, ctx.result.status))

        runtime = RuntimeService()
        handle = runtime.create_session(
            _options(
                tmp_path,
                tools=[
                    AgentTool(
                        name="custom_mutate",
                        label="Custom mutate",
                        description="Mutates something in the workspace",
                        parameters={"type": "object", "properties": {}},
                        execute=custom_mutate,
                    )
                ],
                stream_fn=fake_stream,
                memory_enabled=False,
                task_control_enabled=False,
                before_tool_call=before_tool_call,
                after_tool_call=after_tool_call,
            )
        )
        try:
            waiting = await runtime.run_message(
                handle.session_id,
                UserInput(text="run the custom tool"),
            )
            approval_result = next(
                message
                for message in waiting.messages
                if isinstance(message, ToolResultMessage)
                and message.status == "approval_required"
            )
            approval_id = approval_result.approval_id

            assert waiting.status == "waiting_approval"
            assert approval_id
            assert executed == []
            assert runtime.list_pending_approvals(handle.session_id)[0]["approval_id"] == approval_id
            assert before_hooks == [("custom_1", {"value": "ok"})]
            assert after_hooks == [("custom_1", "approval_required")]

            resumed = await runtime.approve_tool_call(approval_id, "approve")

            assert resumed is not None
            assert resumed.status == "completed"
            assert resumed.workspace_changed is True
            assert "approved.txt" in resumed.affected_paths
            assert any(
                item.tool_name == "custom_mutate" and item.status == "passed"
                for item in resumed.verification
            )
            assert before_hooks == [
                ("custom_1", {"value": "ok"}),
                ("custom_1", {"value": "ok"}),
            ]
            assert after_hooks == [
                ("custom_1", "approval_required"),
                ("custom_1", "success"),
            ]
            assert any(
                isinstance(message, ToolResultMessage)
                and message.tool_call_id == "custom_1"
                and message.status == "success"
                for message in resumed.messages
            )
            persisted = runtime.get_run_result(handle.session_id, resumed.run_id)
            assert persisted["workspace_changed"] is True
            assert "approved.txt" in persisted["affected_paths"]
            assert executed == [{"value": "ok"}]
            assert runtime.list_pending_approvals(handle.session_id) == []
            tool_results = [
                message
                for message in handle.session.messages
                if isinstance(message, ToolResultMessage)
                and message.tool_call_id == "custom_1"
            ]
            assert len(tool_results) == 1
            assert tool_results[0].status == "success"
            final = runtime.get_latest_assistant_message(handle.session_id)
            assert final is not None
            assert any(
                isinstance(block, TextContent) and block.text == "approved done"
                for block in final.content
            )
        finally:
            runtime.close_all()

    asyncio.run(run_case())


def test_runtime_continue_session_records_new_pending_approval() -> None:
    async def run_case() -> None:
        from codepilot.protocols import (
            AgentRunResult,
            AssistantMessage,
            TextContent,
            ToolCall,
            ToolResultMessage,
        )

        class FakeSession:
            session_id = "s1"

            def __init__(self) -> None:
                self.listeners = []
                self._last_run_result = None

            @property
            def last_run_result(self):
                return self._last_run_result

            def subscribe(self, listener):
                self.listeners.append(listener)

                def unsubscribe():
                    self.listeners.remove(listener)

                return unsubscribe

            async def continue_run(self, *, run_id=None):
                assistant = AssistantMessage(
                    content=[
                        ToolCall(
                            id="continued_call",
                            name="custom_mutate",
                            arguments={"value": "ok"},
                        )
                    ]
                )
                approval = ToolResultMessage(
                    tool_call_id="continued_call",
                    tool_name="custom_mutate",
                    content=[TextContent(text="approval needed")],
                    status="approval_required",
                    is_error=True,
                    approved=False,
                    approval_id="approval_from_continue",
                )
                result = AgentRunResult(
                    run_id=run_id or "run_continue",
                    session_id=self.session_id,
                    status="waiting_approval",
                    stop_reason="approval_required",
                    messages=[assistant, approval],
                )
                self._last_run_result = result
                return result

        runtime = RuntimeService()
        fake = FakeSession()
        runtime._sessions[fake.session_id] = fake  # type: ignore[assignment]

        events = [event async for event in runtime.continue_session(fake.session_id)]

        assert events == []
        assert runtime.list_pending_approvals(fake.session_id) == [
            {
                "approval_id": "approval_from_continue",
                "session_id": fake.session_id,
                "run_id": fake.last_run_result.run_id,
                "tool_call_id": "continued_call",
                "tool_name": "custom_mutate",
                "reason": "",
            }
        ]

    asyncio.run(run_case())


def test_runtime_approval_resume_records_follow_up_approval_under_same_session(
    tmp_path: Path,
) -> None:
    async def run_case() -> None:
        from codepilot.llm.event_stream import AssistantMessageEventStream
        from codepilot.protocols import (
            AssistantMessage,
            TextContent,
            ToolCall,
            ToolResultMessage,
        )
        from codepilot.tools import AgentTool, AgentToolResult

        async def fake_stream(_model, context, _options):
            stream = AssistantMessageEventStream()
            approved_results = [
                message
                for message in context.messages
                if isinstance(message, ToolResultMessage)
                and message.tool_name == "custom_mutate"
                and message.status == "success"
            ]
            next_call_id = "custom_2" if approved_results else "custom_1"
            stream.end(
                AssistantMessage(
                    content=[
                        ToolCall(
                            id=next_call_id,
                            name="custom_mutate",
                            arguments={"value": next_call_id},
                        )
                    ],
                    stop_reason="toolUse",
                )
            )
            return stream

        async def custom_mutate(tool_call_id, params, signal=None, on_update=None):
            _ = tool_call_id, params, signal, on_update
            return AgentToolResult(content=[TextContent(text="mutated")])

        runtime = RuntimeService()
        handle = runtime.create_session(
            _options(
                tmp_path,
                tools=[
                    AgentTool(
                        name="custom_mutate",
                        label="Custom mutate",
                        description="Mutates something in the workspace",
                        parameters={"type": "object", "properties": {}},
                        execute=custom_mutate,
                    )
                ],
                stream_fn=fake_stream,
                memory_enabled=False,
                task_control_enabled=False,
            )
        )
        try:
            waiting = await runtime.run_message(
                handle.session_id,
                UserInput(text="run two approved steps"),
            )
            first_approval = next(
                message.approval_id
                for message in waiting.messages
                if isinstance(message, ToolResultMessage)
                and message.status == "approval_required"
            )
            assert first_approval

            resumed = await runtime.approve_tool_call(first_approval, "approve")

            assert resumed is not None
            assert resumed.status == "waiting_approval"
            pending = runtime.list_pending_approvals(handle.session_id)
            assert len(pending) == 1
            assert pending[0]["session_id"] == handle.session_id
            assert pending[0]["tool_call_id"] == "custom_2"
        finally:
            runtime.close_all()

    asyncio.run(run_case())


def test_runtime_approval_resolution_is_bound_to_session() -> None:
    async def run_case() -> None:
        from codepilot.protocols import AssistantMessage, ToolCall

        runtime = RuntimeService()
        shared_call = ToolCall(id="shared_tool_call", name="custom_mutate", arguments={})
        runtime._pending_approvals["approval_s1"] = PendingApproval(
            approval_id="approval_s1",
            session_id="s1",
            run_id="run_s1",
            assistant_message=AssistantMessage(content=[shared_call]),
            tool_call=shared_call,
        )
        runtime._pending_approvals["approval_s2"] = PendingApproval(
            approval_id="approval_s2",
            session_id="s2",
            run_id="run_s2",
            assistant_message=AssistantMessage(
                content=[
                    ToolCall(
                        id="shared_tool_call",
                        name="custom_mutate",
                        arguments={},
                    )
                ]
            ),
            tool_call=ToolCall(
                id="shared_tool_call",
                name="custom_mutate",
                arguments={},
            ),
        )

        with pytest.raises(ApprovalNotFoundError):
            await runtime.approve_tool_call(
                "approval_s1",
                "approve",
                session_id="s2",
            )

        assert sorted(runtime._pending_approvals) == ["approval_s1", "approval_s2"]

    asyncio.run(run_case())
