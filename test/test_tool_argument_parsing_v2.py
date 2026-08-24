import asyncio

from tool_runtime_testkit import execute_tool


def test_streamed_tool_arguments_are_strictly_finalized() -> None:
    from codepilot.llm.providers.common import finalize_tool_arguments
    from codepilot.protocols import ToolCall

    valid = ToolCall(id="valid", name="read")
    finalize_tool_arguments(valid, '{"path":"README.md"}')
    assert valid.arguments == {"path": "README.md"}
    assert valid.raw_arguments == '{"path":"README.md"}'
    assert "argument_parse_error" not in valid.metadata

    invalid = ToolCall(id="invalid", name="read")
    finalize_tool_arguments(invalid, '{"path":')
    assert invalid.arguments == {}
    assert invalid.raw_arguments == '{"path":'
    assert "argument_parse_error" in invalid.metadata


def test_runtime_returns_validation_result_for_invalid_model_arguments(tmp_path) -> None:
    from codepilot.tools.builtins import create_builtin_registrations
    from codepilot.tools.contracts import ToolExecutionRequest
    from codepilot.tools.registry import ToolRegistry
    from codepilot.tools.runtime import ToolRuntime

    registry = ToolRegistry()
    registration = create_builtin_registrations(tmp_path, enabled_names=["read"])[0]
    registration_id = registry.register(registration)
    result = asyncio.run(
        execute_tool(
            ToolRuntime(registry),
            ToolExecutionRequest(
                run_id="run1",
                session_id="session1",
                tool_call_id="bad_args",
                tool_name="read",
                arguments={},
                raw_arguments='{"path":',
                argument_parse_error="Invalid tool arguments",
                mode="execute",
                registration_id=registration_id,
            )
        )
    )

    assert result.status == "error"
    assert result.error.code == "tool.arguments.invalid_json"
