# Extensions Design

`extensions/` is Codepilot's light external capability boundary. It should stay small: external features are loaded first, then normalized before they enter runtime.

## Capability Model

External sources become four capability types:

- Tool: converted to `AgentTool`, then executed through `tools/` permissions and safety checks.
- Command: registered as a runtime slash command, visible through `/help`.
- Prompt: added as guidelines or appended system prompt text.
- Hook: composed into before/after tool or prompt lifecycle pipelines.

Sources can be different, but runtime should not care about their original shape:

```text
Python extension -> tools / commands / prompt / hooks
Markdown skill   -> command + prompt
MCP config       -> proxy tools
```

## Boundaries

- `extensions/` loads and normalizes external capabilities.
- `runtime/` decides when loaded capabilities are assembled into a session.
- `tools/` still owns tool execution, permissions, approval, and sandbox behavior.
- `core/` should only see ordinary tools, prompts, and hooks after assembly.

For this learning project, avoid building a full plugin platform. A few runnable examples are enough to show how the boundary works.
