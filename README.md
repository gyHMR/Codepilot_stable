# Codepilot

Codepilot is a local AI coding-agent project. Its current direction is a Claude Code-style programming assistant with a reusable runtime, structured tool execution, session persistence, event logs, and future Web Console support.

## Architecture Snapshot

Codepilot uses a single `codepilot` Python namespace:

- `interfaces/cli`, `interfaces/web`, and `interfaces/im` adapt user-facing transports.
- `runtime` assembles config, model providers, prompts, tools, sessions, hooks, and commands through `RuntimeService`.
- `core` contains the minimal agent loop.
- `tools` contains `ToolRegistry`, `ToolRuntime`, metadata, permission policy, approval hooks, sandboxing, and builtin coding tools.
- `sessions` owns session lifecycle, store, compaction, branching, checkpoint, and memory.
- `core` owns one complete Run, including model attempts, tool-loop guards, retry, stop semantics, and RunResult.
- `observability` normalizes JSONL events and prepares summaries for Web Console and future eval reports.

Older runtime import paths have been removed. New code should import from the module that owns the responsibility.

## Local CLI

Install in editable mode for development:

```bash
pip install -e ".[dev]"
```

Run the CLI:

```bash
codepilot --help
python -m codepilot.interfaces.cli --help
```

## Docker Quick Start

Create a `.env` file from `docker/env.example`, fill in the required API keys and Feishu credentials, then run:

```bash
docker compose up --build codepilot-im
```

For interactive CLI mode:

```bash
docker compose run --rm codepilot-cli
```

Helper scripts live under `scripts/`:

```bash
./scripts/dev.sh --mode cli
```

## Verification

After structural changes, run:

```bash
python -m compileall -q src/codepilot
python -m pytest test -q
python -m codepilot.interfaces.cli --help
```

For dependency-boundary cleanup:

```bash
rg "codepilot[.]interfaces|from [.]\\.[.]interfaces|from interfaces" src/codepilot/runtime src/codepilot/core src/codepilot/tools src/codepilot/sessions
rg "codepilot[.]runtime[.](agent_session|builtin_tools|cli|runner|session_store|serde|memory)|python -m codepilot[.]runtime|runtime[.]runner|runtime[.]cli" src test docs README.md pyproject.toml
```
