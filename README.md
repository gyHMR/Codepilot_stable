# Codepilot

Codepilot is a Python coding-agent system with a unified LLM API layer, an agent orchestration core, built-in coding tools, CLI sessions, and a Feishu IM bridge.

## Docker Quick Start

Create a `.env` file from `.env.example`, fill in the required API keys and Feishu credentials, then run:

```bash
docker compose up --build codepilot-im
```

For interactive CLI mode:

```bash
docker compose run --rm codepilot-cli
```
