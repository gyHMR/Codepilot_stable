# Codepilot

Codepilot 是一个本地 AI 编程代理学习项目，聚焦于清晰的 CLI 工作流：理解代码仓库、编辑代码、运行验证，并保留结构化的运行结果。

## 架构概览

Codepilot 使用单一的 `codepilot` Python 命名空间：

- `interfaces/cli` 是主要的用户界面。
- `runtime` 通过 `RuntimeService` 组装配置、模型 provider、提示词、工具、会话、钩子和命令。
- `core` 包含最小化的 Agent 循环。
- `tools` 包含 `ToolRegistry`、`ToolRuntime`、元数据、权限策略、审批钩子、沙箱和内置编程工具。
- `sessions` 管理会话生命周期、存储、上下文压缩、分支、检查点和记忆。
- `core` 管理一次完整的运行（Run），包括模型调用尝试、工具循环保护、重试、停止语义和运行结果。
- `observability` 负责归一化 JSONL 事件并生成运行摘要。

旧的 runtime 导入路径已被移除。新代码应从拥有对应职责的模块导入。

## 本地 CLI

以可编辑模式安装以进行开发：

```bash
pip install -e ".[dev]"
```

运行 CLI：

```bash
codepilot --help
python -m codepilot.interfaces.cli --help
```

## 模型配置

创建项目本地模型配置：

```bash
codepilot --init-config
```

然后直接编辑 `.codepilot/model.local.json`。DeepSeek 和多数第三方网关都使用 `openai-compatible` 协议：

```json
{
  "api": "openai-compatible",
  "provider": "deepseek",
  "model_id": "deepseek-chat",
  "base_url": "https://api.deepseek.com/v1",
  "api_key": "",
  "api_key_env": "DEEPSEEK_API_KEY",
  "context_window": 64000,
  "max_tokens": 8192,
  "reasoning": false,
  "vision": false
}
```

`api_key_env` 对应的环境变量优先于 `api_key`。该本地文件已被 Git 忽略，但可能包含明文密钥，请勿分享。API Key 不会写入会话或事件日志。

检查配置（不发送网络请求）：

```bash
codepilot --check-config
```

## 验证

在进行结构性变更后，运行以下命令进行验证：

```bash
python -m compileall -q src/codepilot
python -m pytest test -q
python -m codepilot.interfaces.cli --help
```

清理依赖边界：

```bash
rg "codepilot[.]interfaces|from [.]\\.[.]interfaces|from interfaces" src/codepilot/runtime src/codepilot/core src/codepilot/tools src/codepilot/sessions
rg "codepilot[.]runtime[.](agent_session|builtin_tools|cli|runner|session_store|serde|memory)|python -m codepilot[.]runtime|runtime[.]runner|runtime[.]cli" src test docs README.md pyproject.toml
```
