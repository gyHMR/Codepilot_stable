优化思路
1.评测与审计记录体系
2.上下文治理
3.记忆管理，在这种代码框架下，应该实现什么样子的记忆，整体应该怎么做
4.工具安全设计，怎么确保实现安全性设计



# Codepilot

一个仿照 Claude Code 的本地 AI 编程智能体，适用于 Agent 学习、Agent实习项目。

## 项目介绍

Codepilot 是一个基于 Python 构建的编程智能体（Coding Agent），参考了 Claude Code 的设计理念：理解代码仓库、编辑代码、运行验证，并保留结构化的运行结果。

**适用场景：**

- **Agent 学习**：通过阅读和修改本项目，理解 LLM Agent 的核心架构（工具调用、会话管理、上下文压缩、重试机制等）
- **Agent 实践**：在此基础上扩展自定义工具、接入新的模型 Provider、实现新的交互模式


**核心特性：**

- 多模型支持：Anthropic、OpenAI、DeepSeek 及任意 OpenAI-compatible API
- 内置编程工具：文件读写、代码搜索、Shell 执行
- 会话管理：持久化、分支、检查点、上下文压缩
- 安全控制：只读模式、危险命令拦截、编辑严格匹配

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/Codepilot.git
cd Codepilot

# 以可编辑模式安装（开发用）
pip install -e ".[dev]"
```

### 配置模型

```bash
# 生成配置模板
codepilot --init-config
```

编辑 `.codepilot/model.local.json`，填入你的 API Key：

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

- `api_key_env`：优先从环境变量读取 Key，比明文 `api_key` 更安全
- 支持的 API 协议：`openai-compatible`、`anthropic-messages`

验证配置：

```bash
codepilot --check-config
```

### 启动

```bash
# 交互式模式（默认）
codepilot

# 单次输出模式
codepilot --mode print --prompt "解释 main 函数的作用"

# 指定工作区目录
codepilot --workspace /path/to/project
```

## 架构概览

```
src/codepilot/
├── interfaces/     # 用户界面层
│   └── cli/        #   CLI 入口、参数解析、交互式/打印/RPC 模式
├── runtime/        # 运行时组装层
│   ├── config.py   #   配置解析与合并
│   ├── factory.py  #   Agent 会话工厂
│   └── resources.py#   工作区资源加载（model.local.json、自定义 prompt/tools）
├── core/           # Agent 核心循环
│   └── agent_loop  #   消息编排、工具调用调度、重试与停止语义
├── tools/          # 工具系统
│   ├── registry.py #   ToolRegistry 工具注册与元数据
│   ├── runtime.py  #   ToolRuntime 执行与权限策略
│   └── builtin/    #   内置工具（file_tools / search_tools / shell_tools）
├── sessions/       # 会话管理
│   ├── session.py  #   AgentSession 生命周期
│   ├── store.py    #   持久化存储
│   ├── compaction.py#  上下文压缩（token/消息数阈值）
│   └── checkpoint.py#  检查点记录
├── llm/            # LLM 调用层
│   └── provider    #   Anthropic / OpenAI / OpenAI-compatible 适配
├── extensions/     # 扩展机制
│   └── mcp/        #   MCP 工具桥接
├── protocols/      # 类型定义与接口协议
└── observability/  # 可观测性
    └── events      #   JSONL 事件归一化、运行摘要
```

### 分层职责

| 层 | 职责 |
|----|------|
| **interfaces** | 用户交互入口，将用户输入转化为内部调用 |
| **runtime** | 组装配置、模型、提示词、工具、会话、钩子和命令 |
| **core** | Agent 循环核心：接收用户消息 → 调用 LLM → 执行工具 → 返回结果 |
| **tools** | 工具注册、执行、权限控制和沙箱隔离 |
| **sessions** | 会话持久化、分支管理、上下文压缩和记忆 |
| **llm** | 模型调用抽象，屏蔽不同 Provider 的 API 差异 |
| **observability** | 运行事件记录和摘要生成 |

## CLI 参数速查

| 参数 | 说明 |
|------|------|
| `--mode` | 运行模式：`interactive`（默认）/ `print` / `rpc` |
| `--workspace` | 工作区目录（默认 `.`） |
| `--provider` | 模型提供商（如 `anthropic`、`openai`、`deepseek`） |
| `--model-id` | 模型 ID |
| `--system-prompt` | 自定义系统提示词 |
| `--thinking-level` | 思考级别：`off` / `minimal` / `low` / `medium` / `high` / `xhigh` |
| `--tool-execution` | 工具执行模式：`parallel`（默认）/ `sequential` |
| `--read-only` | 只读模式，禁用写入/编辑/Shell |
| `--no-tool-events` | 隐藏工具事件输出 |
| `--session-id` | 恢复已有会话 |
| `--init-config` | 生成模型配置模板 |
| `--check-config` | 校验模型配置 |

## 验证

```bash
# 编译检查
python -m compileall -q src/codepilot

# 运行测试
python -m pytest test -q

# CLI 可用性
codepilot --help
```
