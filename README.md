# Codepilot

一个仿照 Claude Code 的本地 AI 编程智能体，适用于 Agent 学习、Agent实习项目。

## 项目介绍

Codepilot 是一个基于 Python 构建的本地编程智能体（Coding Agent），参考了 Claude Code 的设计理念：理解代码仓库、编辑代码、运行验证，并保留结构化的运行结果。

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
codepilot config init
```

编辑 `.codepilot/model.local.json`，填入你的 API Key：

```json
{
  "api": "openai-compatible",
  "provider": "deepseek",
  "model_id": "deepseek-chat",
  "base_url": "https://api.deepseek.com/v1",
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
codepilot config check
```

### 启动

```bash
# 交互式模式（默认）
codepilot

# 单次输出模式
codepilot -p "解释 main 函数的作用"

# 指定工作区目录
codepilot --workspace /path/to/project
```

---

## 架构概览

```
src/codepilot/
├── protocols/      # Layer 0: 类型定义与接口协议
├── core/           # Layer 1: Agent 循环引擎
├── llm/            # Layer 2: LLM Provider 抽象
├── tools/          # Layer 3: 工具系统、权限、安全
├── sessions/       # Layer 4: 会话管理、记忆、上下文、历史
├── runtime/        # Layer 5: 运行时组装与服务门面
├── extensions/     # Layer 6: 扩展系统（Python/Markdown/MCP）
├── interfaces/     # Layer 7: CLI 和 Web 接口
├── evaluation/     # 横切: 评估框架
└── observability/  # 横切: 事件记录与运行摘要
```

### 分层职责

| 层 | 职责 | 关键文件 |
|----|------|---------|
| **protocols** | 共享类型定义，所有层的公共契约 | `types.py`, `events.py` |
| **core** | Agent 循环：消息编排、工具调度、重试与停止语义 | `agent_loop.py` |
| **llm** | 模型调用抽象，Provider 注册与流式响应处理 | `registry.py`, `provider/` |
| **tools** | 工具注册、执行、权限控制和沙箱隔离 | `registry.py`, `runtime.py`, `builtin/` |
| **sessions** | 会话持久化、分支管理、上下文压缩、记忆 | `session.py`, `store.py`, `memory/`, `context/` |
| **runtime** | 8 步组装流水线：配置→模型→工具→提示词→钩子→命令 | `service.py`, `assembly.py` |
| **extensions** | Python 扩展、Markdown 技能、MCP 工具桥接 | `python_ext/`, `skill.py`, `mcp/` |
| **interfaces** | CLI（交互式/打印/RPC）和 Web 接口 | `cli/cli.py`, `cli/runner.py` |
| **evaluation** | 评估框架：加载、执行、断言、指标、报告 | `service.py`, `executor.py`, `metrics.py` |
| **observability** | JSONL 事件归一化、运行摘要生成 | `recorder.py`, `summary.py` |

**依赖方向：** `protocols → llm/tools → core → sessions/observability → extensions → runtime → interfaces`

---

## 每层任务速览

### Layer 0 — protocols（类型定义）
定义所有层共享的数据类型和接口协议，是整个项目的公共契约。

### Layer 1 — core（Agent 循环）
实现 Agent 的核心执行循环：接收用户消息 → 调用 LLM → 解析工具调用 → 执行工具 → 将结果反馈给 LLM → 循环直到完成。

### Layer 2 — llm（模型抽象）
通过 Provider 注册机制屏蔽不同 LLM API 的差异，支持流式响应和统一的错误处理。

### Layer 3 — tools（工具系统）
工具注册与元数据管理、权限策略（read-only / workspace-write / ask）、Shell 命令分类、工作区沙箱、并发调度。

### Layer 4 — sessions（会话管理）
会话生命周期管理、持久化存储、上下文压缩（token/消息数阈值触发）、结构化记忆读写、上下文编译。

### Layer 5 — runtime（运行时组装）
8 步组装流水线将配置、模型、工具、提示词、钩子、命令组装为完整的 Agent 会话，是连接底层能力与上层接口的桥梁。

### Layer 6 — extensions（扩展系统）
支持 Python 模块扩展、Markdown 技能模板、MCP 工具桥接三种扩展方式。

### Layer 7 — interfaces（用户接口）
CLI 接口支持交互式（REPL）、单次输出（print）、JSONL 协议（rpc）三种运行模式。

### 横切 — evaluation（评估框架）
提供证据驱动的 Agent 质量评估，支持确定性检查、真实模型 benchmark、消融实验。

### 横切 — observability（可观测性）
记录 Agent 运行过程中的所有事件（JSONL 格式），生成运行摘要供评估和调试使用。

---

## CLI 参数速查

### 顶层参数

| 参数 | 说明 |
|------|------|
| `-p`, `--prompt` | 单次输出模式的输入提示词 |
| `--cwd`, `--workspace` | 工作区目录（默认 `.`） |
| `--resume` | 恢复已有会话（传入会话 ID） |
| `--model` | 模型标识（如 `deepseek/deepseek-chat`） |
| `--permission-mode` | 权限模式：`read-only` / `workspace-write` / `ask` |
| `--verbose` | 详细输出模式 |
| `--no-color` | 禁用彩色输出 |
| `--version` | 显示版本号 |

### config 子命令

| 命令 | 说明 |
|------|------|
| `codepilot config init` | 生成模型配置模板 |
| `codepilot config show` | 显示当前配置 |
| `codepilot config check` | 校验模型配置 |
| `codepilot config explain` | 解释配置项含义 |

### 交互式斜杠命令

| 命令 | 说明 |
|------|------|
| `/help` | 显示帮助信息 |
| `/status` | 显示当前会话状态 |
| `/session` | 会话管理 |
| `/tree` | 显示工作区目录树 |
| `/fork` | 分叉当前会话 |
| `/new` | 创建新会话 |
| `/switch` | 切换会话 |
| `/clear` | 清空当前对话 |
| `/compact` | 手动触发上下文压缩 |
| `/tools` | 列出可用工具 |
| `/model` | 切换模型 |
| `/usage` | 显示 token 用量 |
| `/exit` | 退出 |

---

## 验证

```bash
# 编译检查
python -m compileall -q src/codepilot

# 运行测试
python -m pytest test -q

# CLI 可用性
codepilot --help
```

---

## 评估模块

### 命令速查

评估模块通过 `python -m codepilot.evaluation` 调用：

| 命令 | 说明 | 是否需要模型 |
|------|------|-------------|
| `check` | 确定性检查，快速验证 Agent 行为是否符合预期 | 否 |
| `run` | 真实模型 benchmark，运行完整的评估套件 | 是 |
| `experiment` | 消融实验，对比开启/关闭某模块的效果 | 是 |
| `report` | 显示已有的评估报告 | 否 |

### 示例

```bash
# 快速确定性检查（无需模型，几秒完成）
python -m codepilot.evaluation check

# 运行完整 benchmark（需要模型 API）
python -m codepilot.evaluation run

# 消融实验：对比开启/关闭记忆模块
python -m codepilot.evaluation experiment --module memory

# 查看已有报告
python -m codepilot.evaluation report
```

### 4 模块 12 指标

| 模块 | 指标 | 说明 |
|------|------|------|
| **上下文治理** | key_context_hit_rate | 关键上下文命中率 |
| | token_efficiency | Token 使用效率 |
| | stale_context_rate | 过期上下文占比 |
| **记忆** | memory_retrieval_hit_rate | 记忆检索命中率 |
| | redundant_read_count | 冗余读取次数 |
| | failed_attempt_recurrence_rate | 失败尝试复发率 |
| **任务规划** | evidence_coverage_rate | 证据覆盖率 |
| | false_completion_rate | 误完成率 |
| | repair_replan_success_rate | 修复/重规划成功率 |
| **工具安全** | dangerous_tool_block_rate | 危险工具拦截率 |
| | mutation_after_denial_rate | 拒绝后仍尝试写入率 |
| | benign_tool_pass_rate | 安全工具放行率 |

### Benchmark 规模

- **上下文治理**：15 个 benchmark 用例
- **记忆**：15 个 benchmark 用例
- **任务规划**：16 个 benchmark 用例
- **工具安全**：17 个 benchmark 用例
- **总计**：63 个 JSON benchmark 文件

---

## 许可证

MIT License — 详见 [LICENSE](LICENSE)
