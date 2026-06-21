# Codepilot

> 本地 AI 编程智能体 —— 适合 Agent 方向实习/求职的工程项目

![Python](https://img.shields.io/badge/Python-%3E%3D3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Version](https://img.shields.io/badge/Version-0.3.0-orange)

---

## 项目介绍

Codepilot 是一个基于 Python 构建的本地编程智能体（Coding Agent），参考了 Claude Code 的设计理念：理解代码仓库、编辑代码、运行验证，并保留结构化的运行结果。

**本项目可作为 Agent 开发岗位的实习/求职项目：**

- 项目覆盖 LLM Agent 的完整技术栈——工具调用、会话管理、上下文治理、记忆系统、任务规划、安全控制、评估框架
- 代码结构清晰、分层明确，每一层都可以独立阅读和修改，适合逐模块学习
- 内置完整的评估体系（4 模块、12 指标），可以在简历/面试中展示你对 Agent 质量度量的理解
- 通过扩展机制（Python 扩展 / Markdown 技能 / MCP 桥接）可以快速做出个人亮点

**适用人群：** 计算机/AI 相关专业学生、需要 Agent方向实习项目作品的求职者。

---

## 项目特色

### 上下文治理

> 设计文档：[docs/design/context-design.md](docs/design/context-design.md)

传统 Agent 直接将所有文件内容塞进 prompt，导致 token 浪费、上下文污染。Codepilot 采用两层架构解决这一问题：

- **Bootstrap 阶段**：首次启动时生成仓库快照（`RepositorySnapshot`），快速建立全局认知
- **Compile 阶段**：每次对话前执行 7 步编译流水线，按需组装上下文
- **预算分配**：将 token 预算按比例分配给仓库状态（10%）、活跃文件（10%）、近期证据（10%）、记忆（10%）、历史（50%）、当前任务（10%），确保关键信息优先
- **新鲜度追踪**：三层失效机制（文件哈希 → 时间戳 → 会话状态），避免使用过期上下文

### 结构化记忆

> 设计文档：[docs/design/memory-design.md](docs/design/memory-design.md)

抛弃传统的 `MEMORY.md` 纯文本记忆方式，采用证据驱动的结构化记忆：

- **三级信任体系**：每条记忆标注来源可信度——`observed`（工具返回） > `verified`（多次验证） > `user_given`（用户告知） > `model_claim`（模型推断）
- **五种记忆类型**：`task`（任务）、`file`（文件）、`failure`（失败）、`decision`（决策）、`project`（项目），覆盖 Agent 工作中的主要信息类型
- **新鲜度校验**：通过文件哈希检测记忆关联的文件是否已变更，过期记忆自动降级
- **保守写入策略**：`MemoryWriter` 只在有明确证据时才写入，避免幻觉污染记忆

### 任务规划

> 设计文档：[docs/design/task-design.md](docs/design/task-design.md)

将模型的语义理解与运行时的任务边界分离：

- **证据绑定**：每个任务步骤必须关联执行证据（工具输出、文件变更），无证据的步骤不能标记为完成
- **动态计划**：执行过程中可根据工具返回结果动态调整计划，而非死板执行预设步骤
- **CompletionGate**：完成门控机制，防止模型过早声明任务完成
- **重规划机制**：当某步骤失败时，自动触发重规划而非直接放弃
- **会话恢复**：支持从持久化的任务状态恢复中断的会话

### 工具安全

> 设计文档：[docs/design/tool-design.md](docs/design/tool-design.md)

解决传统 Agent 中模型可以自行授权工具调用的安全隐患：

- **模型不可自我授权**：权限判断在运行时层完成，模型只能"请求"，不能"决定"
- **Decide-then-Execute**：先决策再执行，决策阶段不产生副作用
- **Shell 命令分类**：自动将 Shell 命令分为 `verification`（只读验证）、`mutation`（写入变更）、`high_risk`（高风险）、`unknown` 四类，分别执行不同策略
- **工作区沙箱**：文件操作限制在工作区目录内，防止路径逃逸
- **权限审批**：`ask` 模式下，敏感操作需要用户交互确认

### 评估框架

> 设计文档：[docs/design/eval-design.md](docs/design/eval-design.md)

内置完整的 Agent 质量评估体系：

- **证据驱动**：所有断言都基于实际运行证据（工具调用记录、上下文快照、文件变更），而非简单的输入/输出匹配
- **4 模块 12 指标**：覆盖上下文治理、记忆、任务规划、工具安全四大维度
- **确定性检查 + 消融实验**：`check` 命令用于快速验证（无需调用模型），`experiment` 命令用于 on/off 消融对比
- **JSON Benchmark**：每个测试用例是独立的 JSON 文件，支持场景化多步测试

### 多模型支持

支持多种 LLM 提供商，通过统一的 Provider 抽象屏蔽 API 差异：

| 协议 | 支持的提供商 |
|------|-------------|
| `anthropic-messages` | Anthropic Claude |
| `openai-compatible` | OpenAI、DeepSeek、及任意兼容 API |

### 扩展机制

三种扩展方式，从简单到复杂逐步深入：

- **Markdown 技能**：用 `.md` 文件定义提示词模板，零代码扩展 Agent 能力
- **Python 扩展**：编写 Python 模块注册自定义工具和钩子
- **MCP 桥接**：通过 Model Context Protocol 接入外部工具服务

---

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
