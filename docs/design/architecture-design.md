# Codepilot 架构设计文档

> **版本**: v0.3.0 | **更新日期**: 2026-06-21 | **状态**: 生效中

## 1. 项目概览

Codepilot 是一个本地 AI 编程助手，专注于 CLI 工作流（未来将支持 Web 控制台）。采用 Python (>=3.10) 构建，基于 `asyncio` 异步优先设计。

**核心依赖**:
- `httpx` - HTTP 客户端（调用 LLM API）
- `rich` - 终端渲染（消息、工具事件）
- `prompt_toolkit` - 交互式 CLI 输入

**入口点**: `codepilot = "codepilot.interfaces.cli.main:main"` (定义于 `pyproject.toml`)

---

## 2. 分层架构

Codepilot 采用严格的分层架构，每一层职责清晰，依赖关系单向流动。

```
┌─────────────────────────────────────────────────────────────┐
│                    Layer 7: interfaces/                      │
│                      (用户接口层)                            │
├─────────────────────────────────────────────────────────────┤
│                    Layer 6: extensions/                      │
│                      (扩展系统层)                            │
├─────────────────────────────────────────────────────────────┤
│                    Layer 5: runtime/                         │
│                      (运行时组装层)                          │
├─────────────────────────────────────────────────────────────┤
│                    Layer 4: sessions/                        │
│                      (会话管理层)                            │
├─────────────────────────────────────────────────────────────┤
│                    Layer 1: core/                            │
│                      (Agent 循环引擎)                        │
├──────────────────────┬──────────────────────────────────────┤
│   Layer 2: llm/      │      Layer 3: tools/                 │
│   (LLM 提供者层)     │      (工具系统层)                     │
├──────────────────────┴──────────────────────────────────────┤
│                    Layer 0: protocols/                       │
│                      (共享类型定义层)                        │
└─────────────────────────────────────────────────────────────┘
```

**依赖规则** (架构强制执行):

| 层级 | 包名 | 依赖项 |
|------|------|--------|
| Layer 0 | `protocols/` | 无内部依赖 (纯类型定义) |
| Layer 1 | `core/` | 仅依赖 `protocols/` |
| Layer 2 | `llm/` | 依赖 `protocols/` |
| Layer 3 | `tools/` | 依赖 `protocols/` |
| Layer 4 | `sessions/` | 依赖 `protocols/`, `core/`, `observability/` |
| Layer 5 | `runtime/` | 依赖 `protocols/`, `core/`, `llm/`, `tools/`, `sessions/`, `extensions/`, `observability/` |
| Layer 6 | `extensions/` | 依赖 `protocols/`, `core/`, `tools/` |
| Layer 7 | `interfaces/` | 依赖 `runtime/` (传递性依赖所有层) |

**横切关注点**:
- `evaluation/` - 评估框架，依赖 `runtime/`, `observability/`
- `observability/` - 可观测性，依赖 `protocols/`

---

## 3. 各层详细设计

### 3.1 Layer 0: `protocols/` — 共享类型定义层

**职责**: 定义整个系统共享的数据类型，零内部依赖，是整个架构的基石。

**核心文件与类型**:

| 文件 | 核心类型 | 说明 |
|------|----------|------|
| `content.py` | `TextContent`, `ImageContent` | 内容块定义 |
| `messages.py` | `UserMessage`, `AssistantMessage`, `ToolResultMessage` | 消息类型 |
| `tools.py` | `Tool`, `ToolCall`, `ToolResult`, `ToolMetadata` | 工具相关类型 |
| `llm.py` | `Model`, `ModelCapabilities`, `Context`, `StreamOptions` | LLM 相关类型 |
| `events.py` | `AgentEventBase`, `AgentEvent` | 事件类型 |
| `runs.py` | `AgentRunResult`, `AgentRunStopReason`, `TaskSummary` | 运行结果类型 |
| `context.py` | `Context`, `ContextPreparationRequest`, `ContextPreparationResult` | 上下文类型 |

**设计原则**:
- 纯数据定义，无业务逻辑
- 使用 `dataclass` 或 `TypedDict` 定义
- 所有字段有明确的类型注解
- 不依赖任何内部模块

---

### 3.2 Layer 1: `core/` — Agent 循环引擎

**职责**: 实现核心 Agent 执行循环：用户提示 → LLM 推理 → 工具执行 → 循环直到完成。

**核心组件**:

| 文件 | 类/函数 | 职责 |
|------|---------|------|
| `agent_loop.py` | `run_agent_loop()`, `_run_loop()` | 主循环：LLM 流式响应 → 提取工具调用 → 执行工具 → 更新任务状态 → 重复 |
| `agent.py` | `Agent` | 封装 Agent 循环，管理状态、事件订阅、引导消息、上下文准备 |
| `llm_runner.py` | `LLMStreamRunner` | 桥接 Agent 循环与 LLM 提供者，处理上下文准备、能力验证、流式响应 |
| `tool_coordinator.py` | `ToolCallCoordinator` | 执行工具调用批次（顺序或并行） |
| `task_controller.py` | `TaskController` | 任务规划、步骤跟踪、完成检查、重规划决策 |
| `events.py` | `AgentEventEmitter` | 封装事件发射，带 runId/turnId/eventId 和序列号 |
| `run_state.py` | `RunState` | 跟踪运行计数器、重复调用检测、结果组装 |
| `message_conversion.py` | `convert_to_llm()` | 内部消息格式转换为 LLM 提供者兼容格式 |

**Agent 循环流程**:
```
┌─────────────────────────────────────────────────────────────┐
│                     Agent Loop                              │
├─────────────────────────────────────────────────────────────┤
│  1. LLMStreamRunner.stream_assistant_response()             │
│     ↓                                                        │
│  2. 提取 ToolCall 对象                                       │
│     ↓                                                        │
│  3. ToolCallCoordinator.execute_batch()                      │
│     ↓                                                        │
│  4. TaskController 更新任务状态，检查完成                     │
│     ↓                                                        │
│  5. 重复直到：无工具调用 | 任务完成 | 最大迭代 | 需要审批 | 错误 │
└─────────────────────────────────────────────────────────────┘
```

---

### 3.3 Layer 2: `llm/` — LLM 提供者层

**职责**: 提供 LLM 提供者抽象，支持多模型切换。

**核心组件**:

| 文件 | 功能 |
|------|------|
| `api_registry.py` | 全局提供者注册表，通过 `model.api` 字符串分发 |
| `models.py` | 内置模型目录（Anthropic, OpenAI, DeepSeek） |
| `providers/anthropic.py` | Anthropic Messages API 适配器 |
| `providers/openai_compatible.py` | OpenAI 兼容 Chat Completions 适配器 |
| `providers/_common.py` | 共享工具函数：消息转换、工具转换、JSON 解析 |
| `providers/register_builtins.py` | 内置提供者注册（模块导入时自动执行） |
| `event_stream.py` | `AssistantMessageEventStream` 流式响应 |
| `overflow.py` | 上下文溢出检测 |
| `env_api_keys.py` | 环境变量 API Key 解析 |

**提供者注册模式**:
```python
# 注册提供者
register_api_provider("anthropic-messages", stream_anthropic, stream_simple_anthropic)

# 使用提供者
stream = await stream(model, messages, tools, options)
# 根据 model.api 自动分发到正确的提供者
```

**支持的 API 类型**:
- `anthropic-messages` — Anthropic Messages API
- `openai-compatible` — OpenAI Chat Completions API（兼容 DeepSeek 等）

---

### 3.4 Layer 3: `tools/` — 工具系统层

**职责**: 管理工具注册、执行、权限和沙箱。

**核心组件**:

| 文件 | 类/函数 | 职责 |
|------|---------|------|
| `types.py` | `AgentTool` | 可执行工具，包含 `execute` 函数 |
| `registry.py` | `ToolRegistry` | 工具存储和元数据，`infer_tool_metadata()` 自动推断元数据 |
| `runtime.py` | `ToolRuntime` | 执行引擎：权限检查 → 审批 → 执行 → 结果 |
| `permissions.py` | `PermissionPolicy`, `ToolDecision` | 权限模式：read-only, workspace-write, ask |
| `approval.py` | `ApprovalProvider` | 审批提供者协议 |
| `sandbox.py` | `WorkspaceSandbox` | 路径隔离，防止文件操作越界 |
| `shell_policy.py` | `ShellExecutionPolicy` | Shell 命令超时、输出限制、环境变量过滤 |

**内置工具** (8个):

| 工具 | 功能 |
|------|------|
| `ls` | 列出目录内容 |
| `read` | 读取文件 |
| `write` | 写入文件 |
| `edit` | 编辑文件 |
| `grep` | 搜索文件内容 |
| `find` | 查找文件 |
| `bash` | 执行 Shell 命令 |
| `workspace_status` | 获取工作区状态 (git status) |

**工具执行流程**:
```
ToolCall → ToolRuntime.execute()
  ↓
1. 权限检查 (PermissionPolicy)
  ↓
2. 审批检查 (ApprovalProvider)
  ↓
3. 执行工具 (AgentTool.execute)
  ↓
4. 返回结果 (ToolResult)
```

---

### 3.5 Layer 4: `sessions/` — 会话管理层

**职责**: 管理会话状态、记忆、上下文治理和持久化。

**子包结构**:

#### 3.5.1 `session.py` — 会话主类

`AgentSession` 是会话管理的核心，职责包括：
- 生命周期钩子管理
- 记忆系统 (remember/finalize)
- 上下文新鲜度检查
- 上下文压缩
- 会话分支/分叉
- 委托 `Agent` 执行

#### 3.5.2 `context/` — 上下文治理

| 文件 | 功能 |
|------|------|
| `compiler.py` | `ContextCompiler` — 编译上下文，注入记忆检索结果 |
| `compaction.py` | 上下文压缩（当上下文溢出时进行摘要） |
| `state.py` | `SessionContextState` — 跟踪会话上下文状态 |
| `repository_context.py` | 仓库文件跟踪 |

#### 3.5.3 `memory/` — 结构化记忆系统

| 文件 | 类 | 功能 |
|------|-----|------|
| `store.py` | `MemoryStore` | 内存和文件支持的记忆存储 |
| `writer.py` | `MemoryWriter` | 写入任务记录、工具观察、运行终结 |
| `retriever.py` | `MemoryRetriever` | 检索相关记忆用于上下文注入 |
| `records.py` | `MemoryRecord`, `RetrievedMemory` | 记忆类型定义 |
| `files.py` | — | 文件记忆持久化 |
| `rendering.py` | — | 记忆渲染用于上下文注入 |

#### 3.5.4 `history/` — 会话分支

| 文件 | 功能 |
|------|------|
| `branching.py` | `branch_fork_session()`, `branch_switch_to_entry()`, `branch_switch_session()` |
| `checkpoint.py` | `record_checkpoint()` |

#### 3.5.5 `persistence/` — 持久化层

| 文件 | 类 | 功能 |
|------|-----|------|
| `store.py` | `SessionStore` | 基于文件的会话持久化 (events.jsonl, context.jsonl, state.json) |
| `run_store.py` | `RunStore` | 运行结果持久化和上下文新鲜度评估 |
| `serde.py` | — | 序列化/反序列化辅助函数 |

---

### 3.6 Layer 5: `runtime/` — 运行时组装层

**职责**: 组装层，将所有组件连接在一起。

**核心组件**:

| 文件 | 功能 |
|------|------|
| `assembly.py` | 9 步组装流水线，从选项到完整 `AgentSession` |
| `service.py` | `RuntimeService` — 会话生命周期管理 |
| `config.py` | 多源配置解析 (CLI > session > workspace > default) |
| `model_resolver.py` | 模型解析（从选项/工作区/内置目录） |
| `tool_assembler.py` | 工具组装（builtin < caller < extension < MCP） |
| `prompt.py` | 系统提示词组合 |
| `context.py` | 运行时上下文构建 |
| `hook_pipeline.py` | 钩子流水线组合 |
| `resources.py` | 工作区资源加载 |

**9 步组装流水线**:
```
1. load_runtime_inputs()           — 加载工作区资源和会话元数据
2. resolve_model()                 — 从选项/工作区/环境解析模型
3. resolve_runtime_config()        — 多源配置合并
4. assemble_tools()                — 组装工具 (builtin + caller + extension + MCP)
5. build_runtime_context()         — 构建运行时上下文
6. build_runtime_system_prompt()   — 组合系统提示词
7. compose_*()                     — 组合钩子流水线
8. Build AgentSessionOptions       — 构建会话选项
9. Construct RuntimeAssembly       — 构建运行时组装产物
```

---

### 3.7 Layer 6: `extensions/` — 扩展系统层

**职责**: 支持三种扩展机制：Python 扩展、Markdown 技能、MCP 工具。

**核心组件**:

| 文件 | 功能 |
|------|------|
| `api.py` | `ExtensionAPI` — 传递给扩展 `register(api)` 函数的 API 对象 |
| `loader.py` | `load_extensions()` — 发现并加载 `.codepilot/extensions/` 中的 `.py` 文件 |
| `skills.py` | `load_skills()` — 发现并加载 `.codepilot/skills/` 中的 `.md` 文件 |
| `mcp/bridge.py` | `MCPClient`, `create_mcp_proxy_tools()` — 创建 MCP 代理工具 |
| `types.py` | `LoadedExtensions` — 标准化的能力集合 |

**三种扩展机制**:

1. **Python 扩展** (`.codepilot/extensions/*.py`):
   - 定义 `register(api: ExtensionAPI)` 函数
   - 通过 `api` 注册工具、钩子、命令、提示词指南

2. **Markdown 技能** (`.codepilot/skills/*.md`):
   - 使用 frontmatter 定义元数据
   - 注册为命令和提示片段

3. **MCP 工具**:
   - 通过 `MCPClient` 协议连接 MCP 服务器
   - 创建代理工具转发调用

**标准化能力类型**（所有扩展源统一为 4 种）:
- `tools` — 工具
- `commands` — 命令
- `prompt_text` — 提示文本
- `hooks` — 钩子

---

### 3.8 Layer 7: `interfaces/` — 用户接口层

**职责**: 提供用户交互界面，目前支持 CLI 和 Web（骨架）。

#### 3.8.1 CLI (`interfaces/cli/`)

| 文件 | 功能 |
|------|------|
| `cli.py` | `main()`, `build_parser()` — 解析 CLI 参数，创建 `RuntimeService` |
| `runner.py` | `run()` — 分发到交互式 shell、单提示或 RPC 模式 |
| `shell.py` | 交互式 REPL shell（基于 prompt_toolkit） |
| `renderer.py` | `TerminalRenderer` — Rich 终端输出（消息、工具事件、错误） |
| `approval.py` | `CliApprovalProvider` — 交互式终端审批 |

**CLI 运行模式**:
- **交互式 shell**: 持续对话
- **单提示模式**: 执行单个提示后退出
- **RPC 模式**: 程序化调用

#### 3.8.2 Web (`interfaces/web/`)

| 文件 | 功能 |
|------|------|
| `api.py` | `WebConsoleBackend` — 框架无关的后端，委托给 `RuntimeService` |
| `app.py` | `create_web_app()` 工厂函数 |
| `websocket.py` | `WebSocketSessionStream` — 将 Agent 事件转换为 WebSocket 流 |
| `schemas.py` | Web 特定数据模式 |
| `event_adapter.py` | `agent_event_to_web()` 适配器 |

---

### 3.9 横切关注点

#### 3.9.1 `evaluation/` — 评估框架

| 文件 | 功能 |
|------|------|
| `__main__.py` | CLI 入口点，命令：check, run, experiment, report |
| `service.py` | `EvaluationService` — 运行基准测试套件和实验 |
| `executor.py` | 基准测试执行引擎 |
| `assertions.py` | 评估断言 |
| `metrics.py` | 评估指标计算 |
| `report.py` | 报告生成 |

#### 3.9.2 `observability/` — 可观测性

| 文件 | 功能 |
|------|------|
| `recorder.py` | `EventRecorder` — JSONL 事件写入器 |
| `events.py` | 事件归一化和聚合 |
| `metrics.py` | `RunMetrics`, `ModelCallRecord`, `ToolCallRecord` — 结构化指标提取 |
| `audit.py` | `AuditBundle` — 审计产物（敏感数据脱敏） |
| `summaries.py` | `build_run_report()` — 人类可读运行报告 |

---

## 4. 核心执行流程

当用户发送消息时的完整执行流程：

```
用户输入
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. CLI (cli.py:main)                                        │
│    解析参数 → 创建 RuntimeService → 调用 runner.run()        │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. RuntimeService (service.py)                              │
│    委托给 AgentSession.run(text)                             │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. AgentSession (session.py:run)                            │
│    ├─ 运行 before_prompt 生命周期钩子                        │
│    ├─ 写入任务到结构化记忆 (_remember_task)                   │
│    ├─ 检查上下文新鲜度 (外部文件变更)                         │
│    ├─ 检查/压缩上下文 (如果溢出)                             │
│    ├─ 调用 Agent.run(text)                                   │
│    ├─ 持久化运行结果                                         │
│    ├─ 终结记忆 (运行结果)                                    │
│    ├─ 压缩上下文 (如果需要)                                  │
│    └─ 运行 after_prompt 生命周期钩子                         │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Agent (agent.py:run)                                     │
│    准备上下文 (调用 prepare_context 注入记忆)                 │
│    → 调用 run_agent_loop()                                   │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. agent_loop (agent_loop.py:_run_loop)                     │
│    ┌─────────────────────────────────────────────────────┐  │
│    │  LLMStreamRunner.stream_assistant_response()        │  │
│    │    ↓                                                │  │
│    │  提取 ToolCall 对象                                 │  │
│    │    ↓                                                │  │
│    │  ToolCallCoordinator.execute_batch()                │  │
│    │    ↓                                                │  │
│    │  TaskController 更新任务状态，检查完成               │  │
│    │    ↓                                                │  │
│    │  重复直到：无工具调用 | 任务完成 | 最大迭代           │  │
│    └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. ToolRuntime (runtime.py:execute)                         │
│    权限检查 → 审批 → 执行 → 结果                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 关键设计模式

### 5.1 提供者注册模式 (Provider Registry)

**位置**: `llm/api_registry.py`

通过 `model.api` 字符串分发到正确的 LLM 提供者。新提供者只需注册即可使用。

```python
# 注册
register_api_provider("anthropic-messages", stream_anthropic, stream_simple_anthropic)

# 使用 — 根据 model.api 自动分发
stream = await stream(model, messages, tools, options)
```

### 5.2 组装流水线模式 (Assembly Pipeline)

**位置**: `runtime/assembly.py`

9 步确定性组装，从 `CreateAgentSessionOptions` 到完整的 `AgentSession`。每个配置值都可追溯到其来源。

### 5.3 钩子流水线模式 (Hook Pipeline)

**位置**: `runtime/hook_pipeline.py`

可组合的 before/after 钩子，用于工具调用和提示。扩展和调用者都可以注册钩子。

### 5.4 事件驱动架构 (Event-Driven Architecture)

所有状态变更都发射 `AgentEvent` 对象：
- 会话订阅以持久化事件
- CLI 订阅以渲染事件
- Web 接口通过 WebSocket 流式传输事件

### 5.5 标准化扩展模型 (Normalized Extension Model)

**位置**: `extensions/types.py:LoadedExtensions`

所有扩展源（Python, Markdown, MCP）都标准化为 4 种能力类型：
- `tools` — 工具
- `commands` — 命令
- `prompt_text` — 提示文本
- `hooks` — 钩子

### 5.6 结构化记忆系统 (Structured Memory System)

**位置**: `sessions/memory/`

结构化记忆，包含任务跟踪、工具观察和运行终结。记忆在每次 LLM 调用前被检索并注入上下文。

### 5.7 上下文治理 (Context Governance)

**位置**: `sessions/context/`

- 当消息数或 token 数超过限制时自动压缩上下文
- 外部修改文件的上下文新鲜度检测

---

## 6. 学习路径指南

### 6.1 推荐学习顺序

按照层级从底向上学习，理解每一层的职责后再进入下一层：

```
Layer 0: protocols/        ← 从这里开始，理解数据类型
    ↓
Layer 2: llm/              ← 理解 LLM 集成
    ↓
Layer 3: tools/            ← 理解工具系统
    ↓
Layer 1: core/             ← 理解 Agent 循环核心（最重要）
    ↓
Layer 4: sessions/         ← 理解会话和记忆管理
    ↓
Layer 5: runtime/          ← 理解运行时组装（串联一切）
    ↓
Layer 6: extensions/       ← 理解扩展机制
    ↓
Layer 7: interfaces/       ← 最后看用户接口
```

> **为什么先学 llm/ 和 tools/ 再学 core/?** 因为 `core/` 是 LLM 调用和工具执行的编排者，先理解它依赖的两个子系统，再看它如何串联它们，会更容易理解。

### 6.2 各层学习重点与推荐阅读文件

#### Layer 0: `protocols/` — 先读这些文件
1. `protocols/messages.py` — 理解消息类型
2. `protocols/tools.py` — 理解工具类型
3. `protocols/events.py` — 理解事件类型
4. `protocols/llm.py` — 理解 LLM 相关类型

#### Layer 2: `llm/` — 先读这些文件
1. `llm/api_registry.py` — 理解提供者注册模式
2. `llm/models.py` — 理解模型定义
3. `llm/providers/_common.py` — 理解消息/工具转换

#### Layer 3: `tools/` — 先读这些文件
1. `tools/types.py` — 理解 `AgentTool` 定义
2. `tools/registry.py` — 理解工具注册
3. `tools/runtime.py` — 理解工具执行流程
4. `tools/builtin/file_tools.py` — 看一个具体工具实现

#### Layer 1: `core/` — 先读这些文件
1. `core/agent_loop.py` — **最重要**，理解主循环
2. `core/agent.py` — 理解 Agent 封装
3. `core/llm_runner.py` — 理解 LLM 调用桥接
4. `core/tool_coordinator.py` — 理解工具执行协调

#### Layer 4: `sessions/` — 先读这些文件
1. `sessions/session.py` — 理解会话主类
2. `sessions/memory/store.py` — 理解记忆存储
3. `sessions/memory/writer.py` — 理解记忆写入
4. `sessions/context/compiler.py` — 理解上下文编译

#### Layer 5: `runtime/` — 先读这些文件
1. `runtime/assembly.py` — **最重要**，理解组装流水线
2. `runtime/service.py` — 理解会话生命周期管理
3. `runtime/config.py` — 理解配置解析
4. `runtime/tool_assembler.py` — 理解工具组装

#### Layer 6: `extensions/` — 先读这些文件
1. `extensions/api.py` — 理解扩展 API
2. `extensions/loader.py` — 理解扩展加载
3. `extensions/skills.py` — 理解技能系统

#### Layer 7: `interfaces/` — 先读这些文件
1. `interfaces/cli/cli.py` — 理解 CLI 入口
2. `interfaces/cli/runner.py` — 理解运行模式分发
3. `interfaces/cli/shell.py` — 理解交互式 shell

---

## 7. 核心调用链

### 7.1 完整调用链（用户发送消息）

```
cli.py:main()
  → cli.py:_run_from_args()
    → service.py:RuntimeService.create_session()
      → assembly.py:create_agent_session()  [9步组装]
    → service.py:RuntimeService.send_message()
      → session.py:AgentSession.run()
        → session.py:_run_hooks("before_prompt")
        → session.py:_remember_task()
        → session.py:_check_context_freshness()
        → session.py:_check_and_compact_context()
        → agent.py:Agent.run()
          → agent.py:Agent.prepare_context()
            → compiler.py:ContextCompiler.compile()
              → retriever.py:MemoryRetriever.retrieve()
          → agent_loop.py:run_agent_loop()
            → llm_runner.py:LLMStreamRunner.stream_assistant_response()
              → api_registry.py:stream()
                → providers/anthropic.py:stream_anthropic()  [或 openai_compatible]
            → agent_loop.py:_extract_tool_calls()
            → tool_coordinator.py:ToolCallCoordinator.execute_batch()
              → runtime.py:ToolRuntime.execute()
                → permissions.py:PermissionPolicy.check()
                → approval.py:ApprovalProvider.approve()
                → tool.py:AgentTool.execute()
            → task_controller.py:TaskController.update()
        → session.py:_persist_run_result()
        → session.py:_finalize_memory()
        → session.py:_check_and_compact_context()
        → session.py:_run_hooks("after_prompt")
```

### 7.2 工具注册调用链

```
assembly.py:assemble_tools()
  → tool_assembler.py:assemble_tools()
    → builtin/__init__.py:create_builtin_tools()  [内置工具]
    → caller tools  [调用者提供的工具]
    → extension tools  [扩展工具]
    → mcp tools  [MCP 工具]
    → tool_assembler.py:_merge_tools()  [合并，按优先级]
    → tool_assembler.py:_validate_tools()  [验证]
    → tool_assembler.py:_apply_permissions()  [应用权限]
```

### 7.3 扩展加载调用链

```
assembly.py:assemble_runtime()
  → loader.py:load_extensions()  [Python 扩展]
    → loader.py:_discover_extension_files()
    → loader.py:_execute_extension()
    → api.py:ExtensionAPI.snapshot()
  → skills.py:load_skills()  [Markdown 技能]
    → skills.py:_discover_skill_files()
    → skills.py:_parse_skill_frontmatter()
  → mcp/bridge.py:create_mcp_proxy_tools()  [MCP 工具]
```

---

## 8. 目录结构参考

```
src/codepilot/
├── __init__.py
├── protocols/                  # Layer 0: 共享类型定义
│   ├── content.py
│   ├── messages.py
│   ├── tools.py
│   ├── llm.py
│   ├── events.py
│   ├── runs.py
│   ├── errors.py
│   └── context.py
│
├── core/                       # Layer 1: Agent 循环引擎
│   ├── agent_loop.py           # 主循环
│   ├── agent.py                # Agent 封装
│   ├── llm_runner.py           # LLM 调用桥接
│   ├── tool_coordinator.py     # 工具执行协调
│   ├── task_controller.py      # 任务控制
│   ├── events.py               # 事件发射器
│   ├── run_state.py            # 运行状态
│   ├── types.py                # 核心类型
│   └── message_conversion.py   # 消息转换
│
├── llm/                        # Layer 2: LLM 提供者层
│   ├── api_registry.py         # 提供者注册表
│   ├── models.py               # 模型目录
│   ├── providers/
│   │   ├── anthropic.py        # Anthropic 适配器
│   │   ├── openai_compatible.py# OpenAI 兼容适配器
│   │   ├── _common.py          # 共享工具函数
│   │   └── register_builtins.py# 内置注册
│   ├── event_stream.py         # 事件流
│   ├── overflow.py             # 溢出检测
│   └── env_api_keys.py         # API Key 解析
│
├── tools/                      # Layer 3: 工具系统层
│   ├── types.py                # 工具类型
│   ├── registry.py             # 工具注册表
│   ├── runtime.py              # 工具运行时
│   ├── permissions.py          # 权限策略
│   ├── approval.py             # 审批提供者
│   ├── sandbox.py              # 沙箱
│   ├── shell_policy.py         # Shell 策略
│   └── builtin/
│       ├── __init__.py         # 内置工具工厂
│       ├── file_tools.py       # 文件工具
│       ├── search_tools.py     # 搜索工具
│       ├── shell_tools.py      # Shell 工具
│       └── workspace_tools.py  # 工作区工具
│
├── sessions/                   # Layer 4: 会话管理层
│   ├── session.py              # 会话主类
│   ├── context/
│   │   ├── compiler.py         # 上下文编译器
│   │   ├── compaction.py       # 上下文压缩
│   │   ├── state.py            # 上下文状态
│   │   ├── repository_context.py
│   │   └── repository_tracker.py
│   ├── memory/
│   │   ├── store.py            # 记忆存储
│   │   ├── writer.py           # 记忆写入
│   │   ├── retriever.py        # 记忆检索
│   │   ├── records.py          # 记忆记录类型
│   │   ├── files.py            # 文件持久化
│   │   └── rendering.py        # 记忆渲染
│   ├── history/
│   │   ├── branching.py        # 会话分支
│   │   └── checkpoint.py       # 检查点
│   └── persistence/
│       ├── store.py            # 会话存储
│       ├── run_store.py        # 运行存储
│       └── serde.py            # 序列化
│
├── runtime/                    # Layer 5: 运行时组装层
│   ├── assembly.py             # 组装流水线
│   ├── service.py              # 运行时服务
│   ├── config.py               # 配置解析
│   ├── model_resolver.py       # 模型解析
│   ├── tool_assembler.py       # 工具组装
│   ├── prompt.py               # 提示词构建
│   ├── context.py              # 上下文构建
│   ├── hook_pipeline.py        # 钩子流水线
│   ├── resources.py            # 资源加载
│   └── types.py                # 运行时类型
│
├── extensions/                 # Layer 6: 扩展系统层
│   ├── api.py                  # 扩展 API
│   ├── loader.py               # 扩展加载器
│   ├── skills.py               # 技能加载器
│   ├── types.py                # 扩展类型
│   └── mcp/
│       └── bridge.py           # MCP 桥接
│
├── interfaces/                 # Layer 7: 用户接口层
│   ├── cli/
│   │   ├── cli.py              # CLI 入口
│   │   ├── runner.py           # 运行器
│   │   ├── shell.py            # 交互式 shell
│   │   ├── renderer.py         # 终端渲染器
│   │   └── approval.py         # CLI 审批
│   └── web/
│       ├── api.py              # Web 后端
│       ├── app.py              # Web 应用工厂
│       ├── websocket.py        # WebSocket 流
│       ├── schemas.py          # Web 模式
│       └── event_adapter.py    # 事件适配器
│
├── evaluation/                 # 横切: 评估框架
│   ├── __main__.py             # CLI 入口
│   ├── service.py              # 评估服务
│   ├── executor.py             # 执行器
│   ├── assertions.py           # 断言
│   ├── metrics.py              # 指标
│   └── report.py               # 报告
│
└── observability/              # 横切: 可观测性
    ├── recorder.py             # 事件记录器
    ├── events.py               # 事件处理
    ├── metrics.py              # 指标
    ├── audit.py                # 审计
    └── summaries.py            # 摘要
```

---

## 9. 总结

Codepilot 的架构设计遵循以下核心原则：

1. **分层清晰** — 每层职责单一，依赖关系单向流动
2. **异步优先** — 基于 `asyncio` 的全异步设计
3. **可扩展** — 通过扩展系统支持 Python/Markdown/MCP 三种扩展机制
4. **事件驱动** — 所有状态变更通过事件传播，支持多种消费者
5. **配置驱动** — 多源配置合并，支持 CLI/会话/工作区/默认配置
6. **可观测** — 内置事件记录、指标收集、审计追踪

通过理解这些设计原则和分层结构，开发者可以快速定位代码位置，理解系统行为，并进行有效的功能扩展和问题排查。
