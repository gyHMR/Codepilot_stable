# Codepilot 记忆模块设计与实现

## 1. 为什么需要记忆模块？

在 Coding Agent 的实际使用中，我们发现了一个核心问题：**Agent 无法记住之前学到了什么**。

想象这个场景：
- 第 1 轮：用户让 Agent 读取 `service.py`，Agent 理解了它的职责
- 第 2 轮：用户问"这个文件是干什么的？"，Agent 又要重新读取
- 第 3 轮：Agent 尝试编辑失败了（因为 `old_text` 不唯一）
- 第 4 轮：同样的错误再次发生，Agent 没有"记住"上次的教训

这导致了三个严重问题：
1. **重复探索**：每次都从零开始理解代码
2. **重复犯错**：同样的错误反复出现
3. **过期信息**：文件修改后，旧的结论仍然被使用

## 2. 旧方案的问题

最初的设计非常简单：把所有信息存到一个 `MEMORY.md` 文件中。

```python
# 旧方案：简单读写 MEMORY.md
class MemoryManager:
    def load(self) -> str:
        return Path(".codepilot/MEMORY.md").read_text()
    
    def save(self, content: str):
        Path(".codepilot/MEMORY.md").write_text(content)
```

这种方案存在 **5 个致命缺陷**：

### 2.1 信息混杂，无法分类

```markdown
# MEMORY.md 中的内容
service.py 是运行时门面
上次编辑 service.py 失败了，因为 old_text 不唯一
项目测试命令是 pytest -q
当前正在修复登录 bug
```

系统无法区分：
- 哪些是**当前任务状态**（会话结束就没用了）
- 哪些是**文件认知**（文件变了就过期了）
- 哪些是**失败经验**（应该长期保留）
- 哪些是**项目知识**（跨会话都有用）

### 2.2 没有来源和可信度

一条信息可能来自：
- 用户明确要求（最可信）
- 工具实际观察（比较可信）
- 测试验证（可信）
- 模型推测（不太可信）
- 过期的历史总结（可能有害）

但旧方案无法区分，导致**错误的推测和正确的事实被同等对待**。

### 2.3 没有新鲜度机制

这是最危险的问题：

```python
# 场景：
# 1. Agent 读取 service.py，记住 "service.py 包含 UserService 类"
# 2. 用户删除了 UserService 类
# 3. Agent 仍然"记得" UserService 存在 → 做出错误决策
```

**对 Coding Agent 来说，错误地记住旧代码比没有记忆更危险。**

### 2.4 没有自动沉淀入口

工具执行后产生的结构化证据（文件哈希、退出码、验证结果）无法自动进入记忆系统。

### 2.5 静态注入，无法动态更新

旧方案在启动时读取一次 MEMORY.md，之后不会更新。即使运行中产生了新记忆，当前会话也无法使用。

## 3. 新设计的核心思想

新设计遵循一个核心原则：**记忆是证据驱动的结构化信息，不是对话归档**。

### 3.1 记忆 ≠ 历史

```
Session History（历史）：发生了什么
Memory（记忆）：哪些信息未来仍值得使用
```

历史是完整的、按时间顺序的；记忆是筛选过的、按语义组织的。

### 3.2 三级信任体系

每条记忆都有明确的来源和可信度：

```python
MemoryTrust = Literal[
    "observed",      # 工具实际观察到的（如文件内容）
    "verified",      # 经过验证的（如测试通过）
    "user_given",    # 用户明确给出的
    "model_claim",   # 模型自己推测的（最低可信度）
]
```

这确保了**事实优先于推测**。

### 3.3 自动新鲜度校验

每条文件记忆都绑定文件哈希：

```python
@dataclass
class MemoryRecord:
    source_hashes: dict[str, str]  # {"service.py": "abc123..."}
```

文件变化后，记忆自动失效：

```python
# 文件哈希不匹配 → 记忆标记为 stale
if current_hash != record.source_hashes[path]:
    record.status = "stale"  # 不会注入到上下文中
```

## 4. 架构设计

### 4.1 模块划分

```
src/codepilot/sessions/memory/
├── records.py      # 数据结构定义
├── store.py        # 持久化存储
├── writer.py       # 记忆写入（核心逻辑）
├── retriever.py    # 记忆检索（评分排序）
├── rendering.py    # 渲染为文本
└── files.py        # 文件工具和脱敏
```

### 4.2 五种记忆类型

```python
MemoryKind = Literal[
    "task",       # 当前任务状态
    "file",       # 文件认知
    "failure",    # 失败经验
    "decision",   # 技术决策
    "project",    # 项目知识
]
```

每种类型服务不同目的：

| 类型 | 生命周期 | 示例 |
|------|----------|------|
| task | 当前会话 | "正在修复登录 bug" |
| file | 文件变化前 | "service.py 是运行时门面" |
| failure | 长期保留 | "编辑时 old_text 必须唯一" |
| decision | 长期保留 | "选择方案 A 因为性能更好" |
| project | 永久 | "测试命令是 pytest -q" |

### 4.3 两种作用域

```python
MemoryScope = Literal["session", "project"]
```

- **Session Memory**：当前任务相关的临时信息
- **Project Memory**：跨会话的稳定项目知识

```python
# Session Memory：JSON 快照文件
.codepilot/sessions/<session_id>/memory.json

# Project Memory：JSONL 追加文件
.codepilot/memory/project.jsonl
```

## 5. 核心实现细节

### 5.1 统一记录格式

所有记忆使用同一个 `MemoryRecord` 结构：

```python
@dataclass
class MemoryRecord:
    id: str                              # 唯一标识
    kind: MemoryKind                     # 记忆类型
    scope: MemoryScope                   # 作用域
    content: dict[str, Any]              # 记忆内容
    source: str                          # 来源（如 "user_prompt", "tool:read"）
    source_run_id: str | None            # 关联的运行 ID
    related_paths: list[str]             # 关联的文件路径
    source_hashes: dict[str, str]        # 文件哈希快照
    trust: MemoryTrust                   # 信任度
    status: MemoryStatus                 # 状态（active/stale/deleted）
    created_at: str                      # 创建时间
    updated_at: str                      # 更新时间
```

**设计亮点**：统一格式避免了为每种记忆设计单独的存储协议。

### 5.2 MemoryWriter：保守写入策略

`MemoryWriter` 是记忆系统的核心，它决定了什么值得记住。

#### 5.2.1 任务记忆（TaskMemory）

```python
def remember_task(self, text: str, *, run_id: str | None = None) -> MemoryRecord:
    safe_text = sanitize_memory_text(text, limit=1200)  # 脱敏
    existing = self._active_task()
    
    if existing is None:
        # 创建新任务记忆
        existing = MemoryRecord(
            kind="task",
            scope="session",
            content={"goal": safe_text, "constraints": [], ...},
            trust="user_given",
        )
    else:
        # 更新现有任务
        if existing.content.get("goal") != safe_text:
            existing.content = {"goal": safe_text, ...}  # 新任务替代旧任务
        else:
            existing.content["goal"] = safe_text
    
    return self.store.update(existing)
```

**关键点**：
- 每个会话只有一个活跃任务
- 新任务会重置所有内容（而不是追加）
- 文本会经过脱敏处理

#### 5.2.2 文件记忆（FileKnowledge）

```python
def _remember_file(self, message: ToolResultMessage, state: dict) -> MemoryRecord:
    path = state.get("path")
    source_hash = state.get("sha256")
    
    # 查找已有的文件记忆
    for existing in self.store.load_session():
        if existing.kind == "file" and path in existing.related_paths:
            if existing.source_hashes.get(path) == source_hash:
                # 哈希相同：更新摘要和访问计数
                existing.content["access_count"] += 1
                return self.store.update(existing)
            else:
                # 哈希不同：旧记忆标记为 stale
                existing.status = "stale"
                self.store.update(existing)
    
    # 创建新记录
    return self.store.update(MemoryRecord(
        kind="file",
        content={"path": path, "summary": summary, "access_count": 1},
        source_hashes={path: source_hash},  # 绑定文件哈希
        trust="observed",
    ))
```

**关键点**：
- 文件哈希是新鲜度的唯一判断依据
- 哈希变化时，旧记忆自动失效
- 访问计数用于后续评分

#### 5.2.3 失败经验（FailureLesson）

```python
_FAILURE_CODES = {
    "unexpected_match_count",  # old_text 不唯一
    "multiple_matches",
    "path_not_found",
    "permission_denied",
    ...
}

def _remember_failure(self, message: ToolResultMessage) -> MemoryRecord:
    # 只记录结构化错误，不记录网络错误等
    if message.error_code not in _FAILURE_CODES:
        return
    
    # 查找已有的失败记录
    for existing in self.store.load_session():
        if (existing.kind == "failure" 
            and existing.content["action"] == message.tool_name
            and existing.content["failure_signature"] == message.error_code):
            # 重复失败：增加计数
            existing.content["occurrence_count"] += 1
            return self.store.update(existing)
    
    # 创建新记录
    return self.store.update(MemoryRecord(
        kind="failure",
        content={
            "action": message.tool_name,
            "failure_signature": message.error_code,
            "cause": sanitize_memory_text(error_text),
            "resolution": None,  # 等待后续成功时填入
        },
        trust="observed",
    ))
```

**关键点**：
- 只记录可识别的结构化错误
- 重复失败增加计数（用于后续评分）
- `resolution` 在后续成功时自动填入

#### 5.2.4 自动失效机制

```python
def invalidate_paths(self, paths: list[str]) -> list[MemoryRecord]:
    """当工具修改了工作区文件时，使相关文件记忆失效"""
    for record in self.store.load_session():
        if (record.kind == "file" 
            and record.status == "active"
            and set(paths).intersection(record.related_paths)):
            record.status = "stale"
            self.store.update(record)
    return changed
```

**触发时机**：
- `Edit` 工具成功后
- `Write` 工具成功后
- 任何修改了工作区的工具

### 5.3 MemoryRetriever：智能检索

检索器的任务是：从所有记忆中选出与当前任务最相关的几条。

#### 5.3.1 评分规则

```python
def retrieve(self, query: MemoryQuery) -> list[RetrievedMemory]:
    for record in records:
        score = 0
        reasons = []
        
        # 任务记忆最重要
        if record.kind == "task":
            score += 100
            reasons.append("task_memory")
        
        # 路径匹配加分
        if active_paths.intersection(record.related_paths):
            score += 40
            reasons.append(f"related_path:{path}")
        
        # 关键词匹配加分
        keyword_matches = query_terms.intersection(record_terms)
        score += min(30, len(keyword_matches) * 10)
        
        # 信任度加分
        if record.trust in {"verified", "observed"}:
            score += 20
        
        # 模型推测降分
        if record.trust == "model_claim":
            score -= 20
    
    # 按分数排序，应用类型限制
    return sorted(ranked, key=lambda x: x.score, reverse=True)
```

#### 5.3.2 噪音过滤

```python
# 跳过只出现 1 次且无 resolution 的 failure 记忆
if (record.kind == "failure" 
    and record.content.get("occurrence_count", 1) < 2
    and not record.content.get("resolution")):
    continue  # 太可能是噪音，跳过
```

#### 5.3.3 类型配额限制

```python
limits = {
    "task": 1,      # 最多 1 条任务记忆
    "file": 3,      # 最多 3 条文件记忆
    "failure": 2,   # 最多 2 条失败经验
    "decision": 2,  # 最多 2 条决策
    "project": 3,   # 最多 3 条项目知识
}
```

**设计亮点**：可解释的召回结果

```python
@dataclass(frozen=True)
class RetrievedMemory:
    record: MemoryRecord
    score: int
    reasons: list[str]  # ["related_path:service.py", "trust:verified"]
```

用户可以看到每条记忆为什么被选中。

### 5.4 敏感信息过滤

```python
_SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|token|password|secret|cookie)\s*[:=]\s*\S+"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{12,}\b"),
]

def sanitize_memory_text(text: str, *, limit: int) -> str:
    safe = text
    for pattern in _SECRET_PATTERNS:
        safe = pattern.sub("[REDACTED]", safe)
    return safe[:limit]
```

**关键点**：
- 所有文本在写入前都会脱敏
- API Key、Token、密码等会被替换为 `[REDACTED]`
- 文本长度有限制（避免存储过多内容）

## 6. 与上下文编译器的集成

记忆不是直接拼接到 System Prompt 中，而是通过 `ContextCompiler` 统一管理：

```python
class ContextCompiler:
    def compile(self, context: AgentContext, request: ContextPreparationRequest):
        # 1. 计算总预算
        total_budget = self.policy.input_budget(request)
        memory_budget = int(total_budget * 0.15)  # 记忆占 15%
        
        # 2. 检索相关记忆
        memory_items, retrieved_memories = self._memory_items(context)
        
        # 3. 按优先级选择，超预算则丢弃
        selected_memory, dropped_memory = _select_items(
            "memory", memory_items, memory_budget
        )
        
        # 4. 渲染到 System Prompt
        governance_text = _render_governance_context(
            selected_active, selected_evidence, selected_memory, stale_items
        )
```

**关键设计**：
- 记忆有固定的 token 预算（15%）
- 按评分排序，超预算的会被丢弃
- stale 的记忆会被跳过
- 生成详细的 ContextReport 用于调试

## 7. 存储设计

### 7.1 Session Memory：快照文件

```python
# .codepilot/sessions/<session_id>/memory.json
{
    "schema_version": 1,
    "session_id": "abc123",
    "records": [...]
}
```

**原子写入**：

```python
def _atomic_write_json(path: Path, payload: dict):
    fd, temp_name = tempfile.mkstemp(suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle)
        os.replace(temp_name, path)  # 原子替换
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
```

### 7.2 Project Memory：JSONL 追加

```python
# .codepilot/memory/project.jsonl
{"id": "mem_001", "kind": "project", "content": {...}}
{"id": "mem_002", "kind": "project", "content": {...}}
```

**优势**：
- 适合追加写入
- 单条损坏不影响整体
- 便于审计（可以看到记忆的产生时间）

**加载时按 ID 聚合**：

```python
def load_project(self) -> list[MemoryRecord]:
    latest = {}
    for line in self.project_file.read_text().splitlines():
        record = MemoryRecord.from_dict(json.loads(line))
        latest[record.id] = record  # 后来的覆盖先前的
    return list(latest.values())
```

## 8. 设计亮点总结

### 8.1 证据驱动，而非对话归档

```
传统方案：对话历史 → 存储 → 检索
新方案：工具结果 → 结构化提取 → 评分检索
```

记忆的来源是**工具执行的结构化证据**，而不是模型的自然语言总结。

### 8.2 自动新鲜度管理

```
文件哈希绑定 → 变化自动失效 → 不会使用过期信息
```

这是解决"过期信息污染"问题的关键。

### 8.3 三级信任体系

```
user_given > verified > observed > model_claim
```

确保事实优先于推测，用户指令优先于模型判断。

### 8.4 可解释的召回

```python
RetrievedMemory(
    record=record,
    score=70,
    reasons=["related_path:service.py", "trust:verified", "keyword:RuntimeService"]
)
```

用户可以理解每条记忆为什么被选中。

### 8.5 保守的写入策略

- 模型推测不能直接写入 Project Memory
- 只有结构化错误才记录为 FailureLesson
- 敏感信息自动过滤

### 8.6 与上下文治理的统一

记忆不是独立系统，而是通过 `ContextCompiler` 与活跃文件、证据、历史消息统一管理：

```
ContextCompiler
├── Repository Context（仓库快照）
├── Active Files（活跃文件）
├── Recent Evidence（近期证据）
├── Memory（记忆）← 这里
└── History（历史消息）
```

## 9. 实际效果

### 9.1 任务恢复

```python
# Session 恢复时，Agent 可以直接知道：
# - 当前任务目标
# - 已经完成的步骤
# - 遇到的阻塞
# - 下一步建议
```

### 9.2 避免重复犯错

```python
# 第一次失败：记录 FailureLesson
# 第二次遇到相同错误：从记忆中获取 resolution
# → 不再重复尝试已知会失败的方案
```

### 9.3 智能上下文管理

```python
# 文件变化 → 旧记忆自动失效
# 不会使用过期的文件认知
# 节省 token 预算给真正相关的信息
```

## 10. 学习建议

1. **从 records.py 开始**：理解数据结构是理解整个系统的基础
2. **重点阅读 writer.py**：这是记忆系统的核心逻辑
3. **理解评分规则**：retriever.py 的评分规则决定了什么信息会被使用
4. **看测试用例**：test_memory.py 展示了各种使用场景

## 11. 后续扩展方向

当前设计有意保持简单，后续可以扩展：

- 向量检索（替代关键词匹配）
- 自动知识图谱
- 跨项目记忆共享
- 复杂的置信度模型
- 记忆合并和压缩

但核心的**证据驱动 + 新鲜度管理 + 信任体系**设计是稳定的。
