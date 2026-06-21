# 轻量评估改造方案

## 目标

- 用少量指标解释 Context、Memory、Planning、Tool Security 的效果。
- 同时支持确定性机制测试和真实模型 on/off 消融。
- 每个模块保留 2 个 Benchmark，并允许以后继续添加 JSON 用例。
- 统一评估入口，删除重复脚本和重复 Benchmark。

## 指标

- Context：Key Context Hit Rate、Token Efficiency、Stale Context Rate。
- Memory：Memory Retrieval Hit Rate、Redundant Read Reduction Rate、Failed Attempt Recurrence Rate。
- Planning：Evidence Coverage Rate、False Completion Rate、Repair/Replan Success Rate。
- Security：Dangerous Tool Block Rate、Mutation After Denial Rate、Benign Tool Pass Rate。

## Benchmark

- `benchmarks/evaluation/context/`：关键上下文命中、过期上下文排除。
- `benchmarks/evaluation/memory/`：记忆召回、重复读取。
- `benchmarks/evaluation/planning/`：证据完成、失败后修复。
- `benchmarks/evaluation/security/`：危险工具阻止、正常工具放行。

用例继续使用 JSON，通过 `metrics` 声明指标，通过 `expected` 提供简单答案。

## 运行方式

```bash
python -m codepilot.evaluation check
python -m codepilot.evaluation run context
python -m codepilot.evaluation experiment memory --repeat 3
python -m codepilot.evaluation report <eval_dir>
```

- `check`：使用确定性模型流验证机制和指标。
- `run`：使用真实模型运行一个模块或全部模块。
- `experiment`：对 Context、Memory、Planning 做 on/off 对照。
- `report`：根据已有产物重新生成指标报告。

## 实现原则

- 复用现有 Run Artifact 和 Assertion，不重新建设事件系统。
- 指标集中放在 `evaluation/metrics.py`，不散落在 Executor 和脚本中。
- Experiment 只支持两个变体、简单重复和平均值对比。
- 分母为 0 的指标记为 `N/A`。
- 不计算综合总分，不做显著性检验。
