# Codepilot Benchmarks

轻量评估任务统一放在 `evaluation/`：

```text
evaluation/
├── context/   # 关键上下文命中、过期上下文排除
├── memory/    # 记忆召回、重复读取
├── planning/  # 证据完成、失败后修复
└── security/  # 危险工具阻止、正常工具放行
```

当前每个模块提供两个示例。以后扩展时，只需在对应目录增加 JSON 和
fixture，不需要修改评估框架。

用例通过：

- `metrics` 声明要计算的指标；
- `expected` 提供关键文件、Memory 召回或工具类型等简单答案；
- `assertions` 检查任务是否正常完成以及文件结果是否正确。

常用命令：

```bash
python -m codepilot.evaluation check
python -m codepilot.evaluation run context
python -m codepilot.evaluation experiment memory --repeat 3
```
