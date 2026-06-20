# Harness Regression

这里放置使用 Session 级 `stream_fn` 的确定性 Runtime 合同案例。

JSON 只描述任务和 Verifier；模型事件脚本由 Python runner 注入，避免修改全局
Provider Registry。CI 中的基础合同覆盖位于 `test/test_evaluation.py` 和现有
Agent Loop 测试。
