# Architecture Notes

本 fixture 的当前架构非常简单：

1. README.md 描述当前任务背景。
2. docs/architecture.md 描述当前有效架构。
3. docs/legacy-notes.md 是历史记录，不应覆盖当前说明。
4. src/sample.py 只是一个示例文件，不需要被修改。

关键约束：

- 当前请求永远比历史上下文优先。
- 如果上下文预算紧张，应优先保留当前请求、README.md 和本架构说明。
- legacy notes 可以作为背景，但不能被当作最新事实。

