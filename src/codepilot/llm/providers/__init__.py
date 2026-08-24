"""具体的 LLM Provider 实现。

本包根不提供聚合导出，也不在导入时注册 provider。
需要具体 provider 时导入具体模块；需要注册时显式调用 llm.registry。
"""

__all__: list[str] = []