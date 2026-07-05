# 新手导读：provider 子包不做聚合导出，也不在导入时注册 provider。
# 关注点：需要具体 provider 时导入具体模块；需要注册时显式调用 llm.registry。

"""Concrete LLM provider implementations.

This package root is intentionally not a provider facade. Import concrete
provider functions from their modules, and register built-ins explicitly from
``codepilot.llm.registry`` during runtime bootstrap.
"""

__all__: list[str] = []
