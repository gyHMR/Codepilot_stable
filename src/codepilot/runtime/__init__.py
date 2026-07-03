# 新手导读：包门面文件：集中导出本层最常用的类型和入口，降低学习时的导入成本。
# 关注点：runtime 层负责把配置、模型、工具、扩展、会话和审批恢复装配成可运行服务。

"""Runtime execution base for assembled Codepilot agent sessions."""

from .assembly import assemble_runtime, create_agent_session, explain_runtime_config
from .service import RuntimeService

__all__ = [
    "assemble_runtime",
    "create_agent_session",
    "explain_runtime_config",
    "RuntimeService",
]
