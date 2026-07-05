# 新手导读：包门面文件：集中导出本层最常用的类型和入口，降低学习时的导入成本。
# 关注点：runtime 层负责把配置、模型、工具、扩展、会话和审批恢复装配成可运行服务。

"""Runtime execution base for assembled Codepilot agent sessions."""

from .gateway import RuntimeGateway
from .opening import SessionOpenIntent

__all__ = [
    "RuntimeGateway",
    "SessionOpenIntent",
]
