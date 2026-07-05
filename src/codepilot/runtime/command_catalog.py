from __future__ import annotations

from .views import CommandDescriptor


def builtin_commands() -> list[CommandDescriptor]:
    """Return built-in application commands."""

    return [
        CommandDescriptor(name="help", description="显示可用命令", source="builtin"),
        CommandDescriptor(name="status", description="查看模型、工作区、会话和权限状态", source="builtin"),
        CommandDescriptor(name="mode", description="查看或切换任务模式：read/edit/plan", source="builtin"),
        CommandDescriptor(name="session", description="查看当前会话与叶子节点", source="builtin"),
        CommandDescriptor(name="tree", description="查看当前会话树", source="builtin"),
        CommandDescriptor(name="path", description="查看指定节点路径", source="builtin"),
        CommandDescriptor(name="fork", description="从指定节点分叉新会话", source="builtin"),
        CommandDescriptor(name="new", description="等价于从当前叶子分叉新会话", source="builtin"),
        CommandDescriptor(name="switch", description="切换到指定叶子节点", source="builtin"),
        CommandDescriptor(name="clear", description="清空上下文，创建新会话", source="builtin"),
        CommandDescriptor(name="context", description="查看最近一次上下文投影治理报告", source="builtin"),
        CommandDescriptor(name="memory", description="查看、添加、提升或删除结构化记忆", source="builtin"),
        CommandDescriptor(name="rollback", description="预览或执行最近一次 run 的 Git 回退", source="builtin"),
        CommandDescriptor(name="tools", description="查看当前可用工具", source="builtin"),
        CommandDescriptor(name="model", description="查看当前模型信息", source="builtin"),
        CommandDescriptor(name="usage", description="查看 token 用量和费用", source="builtin"),
        CommandDescriptor(name="exit", description="退出 Codepilot", source="builtin"),
    ]


__all__ = ["builtin_commands"]
