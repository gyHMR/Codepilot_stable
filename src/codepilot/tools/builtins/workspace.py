from __future__ import annotations

"""
内置工具：workspace_status —— 获取当前工作区的 Git 状态快照。

该工具用于结构化地查看当前工作目录下的 Git 信息，包括：
  - 当前分支名
  - HEAD 的短哈希
  - 工作区是否为脏（存在未提交变更）
  - 变更文件路径列表及每个文件的 git 状态
  - diff 统计信息

所有信息以 JSON 格式返回，便于下游组件（如 LLM、UI 面板）消费。
"""

import json
import subprocess
from typing import Any, Callable

from codepilot.protocols import TextContent
from codepilot.tools.contracts import ToolCallRequest, ToolDefinition, ToolResult
from codepilot.tools.registry import get_builtin_tool_metadata
from codepilot.tools.sandbox import WorkspaceSandbox


# ---------------------------------------------------------------------------
# 工厂函数：create_workspace_tools()
#   - 根据 allow 白名单决定是否注册 workspace_status 工具
#   - 构造 ToolDefinition 对象，包含工具元数据和执行函数
#   - 返回一个工具定义列表（通常只有一个元素），供工具注册系统消费
# ---------------------------------------------------------------------------
def create_workspace_tools(
    sandbox: WorkspaceSandbox,
    *,
    allow: Callable[[str], bool],
) -> list[ToolDefinition]:
    """
    创建 workspace_status 工具的工厂函数。

    参数:
        sandbox (WorkspaceSandbox):
            工作区沙箱对象，提供工作区根路径（sandbox.root），即 Git 仓库的根目录。
        allow (Callable[[str], bool]):
            工具白名单回调。传入工具名称，返回该工具是否允许注册。
            若 `allow("workspace_status")` 返回 False，则返回空列表，表示不注册该工具。

    返回:
        list[ToolDefinition]: 包含 workspace_status 工具定义的列表（或空列表）。
    """

    # 检查 workspace_status 是否在白名单中，不在则跳过注册
    if not allow("workspace_status"):
        return []

    # -----------------------------------------------------------------------
    # 内部异步函数：workspace_status
    #   - 工具的实际执行逻辑
    #   - 通过调用 _git() 获取 git 分支、HEAD、状态和 diff 信息
    #   - 将结果封装为 JSON 载荷并返回 ToolResult
    # -----------------------------------------------------------------------
    async def workspace_status(
        request: ToolCallRequest,
        signal=None,
        on_update=None,
    ) -> ToolResult:
        """
        获取并返回当前工作区的 Git 状态快照。

        参数:
            request (ToolCallRequest): 工具调用请求对象（此处未使用其参数）。
            signal: 取消信号（本工具为同步操作，忽略）。
            on_update: 进度更新回调（本工具无中间进度，忽略）。

        返回:
            ToolResult: 包含以下信息的工具结果：
                - content: JSON 字符串，包含分支、HEAD、脏状态、变更路径、diff 统计
                - workspace_changed: 始终为 False（该工具为只读操作）
                - details: 原始载荷字典（供编程消费）
                - metadata: 包含变更路径列表、截断标记和变更文件总数
        """
        # 本工具忽略请求参数、取消信号和进度回调（为纯查询类工具）
        _ = request, signal, on_update

        # 获取 Git 仓库根目录
        root = sandbox.root

        # ---- 执行 Git 命令收集基本信息 ----
        # 当前分支名（如 "main"、"feature/xxx"）；非分支状态返回空字符串
        branch = _git(root, ["branch", "--show-current"])
        # 当前 HEAD 的短哈希（如 "a1b2c3d"）
        head = _git(root, ["rev-parse", "--short", "HEAD"])
        # 工作区文件变更状态（porcelain 格式，每行一个文件变更）
        status_text = _git(root, ["status", "--porcelain", "--", "."])

        # ---- 判断是否为有效的 Git 仓库 ----
        total_changed = 0
        if status_text is None:
            # 非 Git 仓库或 git 命令失败：返回空载荷，指示无仓库状态
            payload = {
                "is_git_repo": False,
                "branch": None,
                "head": None,
                "dirty": None,
                "changed_paths": [],
                "diff_stat": None,
            }
        else:
            # ---- 解析 porcelain 状态输出，构建变更路径列表 ----
            # porcelain 格式每行格式: XY filename
            #   XY 为两个字符的状态码，如 " M"（工作区修改）、"??"（未跟踪）
            #   文件名可能包含 " -> " 表示重命名（如 "R  old -> new"）
            changed_paths = []
            for line in status_text.splitlines():
                if len(line) < 4:
                    continue  # 跳过格式不正确的行
                changed_paths.append(
                    {
                        # 提取文件路径：取第 3 个字符之后的部分
                        # 若为重命名，取 " -> " 之后的新文件名（即目标文件）
                        "path": line[3:].split(" -> ")[-1],
                        # 将 git 状态码转换为人类可读的状态名称
                        "status": _status_name(line[:2]),
                        # 保留原始状态码供程序消费
                        "code": line[:2],
                    }
                )
            total_changed = len(changed_paths)

            # ---- 获取 diff 统计信息 ----
            # "git diff --stat" 返回形如 "file.py | 10 +++++-----" 的统计
            diff_stat = _git(root, ["diff", "--stat", "--", "."])

            # 构建完整载荷
            payload = {
                "is_git_repo": True,
                # branch/head 可能为空字符串（非分支状态），统一转为 None
                "branch": branch or None,
                "head": head or None,
                # dirty 标记：存在变更路径即为脏
                "dirty": bool(changed_paths),
                # 最多返回前 200 个变更路径，超出部分截断以避免载荷过大
                "changed_paths": changed_paths[:200],
                # diff 统计取最后 2000 个字符，超长部分截断；无统计信息时为 None
                "diff_stat": (diff_stat or "")[-2000:] or None,
            }

        # ---- 构造并返回工具结果 ----
        return ToolResult(
            # 内容：JSON 格式的载荷字符串，缩进 2 空格，不转义非 ASCII 字符
            content=[TextContent(text=json.dumps(payload, ensure_ascii=False, indent=2))],
            # 该工具为只读操作，不会修改工作区
            workspace_changed=False,
            # 原始载荷字典，供编程方式访问
            details=payload,
            # 附加元数据：变更路径、截断标记、变更文件数
            metadata={
                "changed_paths": payload["changed_paths"],
                # truncated: 当实际变更文件数超过返回的路径数时，标记为已截断
                "truncated": bool(payload["is_git_repo"] and total_changed > len(payload["changed_paths"])),
                "changed_path_count": total_changed,
            },
        )

    # ---- 获取内置工具元数据 ----
    # 元数据包含重试策略、超时、权限等配置，由 registry 统一管理
    metadata = get_builtin_tool_metadata("workspace_status")
    if metadata is None:
        raise ValueError("Missing builtin metadata for workspace_status")

    # ---- 构造 ToolDefinition 并返回 ----
    return [
        ToolDefinition(
            name="workspace_status",
            label="Workspace Status",
            description="结构化查看 Git 分支、HEAD、工作区变更路径和 diff 统计。",
            # 参数定义：该工具不接受任何参数（空对象 schema）
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            metadata=metadata,
            execute=workspace_status,
        )
    ]


# ---------------------------------------------------------------------------
# 辅助函数：_git(root, args)
#   - 在指定目录下执行 git 命令的安全封装
#   - 统一处理超时、异常和错误返回码
#   - 返回标准输出字符串或 None（表示执行失败）
# ---------------------------------------------------------------------------
def _git(root, args: list[str]) -> str | None:
    """
    在指定工作目录下安全执行 git 命令。

    参数:
        root: Git 仓库根目录路径，作为子进程的工作目录。
        args (list[str]): git 命令的参数列表（不含 "git" 本身），
                          例如 ["status", "--porcelain"]。

    返回:
        str | None:
            成功时返回命令的标准输出（已去除末尾换行符），
            失败时返回 None（包括超时、异常、非零退出码等情况）。

    安全特性:
        - 超时保护（3 秒），防止 git 命令挂起阻塞主流程
        - 异常捕获（OSError、SubprocessError），避免因环境问题导致崩溃
        - 编码容错（errors="replace"），处理非 UTF-8 文件名
        - 非零退出码不抛异常（check=False），由调用方自行处理
    """
    try:
        result = subprocess.run(
            ["git", *args],          # 将 "git" 与参数列表拼接为完整命令
            cwd=root,                # 设置工作目录为仓库根目录
            capture_output=True,     # 捕获标准输出和标准错误
            text=True,               # 以文本模式处理输出（非字节）
            encoding="utf-8",        # 使用 UTF-8 编码
            errors="replace",        # 遇到无法解码的字符时用替换字符代替，避免崩溃
            timeout=3,               # 超时 3 秒，防止 git 命令阻塞
            check=False,             # 非零退出码不抛出 CalledProcessError
        )
    except (OSError, subprocess.SubprocessError):
        # git 未安装、权限不足、或其他子进程异常时返回 None
        return None
    if result.returncode != 0:
        # git 命令执行失败（如不在 git 仓库中），返回 None
        return None
    # 成功：去除末尾的 \r\n 或 \n 后返回
    return result.stdout.rstrip("\r\n")


# ---------------------------------------------------------------------------
# 辅助函数：_status_name(code)
#   - 将 git status porcelain 格式的双字符状态码映射为人类可读的状态名称
#   - 状态码优先级：untracked > deleted > added > renamed > modified
#     （通过 if-elif 链按判定优先级处理，例如 "?M" 会被识别为 untracked）
# ---------------------------------------------------------------------------
def _status_name(code: str) -> str:
    """
    将 git porcelain 状态码转换为人类可读的状态名称。

    porcelain 格式的状态码是一个两字符字符串：
      - 第一个字符：暂存区（index）状态
      - 第二个字符：工作区（working tree）状态
      例如 " M" 表示暂存区无变更、工作区已修改；"??" 表示未跟踪文件。

    参数:
        code (str): 两字符的 git porcelain 状态码。

    返回:
        str: 人类可读的状态名称，可能的值：
            - "untracked" : 未跟踪的文件（状态码包含 "?"）
            - "deleted"    : 已删除的文件（状态码包含 "D"）
            - "added"      : 新添加的文件（状态码包含 "A"）
            - "renamed"    : 重命名的文件（状态码包含 "R"）
            - "modified"   : 已修改的文件（以上条件均不满足时的默认值）
    """
    # 按优先级从特殊到一般进行判断
    # "?" 表示未跟踪（untracked），可能出现在任一位置
    if "?" in code:
        return "untracked"
    # "D" 表示删除（deleted），可能出现在暂存区或工作区
    if "D" in code:
        return "deleted"
    # "A" 表示新增（added），通常出现在暂存区
    if "A" in code:
        return "added"
    # "R" 表示重命名（renamed），通常出现在暂存区
    if "R" in code:
        return "renamed"
    # 默认：修改（modified），包括 "M"、" M"、"MM" 等情况
    return "modified"


__all__ = ["create_workspace_tools"]
