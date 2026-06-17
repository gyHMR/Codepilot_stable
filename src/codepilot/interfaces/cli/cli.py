from __future__ import annotations

"""
coding_agent 命令行入口。

示例：
python -m coding_agent --mode print --prompt "你好"
python -m coding_agent --mode interactive --provider anthropic --model-id glm-4.7
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Sequence

from codepilot.runtime.service import RuntimeService
from codepilot.runtime.types import CreateAgentSessionOptions

from .runner import RunOptions, run


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    定义所有支持的 CLI 参数，包括运行模式、模型配置、会话管理、
    安全限制等选项。

    返回:
        配置好的 argparse.ArgumentParser 实例
    """
    parser = argparse.ArgumentParser(description="Codepilot coding-agent CLI")

    # ── 运行模式 ──────────────────────────────────────────────
    # print: 单次输出模式，打印结果后退出
    # interactive: 交互式对话模式（默认）
    # rpc: 远程过程调用模式，供外部系统调用
    parser.add_argument("--mode", choices=["print", "interactive", "rpc"], default="interactive")

    # ── 工作区与会话 ──────────────────────────────────────────
    parser.add_argument("--workspace", default=".", help="Workspace directory")
    parser.add_argument("--session-id", default=None, help="Existing session id to resume")
    parser.add_argument("--list-entries", action="store_true", help="Print session entry ids and exit")
    parser.add_argument("--show-tree", action="store_true", help="Print session tree as JSON and exit")
    parser.add_argument("--fork-entry", default=None, help="Fork from entry id and print new session id")
    parser.add_argument("--switch-entry", default=None, help="Switch current session leaf to entry id")

    # ── 模型配置 ──────────────────────────────────────────────
    parser.add_argument("--provider", default=None, help="Model provider, e.g. anthropic/openai/deepseek")
    parser.add_argument("--model-id", default=None, help="Model id")
    parser.add_argument("--system-prompt", default="", help="System prompt")
    parser.add_argument("--thinking-level", default="off", help="Thinking level: off/minimal/low/medium/high/xhigh")

    # ── 工具执行与上下文管理 ──────────────────────────────────
    # parallel: 并行执行工具调用（默认，更快）
    # sequential: 顺序执行工具调用（更安全）
    parser.add_argument("--tool-execution", choices=["parallel", "sequential"], default="parallel")
    parser.add_argument("--max-context-messages", type=int, default=None, help="Compaction message threshold")
    parser.add_argument("--max-context-tokens", type=int, default=None, help="Compaction token threshold (approx)")
    parser.add_argument("--retain-recent-messages", type=int, default=24, help="Keep recent messages when compacting")

    # ── 重试策略 ──────────────────────────────────────────────
    parser.add_argument("--no-retry", action="store_true", help="Disable automatic retry on transient errors")
    parser.add_argument("--max-retries", type=int, default=2, help="Maximum retry count")
    parser.add_argument("--retry-base-delay-ms", type=int, default=1200, help="Retry base delay in milliseconds")

    # ── 安全限制 ──────────────────────────────────────────────
    # 只读模式：禁用写入、编辑、bash 等修改性操作
    parser.add_argument("--read-only", action="store_true", help="Enable read-only mode (disable write/edit/bash)")
    # 允许执行危险 bash 命令（默认会阻止 rm -rf 等危险操作）
    parser.add_argument("--allow-dangerous-bash", action="store_true", help="Disable dangerous bash blocking")
    # bash 命令白名单正则（可多次指定，匹配的命令允许执行）
    parser.add_argument(
        "--bash-allow-pattern",
        action="append",
        default=None,
        help="Regex pattern to allow bash command (can be repeated)",
    )
    # bash 命令黑名单正则（可多次指定，匹配的命令阻止执行）
    parser.add_argument(
        "--bash-block-pattern",
        action="append",
        default=None,
        help="Regex pattern to block bash command (can be repeated)",
    )
    # 放松编辑工具的严格匹配要求（默认要求唯一匹配）
    parser.add_argument(
        "--relaxed-edit",
        action="store_true",
        help="Disable strict unique-match requirement for edit tool",
    )

    # ── 输出控制 ──────────────────────────────────────────────
    parser.add_argument("--prompt", default=None, help="Prompt text (required in print mode)")
    parser.add_argument("--no-tool-events", action="store_true", help="Hide tool events in output")

    # ── 工作区资源 ────────────────────────────────────────────
    # 禁用读取 .codepilot/ 目录下的 settings、prompt、tools 配置文件
    parser.add_argument(
        "--disable-workspace-resources",
        action="store_true",
        help="Disable reading .codepilot/{settings,prompt,tools}",
    )

    return parser


async def _run_from_args(args: argparse.Namespace) -> int:
    """根据解析后的命令行参数执行 Agent 会话。

    主要流程：
    1. 将 CLI 参数转换为 CreateAgentSessionOptions 配置对象
    2. 通过工厂函数创建 Agent 会话
    3. 根据参数执行会话管理操作（切换/分叉/列出/查看）或正常运行
    4. 确保会话在退出时正确关闭

    参数:
        args: argparse 解析后的命令行参数命名空间

    返回:
        退出码，0 表示成功
    """
    # 将命令行参数映射到会话配置选项
    options = CreateAgentSessionOptions(
        workspace_dir=Path(args.workspace),           # 工作区目录
        provider=args.provider,                        # 模型提供商
        model_id=args.model_id,                        # 模型 ID
        system_prompt=args.system_prompt,              # 系统提示词
        session_id=args.session_id,                    # 要恢复的会话 ID
        thinking_level=args.thinking_level,            # 思考级别
        tool_execution=args.tool_execution,            # 工具执行模式
        max_context_messages=args.max_context_messages,  # 上下文消息数阈值
        max_context_tokens=args.max_context_tokens,      # 上下文 token 数阈值
        retain_recent_messages=args.retain_recent_messages,  # 压缩时保留的最近消息数
        retry_enabled=not bool(args.no_retry),        # 是否启用自动重试
        max_retries=args.max_retries,                  # 最大重试次数
        retry_base_delay_ms=args.retry_base_delay_ms,  # 重试基础延迟（毫秒）
        read_only_mode=bool(args.read_only),           # 只读模式
        block_dangerous_bash=not bool(args.allow_dangerous_bash),  # 是否阻止危险 bash
        bash_allow_patterns=args.bash_allow_pattern,   # bash 白名单模式
        bash_block_patterns=args.bash_block_pattern,   # bash 黑名单模式
        edit_require_unique_match=not bool(args.relaxed_edit),  # 编辑是否要求唯一匹配
        load_workspace_resources=not bool(args.disable_workspace_resources),  # 是否加载工作区资源
    )

    runtime = RuntimeService()
    session = runtime.create_session(options).session

    try:
        # ── 会话管理操作 ──────────────────────────────────────
        # 这些操作执行后立即返回，不进入对话循环

        # 切换当前会话的叶子节点到指定 entry
        if args.switch_entry:
            session.switch_to_entry(str(args.switch_entry))
            print(json.dumps({"type": "switch_entry", "session_id": session.session_id, "entry_id": args.switch_entry}))
            return 0

        # 从指定 entry 分叉出新会话
        if args.fork_entry:
            forked = session.fork_from_entry(str(args.fork_entry))
            try:
                print(
                    json.dumps(
                        {
                            "type": "forked",
                            "from_session_id": session.session_id,
                            "from_entry_id": args.fork_entry,
                            "new_session_id": forked.session_id,
                        },
                        ensure_ascii=False,
                    )
                )
            finally:
                forked.close()
            return 0

        # 列出当前会话的所有 entry ID
        if args.list_entries:
            print(json.dumps({"session_id": session.session_id, "entry_ids": session.list_entry_ids()}, ensure_ascii=False))
            return 0

        # 以 JSON 格式显示会话树结构
        if args.show_tree:
            print(json.dumps({"session_id": session.session_id, "tree": session.get_session_tree()}, ensure_ascii=False))
            return 0

        # ── 正常运行 ──────────────────────────────────────────
        # 进入交互式/打印/RPC 模式的对话循环
        await run(
            RunOptions(
                mode=args.mode,
                session=session,
                prompt=args.prompt,
                show_tool_events=not bool(args.no_tool_events),
            )
        )
    finally:
        # 确保会话资源被正确释放
        runtime.close_session(session.session_id)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 主入口函数。

    解析命令行参数，启动异步事件循环执行 Agent 会话。
    捕获 ValueError 并通过 parser.error() 输出友好的错误信息。

    参数:
        argv: 命令行参数列表，为 None 时使用 sys.argv

    返回:
        进程退出码，0=成功，2=参数错误
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        # 运行异步主函数
        return asyncio.run(_run_from_args(args))
    except ValueError as exc:
        # 参数验证失败时输出错误信息并返回退出码 2
        parser.error(str(exc))
        return 2


# 脚本直接运行时的入口点
if __name__ == "__main__":
    raise SystemExit(main())
