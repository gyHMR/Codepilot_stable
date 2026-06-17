from __future__ import annotations

"""
运行模式入口。

当前支持：
- print: 单次问答，输出文本与工具事件
- interactive: 交互式 REPL
- rpc: 极简 JSON-RPC 模式，供外部程序调用
"""

from dataclasses import dataclass, field
from dataclasses import asdict, is_dataclass
import inspect
import json
import sys
from typing import Any, Callable

from codepilot.llm.types import AssistantMessage, TextContent
from codepilot.core import AgentEvent

from codepilot.extensions.types import ExtensionCommandContext
from codepilot.runtime.command_registry import format_commands_for_help, list_runtime_commands, resolve_registered_command
from codepilot.runtime.types import InputFn, OutputFn, RunMode
from codepilot.sessions.session import AgentSession


@dataclass
class RunOptions:
    """运行配置选项。

    封装了启动 Agent 运行循环所需的全部参数。
    """
    mode: RunMode                                    # 运行模式：print / interactive / rpc
    session: AgentSession                            # Agent 会话实例
    prompt: str | None = None                        # print 模式下的用户输入
    output: OutputFn = print                         # 输出函数，默认为 print
    input_fn: InputFn = input                        # 输入函数，默认为 input
    show_tool_events: bool = True                    # 是否显示工具执行事件
    exit_commands: tuple[str, ...] = field(default_factory=lambda: ("exit", "quit", ":q"))  # 退出命令列表


def _extract_assistant_text(message: AssistantMessage) -> str:
    """从 AssistantMessage 中提取纯文本内容。

    将消息中所有 TextContent 块拼接为一个字符串。
    """
    return "".join(block.text for block in message.content if isinstance(block, TextContent)).strip()


async def run_print(
    session: AgentSession,
    prompt: str,
    *,
    output: OutputFn = print,
    show_tool_events: bool = True,
) -> AssistantMessage | None:
    """单次问答模式。

    流程：
    1. 订阅会话事件（流式文本 delta、工具执行状态）
    2. 调用 session.prompt() 发送用户输入给 Agent
    3. 实时收集流式输出片段
    4. 输出最终结果和元信息（停止原因、错误信息）

    参数:
        session: Agent 会话实例
        prompt: 用户输入的文本
        output: 输出函数
        show_tool_events: 是否显示工具执行事件

    返回:
        最后一条 Assistant 消息，若无则返回 None
    """
    # 收集流式输出的文本片段
    deltas: list[str] = []

    def on_event(event: AgentEvent) -> None:
        """事件回调：处理工具执行事件和流式文本更新。"""
        t = event["type"]

        # 工具执行开始/结束事件
        if show_tool_events and t in {"tool_execution_start", "tool_execution_end"}:
            output(f"[tool-event] {t}: {event.get('toolName', '')}")
            return

        # 流式文本更新事件：收集 delta 片段
        if t == "message_update":
            assistant_event = event.get("assistantMessageEvent") or {}
            if assistant_event.get("type") == "text_delta":
                delta = str(assistant_event.get("delta", ""))
                deltas.append(delta)

    # 订阅事件，执行完毕后取消订阅
    unsubscribe = session.subscribe(on_event)
    try:
        await session.prompt(prompt)
    finally:
        unsubscribe()

    # 获取最后一条 Assistant 消息
    final_assistant = next((m for m in reversed(session.messages) if isinstance(m, AssistantMessage)), None)

    # 输出结果：优先用流式 delta，否则从最终消息中提取文本
    if deltas:
        output("".join(deltas).strip())
    elif final_assistant is not None:
        output(_extract_assistant_text(final_assistant) or "(empty)")

    # 输出元信息
    if final_assistant is not None:
        output(f"[assistant.stop_reason] {final_assistant.stop_reason}")
        output(f"[assistant.error_message] {final_assistant.error_message}")
    return final_assistant


async def run_interactive(
    session: AgentSession,
    *,
    input_fn: InputFn = input,
    output: OutputFn = print,
    show_tool_events: bool = True,
    exit_commands: tuple[str, ...] = ("exit", "quit", ":q"),
) -> None:
    """交互式 REPL 模式。

    持续读取用户输入并调用 Agent 处理，直到命中退出命令。
    支持以 "/" 开头的内置命令（如 /help, /tree, /fork 等）。

    流程：
    1. 打印欢迎信息和可用命令
    2. 循环读取用户输入
    3. "/" 开头 → 交给命令处理器
    4. 普通文本 → 调用 run_print 发送给 Agent
    """
    output("Entering interactive mode. Type 'exit' or '/exit' to quit.")
    output(format_commands_for_help(session))
    current_session = session

    while True:
        text = input_fn("you> ").strip()
        bare = text.lstrip("/")

        # 检查退出命令
        if bare in exit_commands:
            output("Bye.")
            return

        # 空输入跳过
        if not text:
            continue

        # "/" 开头的命令交给交互命令处理器
        if text.startswith("/"):
            handled, switched = await _handle_interactive_command(current_session, text, output=output)
            # 如果命令返回了新会话（如 /fork, /clear），切换到新会话
            if switched is not None:
                current_session.close()
                current_session = switched
            if handled:
                continue

        # 普通文本 → 发送给 Agent 处理
        await run_print(current_session, text, output=output, show_tool_events=show_tool_events)


def _create_fresh_session(old: AgentSession) -> AgentSession:
    """创建全新空白会话，保留模型/工具/设置但不带历史消息。

    用于 /clear 命令：清空上下文但保留配置。
    """
    from codepilot.runtime.types import AgentSessionOptions
    from codepilot.sessions.store import new_session_id

    return AgentSession(
        AgentSessionOptions(
            model=old.agent.state.model,                # 保留原模型
            workspace_dir=old.workspace_dir,            # 保留工作区
            system_prompt=old.agent.state.system_prompt, # 保留系统提示词
            tools=list(old.agent.state.tools),          # 保留工具列表
            session_id=new_session_id(),                # 生成新会话 ID
            messages=[],                                # 清空消息历史
            thinking_level=old.agent.state.thinking_level,
            tool_execution=old.tool_execution,
            max_context_messages=old.max_context_messages,
            max_context_tokens=old.max_context_tokens,
            retain_recent_messages=old.retain_recent_messages,
            summary_builder=old.summary_builder,
            retry_enabled=old.retry_enabled,
            max_retries=old.max_retries,
            retry_base_delay_ms=old.retry_base_delay_ms,
            mcp_servers=old.mcp_servers,
            mcp_client=old.mcp_client,
            extension_commands=old.extension_commands,
            before_prompt_hooks=old.before_prompt_hooks,
            after_prompt_hooks=old.after_prompt_hooks,
            before_tool_call=old.before_tool_call,
            after_tool_call=old.after_tool_call,
        )
    )


async def _handle_interactive_command(
    session: AgentSession, text: str, *, output: OutputFn = print
) -> tuple[bool, AgentSession | None]:
    """处理交互模式下的 "/" 命令。

    返回:
        (handled, switched_session)
        - handled: 是否成功处理了命令
        - switched_session: 若不为 None，表示需要切换到新会话

    支持的命令：
        /help              - 显示帮助信息
        /session           - 显示当前会话 ID 和叶子节点 ID
        /tree              - 显示会话树结构
        /clear             - 清空上下文，创建新会话
        /new, /fork        - 从指定 entry 分叉新会话
        /switch <entry_id> - 切换到指定 entry
        其他注册的扩展命令   - 由扩展处理器处理
    """
    # 解析命令和参数（命令和参数之间用空格分隔）
    cmd, _, rest = text.partition(" ")
    arg = rest.strip()

    # /help — 显示所有可用命令
    if cmd == "/help":
        output(format_commands_for_help(session))
        return True, None

    # /session — 显示当前会话 ID 和叶子节点 ID
    if cmd == "/session":
        output(f"session_id={session.session_id} leaf_id={session.get_leaf_id()}")
        return True, None

    # /tree — 以缩进树形结构显示会话的所有 entry
    if cmd == "/tree":
        entries = session.list_entries()
        if not entries:
            output("(empty)")
            return True, None
        for item in entries:
            depth = int(item.get("depth", 0))
            prefix = "  " * max(depth, 0)
            leaf_mark = " *" if item.get("is_leaf") else ""
            output(f"{prefix}- {item.get('id')}{leaf_mark}")
        return True, None

    # /clear — 清空上下文，创建全新会话
    if cmd == "/clear":
        fresh = _create_fresh_session(session)
        output(f"context cleared → new session_id={fresh.session_id}")
        return True, fresh

    # /new 或 /fork — 从指定 entry 分叉出新会话
    if cmd in {"/new", "/fork"}:
        from_entry = arg or session.get_leaf_id() or ""
        if not from_entry:
            output("cannot resolve source entry")
            return True, None
        forked = session.fork_from_entry(from_entry)
        output(f"forked to session_id={forked.session_id}")
        return True, forked

    # /switch — 切换当前会话的叶子节点到指定 entry
    if cmd == "/switch":
        if not arg:
            output("usage: /switch <entry_id>")
            return True, None
        session.switch_to_entry(arg)
        output(f"switched leaf -> {session.get_leaf_id()}")
        return True, None

    # 尝试匹配扩展注册的命令（如 /commit, /review 等）
    reg = resolve_registered_command(session, cmd)
    if reg:
        value = reg.handler(
            ExtensionCommandContext(
                name=reg.name,
                args=[p for p in arg.split(" ") if p],
                raw_text=text,
                session=session,
                message=None,
            )
        )
        # 支持异步命令处理器
        if inspect.isawaitable(value):
            value = await value
        if value:
            output(str(value))
        return True, None

    # 未识别的命令
    return False, None


async def run(options: RunOptions) -> AssistantMessage | None:
    """统一运行入口。

    根据 mode 路由到对应的运行模式：
    - print:       单次问答，需要提供 prompt
    - interactive: 交互式 REPL（默认）
    - rpc:         JSON-RPC 模式，供外部程序调用
    """
    # print 模式：单次问答
    if options.mode == "print":
        if not options.prompt:
            raise ValueError("print mode requires prompt")
        return await run_print(
            options.session,
            options.prompt,
            output=options.output,
            show_tool_events=options.show_tool_events,
        )

    # rpc 模式：JSON-RPC 协议
    if options.mode == "rpc":
        await run_rpc(options.session, output=options.output)
        return None

    # interactive 模式（默认）：交互式 REPL
    await run_interactive(
        options.session,
        input_fn=options.input_fn,
        output=options.output,
        show_tool_events=options.show_tool_events,
        exit_commands=options.exit_commands,
    )
    return None


async def run_rpc(
    session: AgentSession,
    *,
    output: OutputFn = print,
) -> None:
    """极简 RPC 模式（JSONL 协议）。

    通过 stdin/stdout 以 JSON 行格式与外部程序通信。
    每行是一个 JSON 对象，包含 type 字段标识请求类型。

    支持的请求类型：
    - {"type":"prompt","text":"..."}     发送用户输入
    - {"type":"continue"}                继续上一次未完成的运行
    - {"type":"state"}                   查询当前会话状态
    - {"type":"list_entries"}            列出所有 entry
    - {"type":"show_tree"}               显示会话树
    - {"type":"entry_path","entry_id":".."}  获取 entry 的路径
    - {"type":"fork_entry","entry_id":".."}  从 entry 分叉新会话
    - {"type":"switch_entry","entry_id":".."} 切换到指定 entry
    - {"type":"get_commands"}            获取所有注册命令
    - {"type":"shutdown"}                关闭 RPC 连接

    响应格式：
    - {"type":"response","id":"...","command":"...","status":"ok","data":{...}}
    - {"type":"response","id":"...","command":"...","status":"error","error":{...}}
    - {"type":"event","event":{...}}   Agent 事件推送
    """
    # ── 辅助函数 ──────────────────────────────────────────────

    def _json_default(value: Any) -> Any:
        """JSON 序列化时的默认转换器，处理 dataclass 和 set 类型。"""
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, set):
            return list(value)
        return str(value)

    def _emit(obj: dict[str, Any]) -> None:
        """向 stdout 输出一行 JSON。"""
        output(json.dumps(obj, ensure_ascii=False, default=_json_default))

    def _emit_error(*, req_id: Any, command: Any, code: str, message: str) -> None:
        """输出错误响应。"""
        _emit(
            {
                "type": "response",
                "id": req_id,
                "command": command,
                "status": "error",
                "error": {"code": code, "message": message},
            }
        )

    def _emit_ok(*, req_id: Any, command: str, data: dict[str, Any] | None = None) -> None:
        """输出成功响应。"""
        payload: dict[str, Any] = {
            "type": "response",
            "id": req_id,
            "command": command,
            "status": "ok",
        }
        if data is not None:
            payload["data"] = data
        _emit(payload)

    # ── 事件订阅 ──────────────────────────────────────────────
    # 将 Agent 事件实时推送给调用方
    unsubscribe = session.subscribe(
        lambda event: _emit({"type": "event", "event": event})
    )

    # 发送就绪信号
    _emit({"type": "rpc_ready", "session_id": session.session_id, "protocol_version": "1.2"})

    try:
        # ── 主循环：逐行读取 stdin 并处理请求 ────────────────
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue

            # 解析 JSON
            try:
                req = json.loads(line)
            except Exception as exc:
                _emit_error(req_id=None, command=None, code="invalid_json", message=f"Invalid JSON: {exc}")
                continue

            if not isinstance(req, dict):
                _emit_error(req_id=None, command=None, code="invalid_request", message="Request must be object")
                continue

            cmd = req.get("type")   # 请求类型
            req_id = req.get("id")  # 请求 ID，用于关联响应

            try:
                # ── 请求路由 ──────────────────────────────────

                if cmd == "prompt":
                    # 发送用户输入给 Agent
                    text = str(req.get("text", ""))
                    await session.prompt(text)
                    _emit_ok(req_id=req_id, command="prompt")

                elif cmd == "continue":
                    # 继续上一次未完成的运行（如工具调用后）
                    await session.continue_run()
                    _emit_ok(req_id=req_id, command="continue")

                elif cmd == "state":
                    # 查询当前会话状态
                    _emit_ok(
                        req_id=req_id,
                        command="state",
                        data={
                            "session_id": session.session_id,
                            "message_count": len(session.messages),
                            "entry_ids": session.list_entry_ids(),
                            "leaf_id": session.get_leaf_id(),
                        },
                    )

                elif cmd == "list_entries":
                    # 列出所有 entry（扁平列表，含导航信息）
                    _emit_ok(
                        req_id=req_id,
                        command="list_entries",
                        data={
                            "session_id": session.session_id,
                            "entry_ids": session.list_entry_ids(),
                            "entries": session.list_entries(),
                            "leaf_id": session.get_leaf_id(),
                        },
                    )

                elif cmd == "show_tree":
                    # 显示会话树结构（按 parent_id 组织）
                    _emit_ok(
                        req_id=req_id,
                        command="show_tree",
                        data={
                            "session_id": session.session_id,
                            "tree": session.get_session_tree(),
                            "leaf_id": session.get_leaf_id(),
                        },
                    )

                elif cmd == "entry_path":
                    # 获取指定 entry 从根到该节点的路径
                    entry_id = str(req.get("entry_id", ""))
                    if not entry_id:
                        raise ValueError("entry_path requires entry_id")
                    _emit_ok(
                        req_id=req_id,
                        command="entry_path",
                        data={"session_id": session.session_id, "entry_id": entry_id, "path": session.get_entry_path(entry_id)},
                    )

                elif cmd == "fork_entry":
                    # 从指定 entry 分叉出新会话
                    entry_id = str(req.get("entry_id", ""))
                    if not entry_id:
                        raise ValueError("fork_entry requires entry_id")
                    forked = session.fork_from_entry(entry_id)
                    try:
                        _emit_ok(
                            req_id=req_id,
                            command="fork_entry",
                            data={
                                "from_session_id": session.session_id,
                                "from_entry_id": entry_id,
                                "new_session_id": forked.session_id,
                            },
                        )
                    finally:
                        forked.close()

                elif cmd == "switch_entry":
                    # 切换当前会话的叶子节点到指定 entry
                    entry_id = str(req.get("entry_id", ""))
                    if not entry_id:
                        raise ValueError("switch_entry requires entry_id")
                    session.switch_to_entry(entry_id)
                    _emit_ok(
                        req_id=req_id,
                        command="switch_entry",
                        data={
                            "session_id": session.session_id,
                            "entry_id": entry_id,
                            "path": session.get_entry_path(entry_id),
                        },
                    )

                elif cmd == "get_commands":
                    # 获取所有注册的扩展命令
                    _emit_ok(
                        req_id=req_id,
                        command="get_commands",
                        data={
                            "session_id": session.session_id,
                            "commands": [
                                {"name": c.name, "description": c.description, "source": c.source}
                                for c in list_runtime_commands(session)
                            ],
                        },
                    )

                elif cmd == "shutdown":
                    # 关闭 RPC 连接
                    _emit_ok(req_id=req_id, command="shutdown")
                    return

                else:
                    # 未知命令
                    _emit_error(req_id=req_id, command=cmd, code="unknown_command", message="Unknown command")

            except Exception as exc:
                # 命令执行异常
                _emit_error(req_id=req_id, command=cmd, code="execution_error", message=str(exc))
    finally:
        # 确保取消事件订阅
        unsubscribe()
