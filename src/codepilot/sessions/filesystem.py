"""会话文件系统布局和 I/O 原语 —— 读写 JSON/JSONL 文件的原子操作。

本文件提供 sessions 层在文件系统上的存储布局和 I/O 工具：
1. SessionFileLayout — 文件路径解析（session.json、messages.jsonl、run.json 等）
2. 原子写（atomic_write_json）—— 通过临时文件 + os.replace 实现
3. JSONL 追加（append_jsonl）—— 追加写消息和事件
4. JSONL 读取（read_jsonl）—— 读取消息链，容忍尾部不完整的行
"""

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SessionFileLayout:
    """会话文件布局 —— 解析会话相关文件在 .codepilot 目录下的路径。

    文件结构:
    .codepilot/
      sessions/
        {session_id}/
          session.json      — 会话状态
          messages.jsonl    — 消息列表（JSONL 格式）
          events.jsonl      — 事件列表（JSONL 格式）
      runs/
        {run_id}/
          run.json          — 运行状态
          events.jsonl      — 运行事件列表
          artifacts/        — 运行生成的文件

    参数:
        workspace_dir: 工作区根目录（Path 或字符串）
    """

    workspace_dir: Path

    def __init__(self, workspace_dir: str | Path) -> None:
        object.__setattr__(self, "workspace_dir", Path(workspace_dir))

    @property
    def codepilot_dir(self) -> Path:
        """.codepilot 目录路径。"""
        return self.workspace_dir / ".codepilot"

    def session_dir(self, session_id: str) -> Path:
        """会话数据目录。"""
        return self.codepilot_dir / "sessions" / session_id

    def session_file(self, session_id: str) -> Path:
        """会话状态 JSON 文件。"""
        return self.session_dir(session_id) / "session.json"

    def messages_file(self, session_id: str) -> Path:
        """消息 JSONL 文件。"""
        return self.session_dir(session_id) / "messages.jsonl"

    def session_events_file(self, session_id: str) -> Path:
        """会话事件 JSONL 文件。"""
        return self.session_dir(session_id) / "events.jsonl"

    def run_dir(self, run_id: str) -> Path:
        """运行数据目录。"""
        return self.codepilot_dir / "runs" / run_id

    def run_file(self, run_id: str) -> Path:
        """运行状态 JSON 文件。"""
        return self.run_dir(run_id) / "run.json"

    def run_events_file(self, run_id: str) -> Path:
        """运行事件 JSONL 文件。"""
        return self.run_dir(run_id) / "events.jsonl"

    def run_artifacts_dir(self, run_id: str) -> Path:
        """运行生成文件的目录。"""
        return self.run_dir(run_id) / "artifacts"


def read_json_object(path: Path) -> dict[str, Any] | None:
    """读取 JSON 文件并返回 dict。

    如果文件不存在，返回 None。
    如果 JSON 不是对象（dict），抛出 ValueError。

    参数:
        path: JSON 文件路径

    返回:
        dict（文件存在时）或 None（文件不存在时）
    """
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """原子化写入 JSON 文件。

    使用"写临时文件 → fsync → os.replace"的策略确保：
    - 写入过程中如果程序崩溃，不会损坏原文件
    - 写入完成后原文件被原子替换

    处理流程:
    1. 创建父目录（如果不存在）
    2. 在目标同目录创建临时文件（.文件名.uuid.tmp）
    3. 写入 JSON 数据后 fsync
    4. os.replace 替换原文件
    5. fsync 父目录（POSIX 系统上确保目录条目落盘）

    参数:
        path: 目标文件路径
        payload: 要序列化的 JSON 对象
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    data = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _fsync_directory(path.parent)
    finally:
        if temp.exists():
            temp.unlink()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """追加一行 JSON 到 JSONL 文件。

    处理流程:
    1. 创建父目录（如果不存在）
    2. 以追加模式打开文件
    3. 写入一行 JSON（不含多余空格）+ 换行符
    4. fsync 确保数据落盘

    参数:
        path: JSONL 文件路径
        payload: 要追加的 JSON 对象
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件，返回 dict 列表。

    特殊处理：
    - 如果文件不存在，返回空列表
    - 如果文件为空，返回空列表
    - 如果最后一行不完整（文件末尾没有换行符），忽略该行
      （因为 append_jsonl 是追加写的，写入过程中程序崩溃可能留下不完整的行）

    参数:
        path: JSONL 文件路径

    返回:
        dict 列表（每行一个对象）
    """
    if not path.exists():
        return []
    raw = path.read_bytes()
    if not raw:
        return []
    lines = raw.splitlines()
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            # 最后一行不完整（写入过程中崩溃） → 忽略
            is_last = index == len(lines) - 1
            has_trailing_newline = raw.endswith(b"\n") or raw.endswith(b"\r")
            if is_last and not has_trailing_newline:
                break
            raise ValueError(f"Invalid JSONL record on line {index + 1}: {path}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record on line {index + 1} must be an object: {path}")
        rows.append(value)
    return rows


def _fsync_directory(path: Path) -> None:
    """同步目录条目（POSIX 系统）。

    os.replace 后，需要 fsync 父目录确保目录元数据写入磁盘。
    Windows 上不需要（NTFS 的原子替换由操作系统保证）。
    """
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "SessionFileLayout",
    "append_jsonl",
    "atomic_write_json",
    "read_json_object",
    "read_jsonl",
]