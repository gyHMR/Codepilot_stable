from __future__ import annotations

# 新手导读：workspace_state.py 负责会话层需要的文件状态快照。
# 关注点：它服务 freshness、rollback 和 context validation，不依赖 tools 的执行管线。

import hashlib
from pathlib import Path
from typing import Any


def file_state_for_path(workspace_dir: str | Path, path: str | Path) -> dict[str, Any]:
    """Return a bounded file-state snapshot for session freshness checks."""

    root = Path(workspace_dir).resolve()
    target = Path(path)
    resolved = target.resolve() if target.is_absolute() else (root / target).resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Path escapes workspace boundary") from exc
    if not resolved.exists() or not resolved.is_file():
        return {
            "path": relative,
            "exists": False,
            "workspace_path": str(root),
        }
    stat = resolved.stat()
    return {
        "path": relative,
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(resolved),
        "workspace_path": str(root),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["file_state_for_path"]
