"""构建 Web 项目栏使用的安全工作区摘要。"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def workspace_summary(workspace: Path) -> dict[str, Any]:
    root = Path(workspace).resolve()
    git = _git_summary(root)
    return {"name": root.name or str(root), "path": str(root), "git": git}


def _git_summary(workspace: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "-b", "--untracked-files=all"],
            cwd=workspace,
            check=False,
            capture_output=True,
            timeout=3,
            encoding="utf-8",
            errors="replace",
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return _unavailable_git()
    if result.returncode != 0:
        return _unavailable_git()

    lines = result.stdout.splitlines()
    branch = _branch_name(lines[0]) if lines and lines[0].startswith("## ") else None
    changes = [_change(line) for line in lines if line and not line.startswith("## ")]
    return {
        "available": True,
        "branch": branch,
        "clean": not changes,
        "change_count": len(changes),
        "changes": changes,
    }


def _branch_name(header: str) -> str:
    value = header[3:].split("...", 1)[0].strip()
    if value.startswith("HEAD ") or value == "HEAD":
        return "detached HEAD"
    return value or "unknown"


def _change(line: str) -> dict[str, str]:
    code = line[:2]
    path = line[3:].split(" -> ")[-1]
    if "?" in code or "A" in code:
        kind = "added"
    elif "D" in code:
        kind = "deleted"
    else:
        kind = "modified"
    return {"path": path, "status": kind}


def _unavailable_git() -> dict[str, Any]:
    return {
        "available": False,
        "branch": None,
        "clean": True,
        "change_count": 0,
        "changes": [],
    }


__all__ = ["workspace_summary"]
