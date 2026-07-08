from __future__ import annotations

"""Built-in workspace_status tool."""

import json
import subprocess
from typing import Any, Callable

from codepilot.protocols import TextContent
from codepilot.tools.contracts import ToolCallRequest, ToolDefinition, ToolResult
from codepilot.tools.registry import get_builtin_tool_metadata
from codepilot.tools.sandbox import WorkspaceSandbox


def create_workspace_tools(
    sandbox: WorkspaceSandbox,
    *,
    allow: Callable[[str], bool],
) -> list[ToolDefinition]:
    if not allow("workspace_status"):
        return []

    async def workspace_status(
        request: ToolCallRequest,
        signal=None,
        on_update=None,
    ) -> ToolResult:
        _ = request, signal, on_update
        root = sandbox.root
        branch = _git(root, ["branch", "--show-current"])
        head = _git(root, ["rev-parse", "--short", "HEAD"])
        status_text = _git(root, ["status", "--porcelain", "--", "."])
        total_changed = 0
        if status_text is None:
            payload = {
                "is_git_repo": False,
                "branch": None,
                "head": None,
                "dirty": None,
                "changed_paths": [],
                "diff_stat": None,
            }
        else:
            changed_paths = []
            for line in status_text.splitlines():
                if len(line) < 4:
                    continue
                changed_paths.append(
                    {
                        "path": line[3:].split(" -> ")[-1],
                        "status": _status_name(line[:2]),
                        "code": line[:2],
                    }
                )
            total_changed = len(changed_paths)
            diff_stat = _git(root, ["diff", "--stat", "--", "."])
            payload = {
                "is_git_repo": True,
                "branch": branch or None,
                "head": head or None,
                "dirty": bool(changed_paths),
                "changed_paths": changed_paths[:200],
                "diff_stat": (diff_stat or "")[-2000:] or None,
            }
        return ToolResult(
            content=[TextContent(text=json.dumps(payload, ensure_ascii=False, indent=2))],
            workspace_changed=False,
            details=payload,
            metadata={
                "changed_paths": payload["changed_paths"],
                "truncated": bool(payload["is_git_repo"] and total_changed > len(payload["changed_paths"])),
                "changed_path_count": total_changed,
            },
        )

    metadata = get_builtin_tool_metadata("workspace_status")
    if metadata is None:
        raise ValueError("Missing builtin metadata for workspace_status")
    return [
        ToolDefinition(
            name="workspace_status",
            label="Workspace Status",
            description="结构化查看 Git 分支、HEAD、工作区变更路径和 diff 统计。",
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


def _git(root, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.rstrip("\r\n")


def _status_name(code: str) -> str:
    if "?" in code:
        return "untracked"
    if "D" in code:
        return "deleted"
    if "A" in code:
        return "added"
    if "R" in code:
        return "renamed"
    return "modified"


__all__ = ["create_workspace_tools"]
