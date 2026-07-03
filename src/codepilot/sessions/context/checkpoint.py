from __future__ import annotations

# 新手导读：ContextCheckpointManager 在高压力下保存结构化 checkpoint。
# 关注点：checkpoint 不是完整历史压缩，而是给下一轮保留目标和下一步动作。

"""结构化上下文 checkpoint。"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from codepilot.protocols import ContextCheckpoint
from codepilot.sessions.layout import SessionLayout


class ContextCheckpointManager:
    """管理 Session 级 ContextCheckpoint。"""

    def __init__(self, *, workspace_dir: str | Path, session_id: str) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.session_id = session_id
        self.layout = SessionLayout.for_workspace(self.workspace_dir, self.session_id)
        self.root = self.layout.session_dir
        self.file = self.layout.context_ledger_file

    def create(
        self,
        *,
        goal: str,
        active_files: list[str],
        changed_files: list[str],
        key_evidence: list[str],
        verification_state: str,
        next_actions: list[str],
        source_refs: list[str],
        open_questions: list[str] | None = None,
    ) -> ContextCheckpoint:
        checkpoint = ContextCheckpoint(
            goal=goal,
            active_files=list(active_files),
            changed_files=list(changed_files),
            key_evidence=list(key_evidence),
            verification_state=verification_state,
            open_questions=list(open_questions or []),
            next_actions=list(next_actions),
            source_refs=list(source_refs),
        )
        self.append(checkpoint)
        return checkpoint

    def append(self, checkpoint: ContextCheckpoint) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.file.open("a", encoding="utf-8", newline="\n") as fp:
            payload = {"type": "checkpoint", **asdict(checkpoint)}
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def load_all(self) -> list[ContextCheckpoint]:
        if not self.file.exists():
            return []
        items: list[ContextCheckpoint] = []
        for line in self.file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict) and payload.get("type") == "checkpoint":
                items.append(_checkpoint_from_dict(payload))
        return items

    def load_latest(self) -> ContextCheckpoint | None:
        items = self.load_all()
        return items[-1] if items else None


def _checkpoint_from_dict(payload: dict[str, Any]) -> ContextCheckpoint:
    return ContextCheckpoint(
        goal=str(payload.get("goal", "")),
        active_files=_string_list(payload.get("active_files")),
        changed_files=_string_list(payload.get("changed_files")),
        key_evidence=_string_list(payload.get("key_evidence")),
        verification_state=str(payload.get("verification_state", "unknown")),
        open_questions=_string_list(payload.get("open_questions")),
        next_actions=_string_list(payload.get("next_actions")),
        source_refs=_string_list(payload.get("source_refs")),
    )


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


__all__ = ["ContextCheckpointManager"]
