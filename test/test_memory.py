from __future__ import annotations

from pathlib import Path

from codepilot.sessions.memory import MemoryQuery, MemoryService


def test_explicit_prompt_admission_uses_the_new_service_and_recall_port(tmp_path: Path) -> None:
    service = MemoryService(
        workspace_dir=tmp_path / "workspace",
        user_home=tmp_path / "home",
    )

    receipt = service.admit_user_prompt(
        "请记住：本项目默认使用 python -m pytest 运行测试。",
        session_id="session_1",
        run_id="run_1",
    )
    recalled = service.recall(
        MemoryQuery(user_request="运行项目测试", task_goal="验证修改")
    )

    assert len(receipt.records) == 1
    assert receipt.records[0].status == "active"
    assert [item.memory_id for item in recalled.retrieved] == [receipt.records[0].id]
