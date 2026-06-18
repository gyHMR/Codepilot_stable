from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_source_files_do_not_contain_common_mojibake_markers() -> None:
    markers = ("锛", "鎬", "浼", "灏", "杩", "棰", "闀", "馃", "鈥", "�")
    offenders: list[str] = []
    paths = [
        *(ROOT / "src" / "codepilot").rglob("*.py"),
        *(ROOT / "docs").rglob("*.md"),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in markers):
            offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []
