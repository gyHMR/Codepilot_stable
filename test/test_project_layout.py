from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_project_layout_keeps_runtime_assets_grouped() -> None:
    assert (ROOT / "src" / "codepilot").is_dir()
    assert (ROOT / "test").is_dir()
    assert not (ROOT / "tests").exists()

    assert not (ROOT / "docker-compose.yml").exists()
    assert not (ROOT / "docker").exists()
    script_files = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "scripts").glob("*.py")
    ) if (ROOT / "scripts").exists() else []
    assert script_files == ["scripts/run_evaluation_v2.py"]
    assert not (ROOT / "Dockerfile").exists()
    assert not (ROOT / "dev.sh").exists()
    assert not (ROOT / "dev.ps1").exists()
    assert not (ROOT / ".env.example").exists()
    assert not (ROOT / ".env.ps1.example").exists()


def test_project_metadata_focuses_on_cli_and_dingtalk() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    assert "IM bridge" not in project["description"]
    assert all("lark" not in dependency.lower() for dependency in project["dependencies"])
    assert "feishu" not in project.get("optional-dependencies", {})
    assert all("dingtalk-stream" not in dependency.lower() for dependency in project["dependencies"])
    assert project.get("optional-dependencies", {})["dingtalk"] == [
        "dingtalk-stream>=0.24.3,<0.25",
    ]
    assert project["scripts"] == {
        "codepilot": "codepilot.interfaces.cli.main:main",
        "codepilot-dingtalk": "codepilot.interfaces.dingtalk.main:main",
    }
