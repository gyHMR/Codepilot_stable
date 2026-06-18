from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_project_layout_keeps_runtime_assets_grouped() -> None:
    assert (ROOT / "src" / "codepilot").is_dir()
    assert (ROOT / "test").is_dir()
    assert not (ROOT / "tests").exists()

    assert (ROOT / "docker-compose.yml").is_file()
    assert (ROOT / "docker" / "Dockerfile").is_file()
    assert (ROOT / "docker" / "env.example").is_file()
    assert (ROOT / "docker" / "env.ps1.example").is_file()

    assert (ROOT / "scripts" / "dev.sh").is_file()
    assert (ROOT / "scripts" / "dev.ps1").is_file()

    assert not (ROOT / "Dockerfile").exists()
    assert not (ROOT / "dev.sh").exists()
    assert not (ROOT / "dev.ps1").exists()
    assert not (ROOT / ".env.example").exists()
    assert not (ROOT / ".env.ps1.example").exists()
