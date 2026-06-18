from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_project_layout_keeps_runtime_assets_grouped() -> None:
    assert (ROOT / "src" / "codepilot").is_dir()
    assert (ROOT / "test").is_dir()
    assert not (ROOT / "tests").exists()

    assert not (ROOT / "docker-compose.yml").exists()
    assert not (ROOT / "docker").exists()
    assert not (ROOT / "scripts").exists()
    assert not (ROOT / "Dockerfile").exists()
    assert not (ROOT / "dev.sh").exists()
    assert not (ROOT / "dev.ps1").exists()
    assert not (ROOT / ".env.example").exists()
    assert not (ROOT / ".env.ps1.example").exists()
