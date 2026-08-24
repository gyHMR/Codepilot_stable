from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
src_path = str(SRC)
if src_path in sys.path:
    sys.path.remove(src_path)
sys.path.insert(0, src_path)


@pytest.fixture
def tmp_path() -> Path:
    base = Path(__file__).resolve().parent / ".tmp_runs"
    base.mkdir(exist_ok=True)
    path = base / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
