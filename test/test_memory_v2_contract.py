from __future__ import annotations

import json
from pathlib import Path

import pytest


def _session_store(tmp_path: Path):
    from codepilot.sessions.store import SessionStore

    store = SessionStore(tmp_path, "session_memory_contract")
    store.ensure_initialized(model_id="test", provider="test", system_prompt="")
    return store


def test_old_memory_normalizer_is_not_public() -> None:
    import codepilot.sessions.memory as memory

    assert not hasattr(memory, "normalize_memory_record_payload")
    with pytest.raises(ImportError):
        from codepilot.sessions.memory import normalize_memory_record_payload  # type: ignore  # noqa: F401


def test_memory_store_rejects_old_jsonl_instead_of_normalizing(tmp_path: Path) -> None:
    from codepilot.sessions.memory import MemoryStore

    session_store = _session_store(tmp_path)
    memory_file = session_store.layout.project_memory_file
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    memory_file.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "id": "legacy",
                "kind": "constraint",
                "key": "constraint:legacy",
                "text": "Legacy memory text.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="Unsupported memory schema_version"):
        MemoryStore(session_store).all_records()


def test_memory_files_module_does_not_export_global_memory_runtime_tools() -> None:
    import codepilot.sessions.memory as files

    assert not hasattr(files, "load_global_memory")
    assert not hasattr(files, "save_global_memory")


def test_memory_recall_result_has_single_canonical_shape() -> None:
    from codepilot.sessions.memory import MemoryRecall

    recall = MemoryRecall()
    assert recall.retrieved == []
    assert recall.dropped == {}
    assert not hasattr(recall, "pinned_text")
    assert not hasattr(recall, "always")
    assert not hasattr(recall, "selected")
