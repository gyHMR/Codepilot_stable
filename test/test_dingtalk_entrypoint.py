from __future__ import annotations

import ast
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_dingtalk_parser_supports_serve_help(capsys: pytest.CaptureFixture[str]) -> None:
    from codepilot.interfaces.dingtalk import main as dingtalk_main

    parser = dingtalk_main.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["serve", "--help"])

    output = capsys.readouterr().out
    assert exc_info.value.code == 0
    assert "start DingTalk Stream bridge" in output
    assert "--allowed-user" in output
    assert "--allow-dirty" in output


def test_dingtalk_main_requires_credentials(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from codepilot.interfaces.dingtalk import main as dingtalk_main

    monkeypatch.delenv("DINGTALK_CLIENT_ID", raising=False)
    monkeypatch.delenv("DINGTALK_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("CODEPILOT_DINGTALK_ALLOWED_USERS", raising=False)

    code = dingtalk_main.main(
        ["serve", "--cwd", str(tmp_path), "--allowed-user", "user_1"]
    )

    assert code == 2
    assert "codepilot-dingtalk: DINGTALK_CLIENT_ID" in capsys.readouterr().err


def test_dingtalk_main_requires_allowed_user(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from codepilot.interfaces.dingtalk import main as dingtalk_main

    monkeypatch.setenv("DINGTALK_CLIENT_ID", "client")
    monkeypatch.setenv("DINGTALK_CLIENT_SECRET", "secret")
    monkeypatch.delenv("CODEPILOT_DINGTALK_ALLOWED_USERS", raising=False)

    code = dingtalk_main.main(["serve", "--cwd", str(tmp_path)])

    assert code == 2
    assert "allowed-user" in capsys.readouterr().err


def test_dingtalk_main_rejects_invalid_model(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from codepilot.interfaces.dingtalk import main as dingtalk_main

    monkeypatch.setenv("DINGTALK_CLIENT_ID", "client")
    monkeypatch.setenv("DINGTALK_CLIENT_SECRET", "secret")

    code = dingtalk_main.main(
        [
            "serve",
            "--cwd",
            str(tmp_path),
            "--allowed-user",
            "user_1",
            "--model",
            "missing-provider-separator",
        ]
    )

    assert code == 2
    assert "--model must use provider/model-id format" in capsys.readouterr().err


def test_dingtalk_main_reports_missing_optional_sdk(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from codepilot.interfaces.dingtalk import main as dingtalk_main

    def missing_transport(**_kwargs):
        raise RuntimeError(
            'DingTalk Stream SDK is not installed. Install it with `pip install "codepilot[dingtalk]"`.'
        )

    monkeypatch.setenv("DINGTALK_CLIENT_ID", "client")
    monkeypatch.setenv("DINGTALK_CLIENT_SECRET", "secret")
    monkeypatch.setattr(dingtalk_main, "create_stream_transport", missing_transport)

    code = dingtalk_main.main(
        ["serve", "--cwd", str(tmp_path), "--allowed-user", "user_1"]
    )

    assert code == 2
    assert "codepilot[dingtalk]" in capsys.readouterr().err


def test_dingtalk_entrypoint_does_not_import_cli() -> None:
    import codepilot.interfaces.dingtalk.main as dingtalk_main

    tree = ast.parse(Path(dingtalk_main.__file__).read_text(encoding="utf-8"))
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")

    assert all("codepilot.interfaces.cli" not in item for item in imported_modules)
    assert all(not item.startswith("interfaces.cli") for item in imported_modules)
