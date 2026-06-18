from __future__ import annotations

import sys
from pathlib import Path
import asyncio


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class FakeAdapter:
    def __init__(self) -> None:
        self.texts = []

    def handle_webhook(self, _headers, _body):
        raise NotImplementedError

    def send_text(self, message):
        self.texts.append(message)
        return "msg_1"

    def update_text(self, _message_id, _text):
        return None

    def send_card(self, _message):
        return "card_1"

    def get_user_info(self, _user_id):
        return None

    def get_chat_info(self, _chat_id):
        return None


def test_im_control_commands_do_not_require_llm(tmp_path: Path) -> None:
    asyncio.run(_run_im_control_command_case(tmp_path))


async def _run_im_control_command_case(tmp_path: Path) -> None:
    from codepilot.interfaces.im.service import IMService, IMServiceConfig
    from codepilot.interfaces.im.session_router import SessionRouter
    from codepilot.interfaces.im.types import IMIncomingMessage

    adapter = FakeAdapter()
    service = IMService(
        adapter,
        IMServiceConfig(workspace_dir=tmp_path, provider="unit-test", model_id="unit-test", stream_updates=False),
        router=SessionRouter(tmp_path),
    )
    message = IMIncomingMessage(platform="feishu", channel_id="chat_1", user_id="user_1", text="/session")

    assert await service._handle_control_command(message) is True
    assert "Current session:" in adapter.texts[-1].text

    message.text = "/new"
    assert await service._handle_control_command(message) is True
    assert "Created a new session:" in adapter.texts[-1].text
