from __future__ import annotations

from types import SimpleNamespace


class ApiGateway:
    def __init__(self) -> None:
        self.opened = []
        self.actions = []

    def open_session(self, intent):
        self.opened.append(intent)
        return SimpleNamespace(session_id=intent.session_id or "s-new")

    def describe(self, session_id):
        return SimpleNamespace(
            status=SimpleNamespace(
                session_id=session_id,
                workspace=str(self.opened[-1].workspace_dir),
                model_id="unit/model",
                permission_mode="workspace-write",
                current_mode="build",
                is_running=False,
                message_count=0,
            ),
            pending_approvals=(),
            session=SimpleNamespace(messages=()),
        )

    async def dispatch(self, session_id, action):
        from codepilot.runtime.actions import ProgressFrame

        self.actions.append(action)
        yield ProgressFrame(event={"type": "status", "message": "ok"})

    def messages(self, session_id):
        return ()

    def close(self, session_id):
        return None

    async def close_all(self):
        return None


def test_health_and_safe_config(tmp_path, monkeypatch) -> None:
    from fastapi.testclient import TestClient
    from codepilot.interfaces.web.app import create_app

    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    app = create_app(workspace=tmp_path, runtime=ApiGateway())

    with TestClient(app) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        payload = client.get("/api/config").json()

    assert payload["workspace"] == str(tmp_path.resolve())
    assert "must-not-leak" not in str(payload)


def test_session_and_action_routes(tmp_path) -> None:
    from fastapi.testclient import TestClient
    from codepilot.interfaces.web.app import create_app

    gateway = ApiGateway()
    app = create_app(workspace=tmp_path, runtime=gateway)

    with TestClient(app) as client:
        created = client.post("/api/sessions", json={})
        assert created.status_code == 201
        session_id = created.json()["session_id"]
        assert client.get(f"/api/sessions/{session_id}").status_code == 200
        assert client.get(f"/api/sessions/{session_id}/messages").json() == []
        accepted = client.post(
            f"/api/sessions/{session_id}/messages", json={"text": "hello"}
        )
        assert accepted.status_code == 202


def test_blank_message_is_validation_error(tmp_path) -> None:
    from fastapi.testclient import TestClient
    from codepilot.interfaces.web.app import create_app

    app = create_app(workspace=tmp_path, runtime=ApiGateway())
    with TestClient(app) as client:
        client.post("/api/sessions", json={})
        response = client.post("/api/sessions/s-new/messages", json={"text": ""})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "web.validation_error"


def test_missing_session_returns_stable_404(tmp_path) -> None:
    from fastapi.testclient import TestClient
    from codepilot.interfaces.web.app import create_app

    class MissingGateway(ApiGateway):
        def open_session(self, intent):
            raise KeyError(intent.session_id)

    app = create_app(workspace=tmp_path, runtime=MissingGateway())
    with TestClient(app) as client:
        response = client.get("/api/sessions/missing")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "web.session_not_found"


def test_sse_route_is_registered_and_formats_events(tmp_path) -> None:
    from codepilot.interfaces.web.app import create_app
    from codepilot.interfaces.web.events import WebEvent
    from codepilot.interfaces.web.routes.events import format_sse

    app = create_app(workspace=tmp_path, runtime=ApiGateway())
    paths = set(app.openapi()["paths"])
    rendered = format_sse(
        WebEvent(
            event_id="e1",
            session_id="s1",
            type="progress",
            sequence=1,
            timestamp="now",
            data={"message": "你好"},
        )
    )

    assert "/api/sessions/{session_id}/events" in paths
    assert "id: e1\n" in rendered
    assert "event: progress\n" in rendered
    assert '"event_id":"e1"' in rendered
    assert '"data":{"message":"你好"}' in rendered


def test_static_assets_and_spa_fallback(tmp_path) -> None:
    from fastapi.testclient import TestClient
    from codepilot.interfaces.web.app import create_app

    frontend = tmp_path / "frontend"
    (frontend / "assets").mkdir(parents=True)
    (frontend / "index.html").write_text("<html>web deck</html>", encoding="utf-8")
    (frontend / "assets" / "app.js").write_text("console.log('ok')", encoding="utf-8")
    app = create_app(workspace=tmp_path, runtime=ApiGateway(), frontend_dir=frontend)

    with TestClient(app) as client:
        assert client.get("/assets/app.js").status_code == 200
        assert "web deck" in client.get("/sessions/s1").text
        assert client.get("/api/missing").status_code == 404
