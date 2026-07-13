from __future__ import annotations

import mimetypes
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from codepilot.runtime import RuntimeGateway

from .routes import actions, events, health, sessions
from .service import WebConflict, WebNotFound, WebService, WebServiceError


def create_app(
    *,
    workspace: Path,
    runtime: RuntimeGateway | None = None,
    frontend_dir: Path | None = None,
) -> FastAPI:
    _register_frontend_mime_types()
    gateway = runtime or RuntimeGateway()
    service = WebService(runtime=gateway, workspace=workspace)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        await service.shutdown()

    app = FastAPI(title="Codepilot Web", lifespan=lifespan)
    app.state.web_service = service
    app.include_router(health.router)
    app.include_router(sessions.router)
    app.include_router(actions.router)
    app.include_router(events.router)

    static_root = frontend_dir or (Path(__file__).parent / "static")
    if (static_root / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=static_root / "assets"), name="web-assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa_fallback(path: str):
        if path == "api" or path.startswith("api/"):
            return _error_response(404, "web.route_not_found", "API route not found")
        index = static_root / "index.html"
        if index.is_file():
            return FileResponse(index)
        return _error_response(
            503,
            "web.frontend_not_built",
            "Web frontend assets are missing; run the frontend build",
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(422, "web.validation_error", "Invalid request", exc.errors())

    @app.exception_handler(KeyError)
    async def missing_session(_request: Request, exc: KeyError) -> JSONResponse:
        return _error_response(404, "web.session_not_found", "Session not found", str(exc))

    @app.exception_handler(WebNotFound)
    async def web_not_found(_request: Request, exc: WebNotFound) -> JSONResponse:
        return _error_response(404, exc.code, exc.message)

    @app.exception_handler(WebConflict)
    async def web_conflict(_request: Request, exc: WebConflict) -> JSONResponse:
        return _error_response(409, exc.code, exc.message)

    @app.exception_handler(WebServiceError)
    async def web_error(_request: Request, exc: WebServiceError) -> JSONResponse:
        return _error_response(400, exc.code, exc.message)

    return app


def _register_frontend_mime_types() -> None:
    # Windows registry MIME mappings can classify JavaScript as text/plain,
    # which causes browsers to reject Vite's ES module entrypoint.
    mimetypes.add_type("text/javascript", ".js")
    mimetypes.add_type("text/javascript", ".mjs")
    mimetypes.add_type("text/css", ".css")


def create_app_from_env() -> FastAPI:
    workspace = Path(os.environ.get("CODEPILOT_WEB_WORKSPACE", "."))
    return create_app(workspace=workspace)


def _error_response(
    status_code: int,
    code: str,
    message: str,
    details: object | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "details": details}},
    )


__all__ = ["create_app", "create_app_from_env"]
