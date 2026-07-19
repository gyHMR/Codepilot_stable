"""提供 Session 创建、查询、消息读取和删除路由。"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from ..schemas import SessionCreate, SessionUpdate


router = APIRouter(prefix="/api/sessions")


@router.get("")
async def list_sessions(request: Request) -> list[dict[str, object]]:
    return await request.app.state.web_service.list_sessions()


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_session(
    request: Request, payload: SessionCreate
) -> dict[str, object]:
    return await request.app.state.web_service.create_session(title=payload.title)


@router.get("/{session_id}")
async def get_session(request: Request, session_id: str) -> dict[str, object]:
    return await request.app.state.web_service.get_session(session_id)


@router.patch("/{session_id}")
async def update_session(
    request: Request, session_id: str, payload: SessionUpdate
) -> dict[str, object]:
    return await request.app.state.web_service.update_session(session_id, payload.title)


@router.get("/{session_id}/timeline")
async def timeline(request: Request, session_id: str) -> list[dict[str, object]]:
    await request.app.state.web_service.ensure_open(session_id)
    return request.app.state.web_service.timeline(session_id)


@router.get("/{session_id}/messages")
async def messages(request: Request, session_id: str) -> list[dict[str, object]]:
    await request.app.state.web_service.ensure_open(session_id)
    return request.app.state.web_service.messages(session_id)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(request: Request, session_id: str) -> Response:
    await request.app.state.web_service.delete_session(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
