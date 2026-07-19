"""提供固定工作区的项目和 Git 摘要。"""

from __future__ import annotations

from fastapi import APIRouter, Request


router = APIRouter(prefix="/api/workspace")


@router.get("/summary")
async def summary(request: Request) -> dict[str, object]:
    return request.app.state.web_service.workspace_summary()


__all__ = ["router"]
