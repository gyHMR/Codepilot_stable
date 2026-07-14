"""提供 Web 服务健康状态和只读配置检查路由。"""

from __future__ import annotations

from fastapi import APIRouter, Request


router = APIRouter(prefix="/api")


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/config")
async def config(request: Request) -> dict[str, object]:
    service = request.app.state.web_service
    sessions = await service.list_sessions()
    current = sessions[0] if sessions else {}
    return {
        "workspace": str(service.workspace),
        "model_id": current.get("model_id"),
        "permission_mode": current.get("permission_mode"),
        "current_mode": current.get("current_mode"),
        "features": {
            "sse": True,
            "approvals": True,
            "cancellation": True,
            "authentication": False,
        },
    }
