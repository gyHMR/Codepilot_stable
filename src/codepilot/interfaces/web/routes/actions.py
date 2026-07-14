"""定义提交消息、命令、审批和取消请求的 HTTP 路由。"""

from __future__ import annotations

from fastapi import APIRouter, Request, status

from ..schemas import AcceptedAction, ApprovalCreate, CommandCreate, MessageCreate


router = APIRouter(prefix="/api/sessions")


@router.post(
    "/{session_id}/messages",
    response_model=AcceptedAction,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_message(
    request: Request, session_id: str, payload: MessageCreate
) -> AcceptedAction:
    await request.app.state.web_service.ensure_open(session_id)
    return await request.app.state.web_service.submit_prompt(
        session_id, payload.text, payload.mode_hint
    )


@router.post(
    "/{session_id}/commands",
    response_model=AcceptedAction,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_command(
    request: Request, session_id: str, payload: CommandCreate
) -> AcceptedAction:
    await request.app.state.web_service.ensure_open(session_id)
    return await request.app.state.web_service.submit_command(session_id, payload.text)


@router.post(
    "/{session_id}/approvals/{approval_id}",
    response_model=AcceptedAction,
    status_code=status.HTTP_202_ACCEPTED,
)
async def decide_approval(
    request: Request,
    session_id: str,
    approval_id: str,
    payload: ApprovalCreate,
) -> AcceptedAction:
    return await request.app.state.web_service.decide_approval(
        session_id, approval_id, payload.decision, payload.reason
    )


@router.post(
    "/{session_id}/cancel",
    response_model=AcceptedAction,
    status_code=status.HTTP_202_ACCEPTED,
)
async def cancel(request: Request, session_id: str) -> AcceptedAction:
    return await request.app.state.web_service.cancel(session_id)
