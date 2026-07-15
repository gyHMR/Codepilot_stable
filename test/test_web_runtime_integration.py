from __future__ import annotations

import asyncio


class EchoModelPort:
    async def stream(self, _request):
        from codepilot.llm.ports import LLMCompleted
        from codepilot.protocols import AssistantMessage, TextContent

        yield LLMCompleted(message=AssistantMessage(content=[TextContent(text="web echo")]))


class PlanModelPort:
    async def stream(self, _request):
        from codepilot.llm.ports import LLMCompleted
        from codepilot.protocols import AssistantMessage, ToolCall

        yield LLMCompleted(
            message=AssistantMessage(
                content=[
                    ToolCall(
                        id="web-plan",
                        name="propose_plan",
                        arguments={
                            "task_understanding": "用户希望审批登录改造方案。",
                            "current_implementation": "当前登录功能仍是演示实现。",
                            "target_design": "实现完整且可验证的登录服务。",
                            "impact_scope": "影响登录模块和测试。",
                            "risks_and_open_questions": ["暂无阻塞待确认项。"],
                            "verification_plan": "运行登录测试。",
                            "summary": "完善登录模块",
                            "completion_criteria": ["登录测试通过"],
                            "items": [
                                {
                                    "step": "实现登录服务",
                                    "details": "替换演示实现。",
                                    "verification": "运行登录测试。",
                                }
                            ],
                        },
                    )
                ]
            )
        )


def unit_model():
    from codepilot.protocols import Model

    return Model(
        id="web-runtime",
        name="Web Runtime",
        api="unit-test",
        provider="unit-test",
        base_url="",
        reasoning=False,
        input=["text"],
        context_window=32_000,
        max_tokens=500,
    )


def test_real_runtime_prompt_reaches_web_events_and_persistence(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.interfaces.web.service import WebService
        from codepilot.runtime.gateway import RuntimeGateway

        runtime = RuntimeGateway(model_port=EchoModelPort())
        service = WebService(
            runtime=runtime,
            workspace=tmp_path,
            session_open_options={"model": unit_model(), "memory_enabled": False},
        )
        created = await service.create_session()
        session_id = created["session_id"]

        await service.submit_prompt(session_id, "hello from web")
        await service.wait_for_idle(session_id)

        replay = service.events_for(session_id).replay_after(None)
        assert replay.events == ()
        all_events = tuple(service.events_for(session_id)._events)  # noqa: SLF001
        assert any(event.type == "run_finished" for event in all_events)
        messages = service.messages(session_id)
        assert any("web echo" in str(message) for message in messages)
        await service.shutdown()

    asyncio.run(run_case())


def test_web_session_detail_exposes_the_canonical_plan(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.interfaces.web.service import WebService
        from codepilot.runtime.gateway import RuntimeGateway

        service = WebService(
            runtime=RuntimeGateway(model_port=PlanModelPort()),
            workspace=tmp_path,
            session_open_options={
                "model": unit_model(),
                "memory_enabled": False,
                "current_mode": "plan",
            },
        )
        created = await service.create_session()
        session_id = created["session_id"]

        await service.submit_prompt(session_id, "完善登录模块，给我一个方案")
        await service.wait_for_idle(session_id)
        detail = await service.get_session(session_id)

        assert detail["plan"]["status"] == "proposed"
        assert detail["plan"]["revision"] == 1
        assert detail["plan"]["definition"]["summary"] == "完善登录模块"
        assert detail["plan"]["steps"][0]["step"] == "实现登录服务"
        await service.shutdown()

    asyncio.run(run_case())
