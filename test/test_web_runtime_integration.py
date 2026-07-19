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


class InteractionModelPort:
    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, _request):
        from codepilot.llm.ports import LLMCompleted
        from codepilot.protocols import AssistantMessage, TextContent, ToolCall

        self.calls += 1
        if self.calls == 1:
            yield LLMCompleted(message=AssistantMessage(content=[ToolCall(
                id="interaction-call",
                name="request_user_input",
                arguments={
                    "prompt": "选择界面",
                    "options": ["CLI", "Web"],
                    "allow_free_text": True,
                },
            )]))
            return
        yield LLMCompleted(message=AssistantMessage(content=[TextContent(text="interaction resumed")]))


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
        assert any(event.type == "run.completed" for event in all_events)
        messages = service.messages(session_id)
        assert any("web echo" in str(message) for message in messages)
        await service.shutdown()

    asyncio.run(run_case())


def test_web_projection_recovers_terminal_state_and_accepts_next_input(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.interfaces.web.service import WebService
        from codepilot.runtime.gateway import RuntimeGateway

        service = WebService(
            runtime=RuntimeGateway(model_port=EchoModelPort()),
            workspace=tmp_path,
            session_open_options={"model": unit_model(), "memory_enabled": False},
        )

        session_id = (await service.create_session())["session_id"]

        await service.submit_prompt(session_id, "first input")
        await service.wait_for_idle(session_id)
        first_projection = service.projection(session_id)
        assert first_projection["execution"] == {"run_id": None, "status": "idle"}
        assert any("web echo" in str(item) for item in service.timeline(session_id))

        await service.submit_prompt(session_id, "second input")
        await service.wait_for_idle(session_id)
        second_projection = service.projection(session_id)
        assert second_projection["execution"] == {"run_id": None, "status": "idle"}
        user_items = [item for item in service.timeline(session_id) if item["type"] == "user_message"]
        assert len(user_items) == 2
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
        assert service.projection(session_id)["execution"]["status"] == "waiting_plan"
        await service.shutdown()

    asyncio.run(run_case())


def test_web_interaction_wait_is_projected_and_resumed_explicitly(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.interfaces.web.service import WebService
        from codepilot.runtime.gateway import RuntimeGateway

        service = WebService(
            runtime=RuntimeGateway(model_port=InteractionModelPort()),
            workspace=tmp_path,
            session_open_options={"model": unit_model(), "memory_enabled": False},
        )
        session_id = (await service.create_session())["session_id"]
        await service.submit_prompt(session_id, "需要启动脚本")
        await service.wait_for_idle(session_id)

        projection = service.projection(session_id)
        interaction = projection["pending_interaction"]
        assert projection["execution"]["status"] == "waiting_user"
        assert interaction["payload"]["prompt"] == "选择界面"
        assert interaction["payload"]["options"] == ["CLI", "Web"]

        await service.respond_interaction(session_id, interaction["request_id"], "Web")
        await service.wait_for_idle(session_id)
        assert service.projection(session_id)["execution"]["status"] == "idle"
        assert any("interaction resumed" in str(item) for item in service.timeline(session_id))
        await service.shutdown()

    asyncio.run(run_case())
