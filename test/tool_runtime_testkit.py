async def execute_tool(runtime, request):
    preparation = runtime.prepare_batch((request,))
    if preparation.results:
        assert len(preparation.results) == 1
        return preparation.results[0]
    results = await runtime.execute_prepared(preparation.batch_id)
    assert len(results) == 1
    return results[0]


async def resume_tool(runtime, response):
    """测试专用：模拟 Runtime 先准备、再执行恢复响应。"""
    prepared = runtime.prepare_resume(response)
    return await runtime.execute_prepared_resume(prepared.resume_id)
