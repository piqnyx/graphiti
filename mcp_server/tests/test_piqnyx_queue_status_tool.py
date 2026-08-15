from __future__ import annotations

from types import SimpleNamespace

import pytest

import piqnyx_queue_status_tool as patch


class FakeMcp:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def register(fn):
            self.tools[fn.__name__] = fn
            return fn

        return register


@pytest.mark.asyncio
async def test_get_queue_status_returns_blocked_episode_identity():
    class FakeQueue:
        def get_failure_status(self, group_id):
            assert group_id == 'main'
            return {
                'group_id': 'main',
                'blocked': True,
                'attempts': 5,
                'last_error': 'llm timeout',
                'pending': 3,
                'episode_uuid': 'episode-7',
                'episode_name': '6bc2a77c6957-7',
                'saga': 'session-1',
            }

    server = SimpleNamespace(
        mcp=FakeMcp(),
        queue_service=FakeQueue(),
        logger=SimpleNamespace(error=lambda _message: None),
    )

    patch.install_get_queue_status_tool(server)
    result = await server.mcp.tools['get_queue_status']('main')

    assert result['blocked'] is True
    assert result['attempts'] == 5
    assert result['episode_uuid'] == 'episode-7'
    assert result['saga'] == 'session-1'
    assert 'content' not in result


@pytest.mark.asyncio
async def test_get_queue_status_is_healthy_when_not_blocked():
    class FakeQueue:
        def get_failure_status(self, group_id):
            return {
                'group_id': group_id,
                'blocked': False,
                'attempts': 0,
                'last_error': None,
                'pending': 0,
                'episode_uuid': None,
                'episode_name': None,
                'saga': None,
            }

    server = SimpleNamespace(
        mcp=FakeMcp(),
        queue_service=FakeQueue(),
        logger=SimpleNamespace(error=lambda _message: None),
    )

    patch.install_get_queue_status_tool(server)
    result = await server.mcp.tools['get_queue_status']('igor')

    assert result['group_id'] == 'igor'
    assert result['blocked'] is False
    assert result['last_error'] is None
