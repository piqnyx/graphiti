from __future__ import annotations

from types import SimpleNamespace

import pytest

import piqnyx_community_tool as patch


class FakeMcp:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def register(fn):
            self.tools[fn.__name__] = fn
            return fn

        return register


class FakeDriver:
    def __init__(self, database='default_db'):
        self.database = database

    def clone(self, database):
        return FakeDriver(database)


class FakeClient:
    def __init__(self):
        self.driver = FakeDriver()
        self.calls = []

    async def build_communities(self, group_ids=None, driver=None):
        self.calls.append({'group_ids': group_ids, 'driver': driver})
        return ([SimpleNamespace(name='Дом и переезды')], [object()])


def build_server(client):
    return SimpleNamespace(
        mcp=FakeMcp(),
        graphiti_service=SimpleNamespace(get_client=lambda: _resolved(client)),
        logger=SimpleNamespace(error=lambda _message: None),
    )


async def _resolved(value):
    return value


@pytest.mark.asyncio
async def test_communities_are_built_against_the_agents_own_graph():
    client = FakeClient()
    server = build_server(client)
    patch.install_build_communities_for_group_tool(server)

    result = await server.mcp.tools['build_communities_for_group']('igor')

    # The whole point: upstream omits the driver and the core then falls back to
    # the default database, which on this deployment holds no agent's data.
    call = client.calls[0]
    assert call['driver'].database == 'igor'
    assert call['group_ids'] == ['igor']
    assert result['communities'] == 1
    assert result['group_id'] == 'igor'


@pytest.mark.asyncio
async def test_a_missing_group_is_refused_rather_than_defaulted():
    client = FakeClient()
    server = build_server(client)
    patch.install_build_communities_for_group_tool(server)

    result = await server.mcp.tools['build_communities_for_group']('   ')

    assert 'group_id is required' in result['error']
    assert client.calls == []
