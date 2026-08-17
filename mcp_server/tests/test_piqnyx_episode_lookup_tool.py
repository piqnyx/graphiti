from __future__ import annotations

from types import SimpleNamespace

import pytest

import piqnyx_episode_lookup_tool as patch


class FakeMcp:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def register(fn):
            self.tools[fn.__name__] = fn
            return fn

        return register


class FakeDriver:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.databases = []
        self.params = {}

    def clone(self, database):
        self.databases.append(database)
        return self

    async def execute_query(self, cypher, routing_=None, **params):
        assert routing_ == 'r', 'episode lookup must never take a write route'
        self.params = params
        return self.rows, None, None


async def _resolved(value):
    return value


def build_server(driver):
    return SimpleNamespace(
        mcp=FakeMcp(),
        graphiti_service=SimpleNamespace(
            get_client=lambda: _resolved(SimpleNamespace(driver=driver))
        ),
        config=SimpleNamespace(graphiti=SimpleNamespace(group_id='default')),
        logger=SimpleNamespace(error=lambda _message: None),
    )


@pytest.mark.asyncio
async def test_fetches_episodes_with_their_text_scoped_to_the_group():
    driver = FakeDriver(
        [{'uuid': 'u-1', 'name': '8248439450-12', 'content': 'Вит: привет', 'created_at': 'now'}]
    )
    server = build_server(driver)
    patch.install_get_episodes_by_ref_tool(server)

    result = await server.mcp.tools['get_episodes_by_ref'](
        uuids=['u-1'], names=['8248439450-11'], group_id='8248439450'
    )

    assert driver.databases == ['8248439450']
    assert driver.params['uuids'] == ['u-1']
    assert driver.params['names'] == ['8248439450-11']
    assert result['episodes'][0]['content'] == 'Вит: привет'


@pytest.mark.asyncio
async def test_refuses_a_lookup_with_nothing_to_look_up():
    server = build_server(FakeDriver())
    patch.install_get_episodes_by_ref_tool(server)

    result = await server.mcp.tools['get_episodes_by_ref'](uuids=[], names=['   '])

    assert 'at least one uuid or name' in result['error']


@pytest.mark.asyncio
async def test_caps_the_number_of_references_it_will_expand():
    driver = FakeDriver()
    server = build_server(driver)
    patch.install_get_episodes_by_ref_tool(server)

    await server.mcp.tools['get_episodes_by_ref'](
        uuids=[f'u-{index}' for index in range(patch.MAX_EPISODES + 25)]
    )

    assert len(driver.params['uuids']) == patch.MAX_EPISODES
    assert driver.params['limit'] == patch.MAX_EPISODES
