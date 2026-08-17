from __future__ import annotations

from types import SimpleNamespace

import pytest

import piqnyx_graph_stats_tool as patch


class FakeMcp:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def register(fn):
            self.tools[fn.__name__] = fn
            return fn

        return register


class FakeDriver:
    """Answers queries by matching a distinctive fragment of their Cypher."""

    def __init__(self, answers, failing=()):
        self.answers = answers
        self.failing = failing
        self.databases = []

    def clone(self, database):
        self.databases.append(database)
        return self

    async def execute_query(self, cypher, routing_=None, **params):
        assert routing_ == 'r', 'graph stats must never take a write route'
        for fragment in self.failing:
            if fragment in cypher:
                raise RuntimeError(f'unsupported: {fragment}')
        for fragment, rows in self.answers.items():
            if fragment in cypher:
                return rows, None, None
        return [], None, None


def build_server(driver):
    return SimpleNamespace(
        mcp=FakeMcp(),
        graphiti_service=SimpleNamespace(
            get_client=lambda: _resolved(SimpleNamespace(driver=driver))
        ),
        config=SimpleNamespace(graphiti=SimpleNamespace(group_id='default')),
        logger=SimpleNamespace(error=lambda _message: None),
    )


async def _resolved(value):
    return value


@pytest.mark.asyncio
async def test_reports_size_shape_and_integrity_for_the_requested_graph():
    driver = FakeDriver(
        {
            'count(n) AS value': [{'value': 42}],
            'MATCH (e:Episodic) RETURN count(e) AS value': [{'value': 13}],
            'count(s) AS value': [{'value': 1}],
            'r:RELATES_TO]->() RETURN count(r)': [{'value': 228}],
            'r:MENTIONS]->() RETURN count(r)': [{'value': 100}],
            'ORDER BY degree DESC': [{'name': 'Вит', 'degree': 57}],
            'ORDER BY e.created_at ASC': [{'name': 'saga-1', 'created_at': '2026-08-16'}],
            'ORDER BY e.created_at DESC': [{'name': 'saga-13', 'created_at': '2026-08-17'}],
        }
    )
    server = build_server(driver)
    patch.install_get_graph_stats_tool(server)

    result = await server.mcp.tools['get_graph_stats'](group_id='8248439450')

    assert driver.databases == ['8248439450'], 'the graph must be scoped to the group'
    assert result['group_id'] == '8248439450'
    assert result['size'] == {
        'entities': 42,
        'episodes': 13,
        'sagas': 1,
        'facts': 228,
        'mentions': 100,
    }
    assert result['top_entities'] == [{'name': 'Вит', 'degree': 57}]
    assert result['newest_episode']['name'] == 'saga-13'
    assert result['integrity']['facts_without_provenance'] == 0
    assert result['query_errors'] == []


@pytest.mark.asyncio
async def test_a_failing_query_is_recorded_without_losing_the_rest_of_the_report():
    driver = FakeDriver(
        {'count(n) AS value': [{'value': 7}]},
        failing=('size(r.episodes)',),
    )
    server = build_server(driver)
    patch.install_get_graph_stats_tool(server)

    result = await server.mcp.tools['get_graph_stats']()

    assert result['size']['entities'] == 7
    assert result['integrity']['facts_without_provenance'] == 0
    assert any('facts_without_provenance' in error for error in result['query_errors'])


@pytest.mark.asyncio
async def test_top_entities_is_clamped_to_a_sane_range():
    captured = {}

    class CapturingDriver(FakeDriver):
        async def execute_query(self, cypher, routing_=None, **params):
            if 'ORDER BY degree DESC' in cypher:
                captured['limit'] = params.get('limit')
            return await super().execute_query(cypher, routing_=routing_, **params)

    server = build_server(CapturingDriver({}))
    patch.install_get_graph_stats_tool(server)

    await server.mcp.tools['get_graph_stats'](top_entities=5000)
    assert captured['limit'] == patch.MAX_TOP_ENTITIES

    # Zero reads as "unspecified" and takes the default; a negative value is
    # nonsense and is pulled up to the smallest useful list.
    await server.mcp.tools['get_graph_stats'](top_entities=0)
    assert captured['limit'] == patch.DEFAULT_TOP_ENTITIES

    await server.mcp.tools['get_graph_stats'](top_entities=-3)
    assert captured['limit'] == 1


@pytest.mark.asyncio
async def test_refuses_when_no_group_can_be_resolved():
    server = build_server(FakeDriver({}))
    server.config.graphiti.group_id = None
    patch.install_get_graph_stats_tool(server)

    result = await server.mcp.tools['get_graph_stats']()

    # ErrorResponse is a TypedDict, so the failure arrives as a plain mapping.
    assert 'no default group_id' in result['error']
