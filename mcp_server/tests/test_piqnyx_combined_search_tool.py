from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import piqnyx_combined_search_tool as patch


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


def edge(uuid, fact, episodes, invalid_at=None, source='n1', target='n2'):
    return SimpleNamespace(
        uuid=uuid, fact=fact, episodes=episodes,
        source_node_uuid=source, target_node_uuid=target,
        created_at=datetime(2026, 8, 17, tzinfo=timezone.utc),
        valid_at=None, invalid_at=invalid_at, expired_at=None,
    )


class FakeClient:
    def __init__(self, results):
        self.driver = FakeDriver()
        self.results = results
        self.calls = []

    async def search_(self, query, config, group_ids, search_filter, driver):
        self.calls.append({
            'query': query, 'config': config, 'group_ids': group_ids,
            'search_filter': search_filter, 'driver': driver,
        })
        return self.results


def build_server(client):
    async def get_client():
        return client

    return SimpleNamespace(
        mcp=FakeMcp(),
        graphiti_service=SimpleNamespace(get_client=get_client),
        config=SimpleNamespace(graphiti=SimpleNamespace(group_id='default_db')),
        logger=SimpleNamespace(error=lambda _m: None),
    )


def results(edges=(), edge_scores=(), nodes=(), node_scores=(), episodes=(), episode_scores=()):
    return SimpleNamespace(
        edges=list(edges), edge_reranker_scores=list(edge_scores),
        nodes=list(nodes), node_reranker_scores=list(node_scores),
        episodes=list(episodes), episode_reranker_scores=list(episode_scores),
    )


@pytest.mark.asyncio
async def test_scores_reach_the_caller_and_the_search_is_scoped():
    client = FakeClient(results(
        edges=[edge('e1', 'Вит любит манго', ['ep-1', 'ep-9'])],
        edge_scores=[0.54],
        nodes=[SimpleNamespace(uuid='n1', name='Оля', summary='бывшая жена',
                               created_at=datetime(2026, 8, 16, tzinfo=timezone.utc))],
        node_scores=[0.48],
        episodes=[SimpleNamespace(uuid='ep-9', name='8248439450-9',
                                  created_at=datetime(2026, 8, 15, tzinfo=timezone.utc))],
        episode_scores=[0.37],
    ))
    server = build_server(client)
    patch.install_search_memory_combined_tool(server)

    out = await server.mcp.tools['search_memory_combined']('манго', group_id='main')

    # The whole point of the tool: the numbers the engine computed survive.
    assert out['facts'][0]['score'] == 0.54
    assert out['facts'][0]['episodes'] == ['ep-1', 'ep-9']
    assert out['entities'][0]['score'] == 0.48
    # An entity has no provenance of its own; the facts touching it name the
    # conversations it came up in, so their endpoints must survive the trip.
    assert out['facts'][0]['source_node_uuid'] == 'n1'
    assert out['episodes'][0]['name'] == '8248439450-9'
    # Isolation: the search runs against the agent's own physical graph.
    assert client.calls[0]['driver'].database == 'main'
    assert client.calls[0]['group_ids'] == ['main']


@pytest.mark.asyncio
async def test_reranking_without_scores_still_returns_every_result():
    # A reranker that populates no scores must not cost us results: zipping a
    # short list would drop them silently, which is the one thing a search
    # must never do.
    client = FakeClient(results(
        edges=[edge('e1', 'первый', []), edge('e2', 'второй', [])],
        edge_scores=[],
    ))
    server = build_server(client)
    patch.install_search_memory_combined_tool(server)

    out = await server.mcp.tools['search_memory_combined']('что угодно', group_id='main')

    assert [f['fact'] for f in out['facts']] == ['первый', 'второй']
    assert [f['score'] for f in out['facts']] == [None, None]


@pytest.mark.asyncio
async def test_the_recipe_never_reranks_with_a_model():
    client = FakeClient(results())
    server = build_server(client)
    patch.install_search_memory_combined_tool(server)

    await server.mcp.tools['search_memory_combined']('x', group_id='main', limit=999)

    config = client.calls[0]['config']
    # Cross-encoder reranking is a model call per search, which an agent-invoked
    # tool cannot afford; RRF fuses BM25 and vector hits arithmetically.
    assert 'cross_encoder' not in str(config.edge_config.reranker)
    assert config.limit == patch.MAX_LIMIT


@pytest.mark.asyncio
async def test_discussed_since_filters_on_creation_not_validity():
    client = FakeClient(results())
    server = build_server(client)
    patch.install_search_memory_combined_tool(server)

    await server.mcp.tools['search_memory_combined'](
        'x', group_id='main', created_at_after='2026-08-10T00:00:00Z'
    )

    search_filter = client.calls[0]['search_filter']
    assert search_filter.created_at is not None
    assert search_filter.valid_at is None
