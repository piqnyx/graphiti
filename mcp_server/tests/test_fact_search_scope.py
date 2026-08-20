#!/usr/bin/env python3
"""The fact search must read the graph its group actually writes to."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

import graphiti_mcp_server


def _install_client(monkeypatch):
    """A client that records what the search was handed."""
    scoped_driver = Mock(name='scoped_driver')
    client = Mock()
    client.driver.clone = Mock(return_value=scoped_driver)
    client.search_ = AsyncMock(return_value=SimpleNamespace(edges=[], edge_reranker_scores=[]))

    service = Mock()
    service.get_client = AsyncMock(return_value=client)
    monkeypatch.setattr(graphiti_mcp_server, 'graphiti_service', service)
    monkeypatch.setattr(
        graphiti_mcp_server,
        'config',
        SimpleNamespace(graphiti=SimpleNamespace(group_id='main')),
        raising=False,
    )
    return client, scoped_driver


@pytest.mark.asyncio
async def test_a_named_group_is_searched_in_its_own_database(monkeypatch):
    """Episodes are written with driver.clone(database=group_id).

    Searching without that clone reads whichever database the shared driver was
    built with, and the group_id filter then matches nothing there -- recall for
    every agent but the configured one comes back empty, and says nothing about it.
    """
    client, scoped_driver = _install_client(monkeypatch)

    await graphiti_mcp_server.search_memory_facts(query='кукуруза', group_ids='igor')

    client.driver.clone.assert_called_once_with(database='igor')
    assert client.search_.await_args.kwargs['driver'] is scoped_driver
    assert client.search_.await_args.kwargs['group_ids'] == ['igor']


@pytest.mark.asyncio
async def test_several_groups_keep_the_shared_driver(monkeypatch):
    """A database is one graph, so a search across two cannot be scoped to either."""
    client, _ = _install_client(monkeypatch)

    await graphiti_mcp_server.search_memory_facts(query='кукуруза', group_ids=['igor', 'red'])

    client.driver.clone.assert_not_called()
    assert client.search_.await_args.kwargs['driver'] is None


@pytest.mark.asyncio
async def test_the_shared_recipe_survives_a_search(monkeypatch):
    """The recipe is a module-level object every concurrent request reads.

    Assigning the caller's limit onto it lets a manual search asking for fifty
    rewrite the limit a recall asking for eight is about to use.
    """
    client, _ = _install_client(monkeypatch)
    before = EDGE_HYBRID_SEARCH_RRF.limit

    await graphiti_mcp_server.search_memory_facts(query='кукуруза', group_ids='main', max_facts=50)

    assert EDGE_HYBRID_SEARCH_RRF.limit == before
    assert client.search_.await_args.kwargs['config'].limit == 50


@pytest.mark.asyncio
async def test_the_pool_is_searched_and_the_answer_is_sliced(monkeypatch):
    """These were one number, which is why nothing below depth sixteen was findable.

    Each search method fetches 2 * limit, so a request for eight facts examined
    sixteen candidates and no amount of asking reached past them.
    """
    client, _ = _install_client(monkeypatch)
    edges = [Mock(name=f'edge{i}') for i in range(30)]
    client.search_ = AsyncMock(
        return_value=SimpleNamespace(
            edges=edges, edge_reranker_scores=[float(i) for i in range(30)]
        )
    )
    monkeypatch.setattr(graphiti_mcp_server, 'format_fact_result', lambda edge: {'fact': str(edge)})

    result = await graphiti_mcp_server.search_memory_facts(
        query='кукуруза', group_ids='main', max_facts=8, pool=40
    )

    assert client.search_.await_args.kwargs['config'].limit == 40
    assert len(result['facts']) == 8


@pytest.mark.asyncio
async def test_the_floor_travels_with_the_request(monkeypatch):
    """Tunable per call, because a threshold nobody can sweep cannot be chosen."""
    client, _ = _install_client(monkeypatch)

    monkeypatch.setenv('GRAPHITI_RERANKER_URL', 'http://127.0.0.1:18080/v1/rerank')
    await graphiti_mcp_server.search_memory_facts(
        query='кукуруза', group_ids='main', rerank=True, min_score=0.0
    )

    config = client.search_.await_args.kwargs['config']
    assert config.reranker_min_score == 0.0
    assert config.edge_config.reranker.value == 'cross_encoder'


@pytest.mark.asyncio
async def test_an_unasked_floor_leaves_the_recipe_alone(monkeypatch):
    """Absent means "as declared", not "zero" -- the two differ for rank fusion."""
    client, _ = _install_client(monkeypatch)

    await graphiti_mcp_server.search_memory_facts(query='кукуруза', group_ids='main')

    config = client.search_.await_args.kwargs['config']
    assert config.reranker_min_score == EDGE_HYBRID_SEARCH_RRF.reranker_min_score
    assert config.edge_config.reranker == EDGE_HYBRID_SEARCH_RRF.edge_config.reranker


@pytest.mark.asyncio
async def test_each_fact_carries_the_score_that_admitted_it(monkeypatch):
    """Without them, "nothing cleared the floor" and "the search found nothing" look alike."""
    client, _ = _install_client(monkeypatch)
    client.search_ = AsyncMock(
        return_value=SimpleNamespace(edges=[Mock(), Mock()], edge_reranker_scores=[5.4, -8.3])
    )
    monkeypatch.setattr(graphiti_mcp_server, 'format_fact_result', lambda edge: {'fact': 'x'})

    result = await graphiti_mcp_server.search_memory_facts(query='кукуруза', group_ids='main')

    assert [fact['score'] for fact in result['facts']] == [5.4, -8.3]


@pytest.mark.asyncio
async def test_reranking_without_a_configured_cross_encoder_is_refused(monkeypatch):
    """The factory's fallback is a paid per-passage reranker against the LLM gateway.

    Serving that silently would put a bill on every turn behind a parameter that
    reads as a ranking choice.
    """
    client, _ = _install_client(monkeypatch)
    monkeypatch.delenv('GRAPHITI_RERANKER_URL', raising=False)

    result = await graphiti_mcp_server.search_memory_facts(
        query='кукуруза', group_ids='main', rerank=True
    )

    assert 'GRAPHITI_RERANKER_URL' in result['error']
    client.search_.assert_not_awaited()
