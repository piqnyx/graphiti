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


@pytest.mark.asyncio
async def test_focus_ranks_by_the_remark_and_retrieves_by_the_context(monkeypatch):
    """One text finds candidates, another chooses among them.

    Measured: a cross-encoder handed 1237 characters of transcript scored every
    fact mentioned in it around 0.5 and could separate none of them; the same
    reranker given the closing question alone scored everything below 0.07 --
    correctly, because the graph held no answer to that question.
    """
    client, _ = _install_client(monkeypatch)
    client.embedder.create = AsyncMock(return_value=[0.1, 0.2, 0.3])
    monkeypatch.setenv('GRAPHITI_RERANKER_URL', 'http://127.0.0.1:18080/v1/rerank')

    await graphiti_mcp_server.search_memory_facts(
        query='длинная стенограмма разговора',
        group_ids='main',
        rerank=True,
        focus='а если бы нам надо было работать с этим репозиторием?',
    )

    kwargs = client.search_.await_args.kwargs
    assert kwargs['query'] == 'а если бы нам надо было работать с этим репозиторием?'
    assert kwargs['query_vector'] == [0.1, 0.2, 0.3]
    client.embedder.create.assert_awaited_once_with(input_data=['длинная стенограмма разговора'])


@pytest.mark.asyncio
async def test_without_focus_nothing_is_embedded_twice(monkeypatch):
    """The search embeds the query itself; doing it here as well would pay twice."""
    client, _ = _install_client(monkeypatch)
    client.embedder.create = AsyncMock()

    await graphiti_mcp_server.search_memory_facts(query='кукуруза', group_ids='main')

    assert client.search_.await_args.kwargs['query_vector'] is None
    client.embedder.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_focus_without_rerank_is_ignored(monkeypatch):
    """Rank fusion has nothing to rank by, so a focus would only misdirect retrieval."""
    client, _ = _install_client(monkeypatch)
    client.embedder.create = AsyncMock()

    await graphiti_mcp_server.search_memory_facts(
        query='длинная стенограмма', group_ids='main', focus='короткий вопрос'
    )

    assert client.search_.await_args.kwargs['query'] == 'длинная стенограмма'
    assert client.search_.await_args.kwargs['query_vector'] is None


class _Edge:
    def __init__(self, fact):
        self.fact = fact


class _Ranker:
    """A cross-encoder with an opinion, recorded per query."""

    def __init__(self, opinions):
        self.opinions = opinions
        self.asked: list[str] = []

    async def rank(self, query, passages):
        self.asked.append(query)
        scores = self.opinions[query]
        return sorted(
            ((p, scores.get(p, 0.0)) for p in passages), key=lambda pair: pair[1], reverse=True
        )


@pytest.mark.asyncio
async def test_the_conversation_rescues_what_the_remark_cannot_judge():
    """ "далеко это от меня вообще?" carries no word to rank by.

    Measured: scoring by that remark alone spread every fact across a single
    hundredth. The conversation knows that "это" was Poti.
    """
    edges = [_Edge('Вит живёт рядом с городом Поти'), _Edge('Вит ел пиде в Terminal Pide.')]
    ranker = _Ranker(
        {
            'далеко это?': {
                'Вит живёт рядом с городом Поти': 0.03,
                'Вит ел пиде в Terminal Pide.': 0.02,
            },
            'разговор про банк в Поти': {
                'Вит живёт рядом с городом Поти': 0.44,
                'Вит ел пиде в Terminal Pide.': 0.01,
            },
        }
    )

    kept, scores = await graphiti_mcp_server._rank_candidates(
        ranker, edges, 'далеко это?', 'разговор про банк в Поти', 0.5, 0.08, 6
    )

    assert [edge.fact for edge in kept] == ['Вит живёт рядом с городом Поти']
    assert scores[0] == pytest.approx(0.22)


@pytest.mark.asyncio
async def test_the_conversation_never_outranks_what_the_remark_recognised():
    """Discounted, not averaged: talk a fact has nothing to do with must not sink it."""
    edges = [_Edge('диски Samsung 990'), _Edge('Эва имеет доступ к strace')]
    ranker = _Ranker(
        {
            'что там за диски': {'диски Samsung 990': 0.15, 'Эва имеет доступ к strace': 0.01},
            'долгий разговор про песочницу': {
                'диски Samsung 990': 0.0,
                'Эва имеет доступ к strace': 0.9,
            },
        }
    )

    kept, _ = await graphiti_mcp_server._rank_candidates(
        ranker, edges, 'что там за диски', 'долгий разговор про песочницу', 0.1, None, 6
    )

    assert kept[0].fact == 'диски Samsung 990'


@pytest.mark.asyncio
async def test_a_zero_weight_asks_the_conversation_nothing():
    """One ranking pass costs one request; the second is bought, not assumed."""
    edges = [_Edge('диски Samsung 990')]
    ranker = _Ranker({'что там за диски': {'диски Samsung 990': 0.15}})

    await graphiti_mcp_server._rank_candidates(
        ranker, edges, 'что там за диски', 'контекст', 0.0, None, 6
    )

    assert ranker.asked == ['что там за диски']


@pytest.mark.asyncio
async def test_everything_below_the_floor_leaves_nothing_to_say():
    """The graph holding no answer is an answer, and the only one worth giving."""
    edges = [_Edge('Вит ел пиде'), _Edge('Эва отредактировала USER.md')]
    ranker = _Ranker(
        {'какая у меня термопаста': {'Вит ел пиде': 0.02, 'Эва отредактировала USER.md': 0.026}}
    )

    kept, scores = await graphiti_mcp_server._rank_candidates(
        ranker, edges, 'какая у меня термопаста', 'контекст', 0.0, 0.08, 6
    )

    assert kept == [] and scores == []
