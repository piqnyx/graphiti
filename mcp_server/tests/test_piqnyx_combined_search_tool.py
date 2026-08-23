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
    """A driver that can answer the provenance query.

    Without execute_query the entity-provenance lookup raised, the raise was
    swallowed, and the swallow then failed on a logger with no warning() -- so a
    test about scores died two levels away from anything it was testing.
    """

    def __init__(self, database='default_db', rows=None):
        self.database = database
        self.rows = list(rows or [])
        self.queries = []

    def clone(self, database):
        clone = FakeDriver(database, self.rows)
        clone.queries = self.queries
        return clone

    async def execute_query(self, query, **kwargs):
        self.queries.append({'query': query, **kwargs})
        return list(self.rows), None, None


class FakeCrossEncoder:
    """Scores passages from a table, and answers the way a real one does.

    The interface returns (passage, score) sorted by score rather than in the
    order it was asked, which is why the tool matches items back by their text;
    a fake that echoed the input order would hide a bug in exactly that step.
    """

    def __init__(self, scores: dict[str, float], fail: bool = False):
        self.scores = scores
        self.fail = fail
        self.calls = []

    async def rank(self, query, passages):
        self.calls.append({'query': query, 'passages': list(passages)})
        if self.fail:
            raise RuntimeError('reranker is down')
        ranked = [(passage, self.scores.get(passage, 0.0)) for passage in passages]
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked


def edge(uuid, fact, episodes, invalid_at=None, source='n1', target='n2'):
    return SimpleNamespace(
        uuid=uuid, fact=fact, episodes=episodes,
        source_node_uuid=source, target_node_uuid=target,
        created_at=datetime(2026, 8, 17, tzinfo=timezone.utc),
        valid_at=None, invalid_at=invalid_at, expired_at=None,
    )


class FakeClient:
    def __init__(self, results, cross_encoder=None, rows=None):
        self.driver = FakeDriver(rows=rows)
        self.results = results
        self.calls = []
        if cross_encoder is not None:
            self.cross_encoder = cross_encoder

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
        logger=SimpleNamespace(error=lambda _m: None, warning=lambda _m: None),
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
async def test_retrieval_never_reranks_with_a_model():
    client = FakeClient(results())
    server = build_server(client)
    patch.install_search_memory_combined_tool(server)

    await server.mcp.tools['search_memory_combined']('x', group_id='main', limit=999)

    config = client.calls[0]['config']
    # Retrieval stays arithmetic: RRF fuses BM25 and vector hits with no model in
    # it. The model call happens afterwards, over what this returned, so a wide
    # pool costs one rank call rather than one per candidate.
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


@pytest.mark.asyncio
async def test_a_hit_below_the_floor_is_not_returned():
    # The whole reason the tool ranks at all: a query about something the graph
    # has never heard of used to come back with a confident-looking ten, because
    # a fusion score is the position restated and every list has a first item.
    client = FakeClient(
        results(edges=[edge('e1', 'близко', []), edge('e2', 'мимо', [])]),
        cross_encoder=FakeCrossEncoder({'близко': 0.42, 'мимо': 0.03}),
    )
    server = build_server(client)
    patch.install_search_memory_combined_tool(server)

    out = await server.mcp.tools['search_memory_combined']('вопрос', group_id='main')

    assert [f['fact'] for f in out['facts']] == ['близко']
    assert out['facts'][0]['score'] == 0.42
    assert out['ranked_by'] == 'relevance'
    assert out['min_score'] == patch.DEFAULT_MIN_SCORE


@pytest.mark.asyncio
async def test_nothing_above_the_floor_answers_with_nothing():
    # An empty list is the answer, and the message has to say so: ten unrelated
    # facts are worse than none, because the caller cannot tell they are unrelated.
    client = FakeClient(
        results(edges=[edge('e1', 'мимо', []), edge('e2', 'тоже мимо', [])]),
        cross_encoder=FakeCrossEncoder({'мимо': 0.04, 'тоже мимо': 0.02}),
    )
    server = build_server(client)
    patch.install_search_memory_combined_tool(server)

    out = await server.mcp.tools['search_memory_combined']('Брендон', group_id='main')

    assert out['facts'] == [] and out['entities'] == [] and out['episodes'] == []
    assert 'Nothing' in out['message']
    assert out['ranked_by'] == 'relevance'


@pytest.mark.asyncio
async def test_the_caller_can_lower_the_floor():
    client = FakeClient(
        results(edges=[edge('e1', 'слабое совпадение', [])]),
        cross_encoder=FakeCrossEncoder({'слабое совпадение': 0.05}),
    )
    server = build_server(client)
    patch.install_search_memory_combined_tool(server)

    strict = await server.mcp.tools['search_memory_combined']('q', group_id='main')
    broad = await server.mcp.tools['search_memory_combined']('q', group_id='main', min_score=0.0)

    assert strict['facts'] == []
    assert [f['fact'] for f in broad['facts']] == ['слабое совпадение']
    assert broad['min_score'] == 0.0


@pytest.mark.asyncio
async def test_ranking_orders_by_score_not_by_retrieval_position():
    # The ranker answers sorted by score rather than in the order it was asked,
    # so items are matched back by their passage text. Retrieval order here is
    # deliberately the reverse of the ranking.
    client = FakeClient(
        results(edges=[edge('e1', 'третий', []), edge('e2', 'первый', []), edge('e3', 'второй', [])]),
        cross_encoder=FakeCrossEncoder({'первый': 0.9, 'второй': 0.5, 'третий': 0.2}),
    )
    server = build_server(client)
    patch.install_search_memory_combined_tool(server)

    out = await server.mcp.tools['search_memory_combined']('q', group_id='main')

    assert [f['fact'] for f in out['facts']] == ['первый', 'второй', 'третий']
    assert [f['score'] for f in out['facts']] == [0.9, 0.5, 0.2]


@pytest.mark.asyncio
async def test_without_a_reranker_the_scores_are_fusion_and_say_so():
    # No cross-encoder configured: the tool still answers, with retrieval order,
    # and names the scale so nobody reads a position as relevance.
    client = FakeClient(results(edges=[edge('e1', 'что-то', [])], edge_scores=[1.0]))
    server = build_server(client)
    patch.install_search_memory_combined_tool(server)

    out = await server.mcp.tools['search_memory_combined']('q', group_id='main')

    assert out['ranked_by'] == 'fusion'
    assert out['min_score'] is None
    assert [f['fact'] for f in out['facts']] == ['что-то']


@pytest.mark.asyncio
async def test_a_reranker_that_is_down_falls_back_instead_of_failing():
    # A search that worked must not become an error because the ranking step did.
    client = FakeClient(
        results(edges=[edge('e1', 'что-то', [])], edge_scores=[1.0]),
        cross_encoder=FakeCrossEncoder({}, fail=True),
    )
    server = build_server(client)
    patch.install_search_memory_combined_tool(server)

    out = await server.mcp.tools['search_memory_combined']('q', group_id='main')

    assert out['ranked_by'] == 'fusion'
    assert [f['fact'] for f in out['facts']] == ['что-то']


@pytest.mark.asyncio
async def test_rerank_off_returns_retrieval_order_without_calling_the_ranker():
    ranker = FakeCrossEncoder({'что-то': 0.9})
    client = FakeClient(results(edges=[edge('e1', 'что-то', [])], edge_scores=[1.0]), cross_encoder=ranker)
    server = build_server(client)
    patch.install_search_memory_combined_tool(server)

    out = await server.mcp.tools['search_memory_combined']('q', group_id='main', rerank=False)

    assert out['ranked_by'] == 'fusion'
    assert ranker.calls == []


@pytest.mark.asyncio
async def test_a_long_passage_is_cut_before_it_reaches_the_ranker():
    # A reranker is given a pair, not a document. One whole episode has already
    # returned HTTP 500 from the local server, so the cut is not a nicety.
    long_fact = 'я' * (patch.PASSAGE_MAX_CHARS + 500)
    ranker = FakeCrossEncoder({long_fact[: patch.PASSAGE_MAX_CHARS]: 0.7})
    client = FakeClient(results(edges=[edge('e1', long_fact, [])]), cross_encoder=ranker)
    server = build_server(client)
    patch.install_search_memory_combined_tool(server)

    out = await server.mcp.tools['search_memory_combined']('q', group_id='main')

    assert len(ranker.calls[0]['passages'][0]) == patch.PASSAGE_MAX_CHARS
    # The fact itself is returned whole; only what the ranker judged was cut.
    assert out['facts'][0]['fact'] == long_fact


@pytest.mark.asyncio
async def test_retrieval_is_asked_for_the_pool_and_the_caller_gets_the_limit():
    # The ranker can only choose among what retrieval returned, so a pool equal
    # to the limit would make the ranking cosmetic: it would reorder exactly the
    # items that were going to be returned anyway.
    facts = [edge(f'e{i}', f'факт {i}', []) for i in range(6)]
    ranker = FakeCrossEncoder({f'факт {i}': 0.9 - i / 100 for i in range(6)})
    client = FakeClient(results(edges=facts), cross_encoder=ranker)
    server = build_server(client)
    patch.install_search_memory_combined_tool(server)

    out = await server.mcp.tools['search_memory_combined']('q', group_id='main', limit=2, pool=40)

    assert client.calls[0]['config'].limit == 40
    assert len(ranker.calls[0]['passages']) == 6
    assert [f['fact'] for f in out['facts']] == ['факт 0', 'факт 1']


@pytest.mark.asyncio
async def test_an_entity_is_judged_by_its_summary_not_by_its_name_alone():
    # A bare name is thin to rank; the summary is what says which thing it is.
    node = SimpleNamespace(uuid='n1', name='Байт', summary='огромный рыжий кот',
                           created_at=datetime(2026, 8, 16, tzinfo=timezone.utc))
    ranker = FakeCrossEncoder({'Байт — огромный рыжий кот': 0.6})
    client = FakeClient(results(nodes=[node]), cross_encoder=ranker,
                        rows=[{'entity_uuid': 'n1', 'episode_uuid': 'ep-3'}])
    server = build_server(client)
    patch.install_search_memory_combined_tool(server)

    out = await server.mcp.tools['search_memory_combined']('кот', group_id='main')

    assert ranker.calls[0]['passages'] == ['Байт — огромный рыжий кот']
    assert out['entities'][0]['score'] == 0.6
    # Provenance still travels: an entity carries the episodes that mention it.
    assert out['entities'][0]['episodes'] == ['ep-3']


@pytest.mark.asyncio
async def test_the_pool_is_not_starved_by_the_library_floor():
    # Two floors, and only one is a judgement. This one decides what leaves the
    # database; the library sets it to 0.6 and a pool of forty then came back with
    # two, which is what made recall lower it. Ranking two out of forty is not
    # ranking, it is agreeing with retrieval.
    client = FakeClient(results(), cross_encoder=FakeCrossEncoder({}))
    server = build_server(client)
    patch.install_search_memory_combined_tool(server)

    out = await server.mcp.tools['search_memory_combined']('q', group_id='main')

    config = client.calls[0]['config']
    assert config.edge_config.sim_min_score == patch.DEFAULT_VECTOR_MIN_SCORE
    assert config.node_config.sim_min_score == patch.DEFAULT_VECTOR_MIN_SCORE
    assert out['vector_min_score'] == patch.DEFAULT_VECTOR_MIN_SCORE


@pytest.mark.asyncio
async def test_the_caller_can_set_what_leaves_the_database():
    client = FakeClient(results(), cross_encoder=FakeCrossEncoder({}))
    server = build_server(client)
    patch.install_search_memory_combined_tool(server)

    await server.mcp.tools['search_memory_combined']('q', group_id='main', vector_min_score=0.8)

    assert client.calls[0]['config'].edge_config.sim_min_score == 0.8
