#!/usr/bin/env python3
"""A cross-encoder served over HTTP, and the switch that reaches for it."""

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graphiti_core.cross_encoder.http_reranker_client import HTTPRerankerClient


def _server(payload, seen=None):
    """An httpx client answering with one canned rerank response."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, json=payload)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_scores_are_attached_to_the_passage_the_server_indexed():
    """The response carries indexes, not text; mapping them back is the whole job."""
    passages = ['Вит живёт в Григолети', 'Эва имеет доступ к grep', 'Барбос курцхаар']
    payload = {
        'results': [
            {'index': 2, 'relevance_score': -8.3},
            {'index': 0, 'relevance_score': 5.4},
            {'index': 1, 'relevance_score': -9.9},
        ]
    }
    client = HTTPRerankerClient(url='http://x/v1/rerank', model='m', client=_server(payload))

    ranked = await client.rank('где живёт Вит', passages)

    assert ranked == [
        ('Вит живёт в Григолети', 5.4),
        ('Барбос курцхаар', -8.3),
        ('Эва имеет доступ к grep', -9.9),
    ]


@pytest.mark.asyncio
async def test_raw_scores_are_not_normalised():
    """bge returns logits, and reranker_min_score filters on >= 0.

    Squashing them into 0..1 would move the boundary between "answers the query"
    and "does not" somewhere that no longer has an obvious default.
    """
    payload = {'results': [{'index': 0, 'relevance_score': -11.03}]}
    client = HTTPRerankerClient(url='http://x/v1/rerank', model='m', client=_server(payload))

    ranked = await client.rank('вопрос', ['факт'])

    assert ranked == [('факт', -11.03)]


@pytest.mark.asyncio
async def test_an_index_the_server_invented_is_dropped():
    """Trusting it would score somebody else's passage, or crash on the boundary."""
    payload = {
        'results': [{'index': 7, 'relevance_score': 1.0}, {'index': 0, 'relevance_score': 2.0}]
    }
    client = HTTPRerankerClient(url='http://x/v1/rerank', model='m', client=_server(payload))

    ranked = await client.rank('вопрос', ['первый', 'второй'])

    assert ranked == [('первый', 2.0)]


@pytest.mark.asyncio
async def test_nothing_to_rank_asks_nothing():
    seen: list[httpx.Request] = []
    client = HTTPRerankerClient(url='http://x/v1/rerank', model='m', client=_server({}, seen))

    assert await client.rank('вопрос', []) == []
    assert seen == []


@pytest.mark.asyncio
async def test_the_query_and_every_passage_are_sent():
    seen: list[httpx.Request] = []
    payload = {
        'results': [{'index': 0, 'relevance_score': 1.0}, {'index': 1, 'relevance_score': 0.5}]
    }
    client = HTTPRerankerClient(
        url='http://x/v1/rerank', model='bge', client=_server(payload, seen)
    )

    await client.rank('где живёт Вит', ['a', 'b'])

    import json

    body = json.loads(seen[0].content)
    assert body['query'] == 'где живёт Вит'
    assert body['documents'] == ['a', 'b']
    assert body['model'] == 'bge'
    assert body['top_n'] == 2


def test_a_url_without_a_model_is_refused():
    with pytest.raises(ValueError):
        HTTPRerankerClient(url='http://x/v1/rerank', model='')


def test_an_explicit_url_is_reached_for_before_any_provider(monkeypatch):
    """The provider search would otherwise pick a paid per-passage reranker.

    With `llm.provider: openai` it returns OpenAIRerankerClient built without a
    model, which defaults to gpt-4.1-nano and spends one chat completion per
    candidate, on every turn, against whatever gateway the LLM key points at.
    """
    from unittest.mock import Mock

    from services.factories import CrossEncoderFactory

    monkeypatch.setenv('GRAPHITI_RERANKER_URL', 'http://127.0.0.1:18080/v1/rerank')
    monkeypatch.setenv('GRAPHITI_RERANKER_MODEL', 'bge-reranker-v2-m3')

    reranker = CrossEncoderFactory.create(Mock(), Mock())

    assert isinstance(reranker, HTTPRerankerClient)
    assert reranker.url == 'http://127.0.0.1:18080/v1/rerank'
    assert reranker.model == 'bge-reranker-v2-m3'


def test_a_url_without_a_model_refuses_rather_than_falling_through(monkeypatch):
    """Falling through here is the expensive silent path, so it fails loudly instead."""
    from unittest.mock import Mock

    from services.factories import CrossEncoderFactory

    monkeypatch.setenv('GRAPHITI_RERANKER_URL', 'http://127.0.0.1:18080/v1/rerank')
    monkeypatch.delenv('GRAPHITI_RERANKER_MODEL', raising=False)

    with pytest.raises(ValueError, match='GRAPHITI_RERANKER_MODEL'):
        CrossEncoderFactory.create(Mock(), Mock())
