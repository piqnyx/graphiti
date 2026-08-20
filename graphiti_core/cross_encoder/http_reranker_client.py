"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from typing import Any

import httpx

from graphiti_core.cross_encoder.client import CrossEncoderClient

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 30.0


class HTTPRerankerClient(CrossEncoderClient):
    """A cross-encoder served over HTTP, in the shape llama.cpp and friends speak.

    The alternative in this package loads BAAI/bge-reranker-v2-m3 into the process
    through sentence-transformers. That is the right answer when nothing is running
    yet, and the wrong one when the same model is already served on the host: a
    second copy costs its own memory and its own startup, to answer the same
    question the first copy is idle for.

    Scores are passed through exactly as the server reported them, and the scale
    is the server's own. A bge reranker returns raw logits -- roughly +5 for a
    passage that answers the query and -10 for one that does not -- so
    ``reranker_min_score`` at its default of 0 already means "keep what is
    relevant". A hosted reranker in the Cohere shape returns 0..1 instead, where 0
    keeps everything and the useful floor has to be measured. Normalising one into
    the other would hide which of those two a deployment is talking to.
    """

    def __init__(
        self,
        url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        client: httpx.AsyncClient | None = None,
    ):
        if not url:
            raise ValueError('HTTPRerankerClient needs a url')
        if not model:
            raise ValueError('HTTPRerankerClient needs a model name')
        self.url = url
        self.model = model
        self.api_key = api_key or None
        self.timeout_s = timeout_s
        self._client = client

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        if not passages:
            return []

        payload: dict[str, Any] = {
            'model': self.model,
            'query': query,
            'documents': passages,
            'top_n': len(passages),
        }

        # Absent for a local server, required by a hosted one. Sent only when
        # there is one, so a llama.cpp on the loopback is not handed a bearer
        # token it never asked for.
        headers = {'Authorization': f'Bearer {self.api_key}'} if self.api_key else None

        client = self._client or httpx.AsyncClient(timeout=self.timeout_s)
        try:
            response = await client.post(self.url, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
        finally:
            if self._client is None:
                await client.aclose()

        rows = body.get('results')
        if not isinstance(rows, list):
            raise ValueError(f'reranker at {self.url} returned no results array')

        ranked: list[tuple[str, float]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            index = row.get('index')
            score = row.get('relevance_score', row.get('score'))
            # An index the server invented would silently rank somebody else's
            # passage, so it is dropped rather than guessed at.
            if not isinstance(index, int) or not 0 <= index < len(passages):
                logger.warning('reranker returned index %r outside 0..%d', index, len(passages) - 1)
                continue
            if not isinstance(score, (int, float)):
                logger.warning('reranker returned no score for index %d', index)
                continue
            ranked.append((passages[index], float(score)))

        # The interface promises every passage back, in descending order. A server
        # that honours top_n returns them all already; one that quietly caps the
        # response would otherwise drop passages the caller still has to account for.
        if len(ranked) != len(passages):
            logger.warning(
                'reranker scored %d of %d passages; the rest are treated as unranked',
                len(ranked),
                len(passages),
            )

        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked
