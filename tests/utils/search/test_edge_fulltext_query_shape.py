"""Query-shape regression tests for edge_fulltext_search on FalkorDB (no live database).

The generic path in ``graphiti_core/search/search_utils.py`` is the one that runs for
FalkorDB (``driver.search_interface`` is never assigned). Its fulltext hits already are
the RELATES_TO relationships; re-matching every hit by uuid
(``MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]->(m:Entity)``) makes FalkorDB scan
every Entity node per hit, so a broad query costs hits x entities. On a 1373-fact graph
that took 20-70 s and hit the 30 s query timeout; consuming the hits directly returned
the identical result in under a second. Upstream PR #1500 makes the same rewrite.

The old re-MATCH also kept only edges between two Entity nodes, so the rewrite keeps
that as an explicit label guard.
"""

from typing import Any

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.falkordb.fulltext import build_falkor_fulltext_query
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import edge_fulltext_search


class RecordingFalkorDriver:
    """Captures the Cypher and params a search function emits, returning no rows."""

    provider = GraphProvider.FALKORDB
    search_interface = None

    def __init__(self):
        self.cypher_query = ''
        self.params: dict[str, Any] = {}

    def build_fulltext_query(
        self, query: str, group_ids: list[str] | None = None, max_query_length: int = 128
    ) -> str:
        return build_falkor_fulltext_query(query, group_ids, max_query_length)

    async def execute_query(self, cypher_query_: str, **kwargs: Any):
        self.cypher_query = cypher_query_
        self.params = kwargs
        return [], None, None


@pytest.mark.asyncio
async def test_falkordb_edge_fulltext_search_consumes_hits_directly():
    driver = RecordingFalkorDriver()

    await edge_fulltext_search(
        driver,  # type: ignore[arg-type]
        'api test system',
        SearchFilters(),
        group_ids=['group-a'],
    )

    assert 'YIELD relationship AS e, score' in driver.cypher_query
    assert 'WITH e, score, startNode(e) AS n, endNode(e) AS m' in driver.cypher_query
    # The per-hit re-MATCH by uuid is the hits x entities scan.
    assert 'uuid: rel.uuid' not in driver.cypher_query
    assert 'WHERE n:Entity AND m:Entity AND e.group_id IN $group_ids' in driver.cypher_query


@pytest.mark.asyncio
async def test_falkordb_edge_fulltext_search_keeps_entity_guard_without_other_filters():
    driver = RecordingFalkorDriver()

    await edge_fulltext_search(
        driver,  # type: ignore[arg-type]
        'api test system',
        SearchFilters(),
        group_ids=None,
    )

    assert 'WHERE n:Entity AND m:Entity' in driver.cypher_query
