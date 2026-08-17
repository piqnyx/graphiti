"""piqnyx read-only MCP tool: one search returning facts, entities and episodes with their scores.

Upstream splits this across `search_memory_facts` and `search_memory_nodes`, and
both discard the numbers. The engine does compute them — `SearchResults` carries a
reranker score beside every list — but the MCP layer keeps only the objects, so a
caller receives a ranked order with no way to say how strong a match is, and no
way to tell a near-certain hit from the last item that scraped in.

Returning the scores is what makes a two-step search possible: the agent picks
what is worth expanding by number rather than by position, and then reads the
conversation behind it.

The RRF recipe is deliberate. The library's default combined recipe reranks with
a cross-encoder, which is a model call per search — unacceptable for something an
agent invokes freely. RRF is pure retrieval: BM25 and vector similarity fused
arithmetically, no LLM involved.

Read-only, LLM-free, scoped to one physical graph, like the other fork tools.
"""

from __future__ import annotations

from typing import Any

from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

from models.response_types import ErrorResponse
from utils.type_config import build_fact_search_filters

DEFAULT_LIMIT = 10
MAX_LIMIT = 50


def install_search_memory_combined_tool(server: Any) -> None:
    """Register search_memory_combined on the already-created FastMCP server."""

    @server.mcp.tool()
    async def search_memory_combined(
        query: str,
        group_id: str | None = None,
        limit: int = DEFAULT_LIMIT,
        valid_at_after: str | None = None,
        valid_at_before: str | None = None,
        created_at_after: str | None = None,
    ) -> dict[str, Any] | ErrorResponse:
        """Search one graph for facts, entities and episodes at once, with scores.

        Returns three ranked lists from a single retrieval pass. Facts answer what
        is known, entities answer who or what something is, and episodes are
        stretches of conversation whose own words matched — each already an anchor
        for reading the surrounding dialog.

        Args:
            query: What to look for.
            group_id: Graph to search; defaults to the configured group.
            limit: Maximum results per list.
            valid_at_after: Only facts that were true at or after this ISO-8601 time.
            valid_at_before: Only facts that were true at or before this time.
            created_at_after: Only facts recorded at or after this time — when the
                subject was discussed, as opposed to when it was true.
        """
        if server.graphiti_service is None:
            return ErrorResponse(error='Graphiti service not initialized')
        if not query or not query.strip():
            return ErrorResponse(error='query is required')

        effective_group_id = group_id or server.config.graphiti.group_id
        if not effective_group_id:
            return ErrorResponse(
                error='No group_id provided and no default group_id is configured'
            )

        bounded = max(1, min(int(limit or DEFAULT_LIMIT), MAX_LIMIT))

        try:
            search_filter = build_fact_search_filters(
                valid_at_after=valid_at_after,
                valid_at_before=valid_at_before,
            )
        except ValueError as exc:
            return ErrorResponse(error=f'Invalid date filter: {exc}')

        if created_at_after:
            # created_at is a separate axis from valid_at: one is when the fact was
            # recorded, the other when it held true. Conflating them answers the
            # wrong question, so it is built here rather than folded into the
            # helper above, which only knows about validity.
            try:
                search_filter = _with_created_after(search_filter, created_at_after)
            except ValueError as exc:
                return ErrorResponse(error=f'Invalid created_at_after: {exc}')

        try:
            client = await server.graphiti_service.get_client()
            scoped_driver = client.driver.clone(database=effective_group_id)
            # Copied rather than rebuilt: the recipe already carries a limit, and
            # unpacking it alongside a new one is a duplicate keyword.
            config = COMBINED_HYBRID_SEARCH_RRF.model_copy(deep=True, update={'limit': bounded})
            results = await client.search_(
                query=query,
                config=config,
                group_ids=[effective_group_id],
                search_filter=search_filter,
                driver=scoped_driver,
            )
        except Exception as exc:
            server.logger.error(f'Error searching memory: {exc}')
            return ErrorResponse(error=f'Error searching memory: {exc}')

        return {
            'message': f"Combined search for group '{effective_group_id}' completed",
            'group_id': effective_group_id,
            'facts': [
                {
                    'uuid': edge.uuid,
                    'fact': edge.fact,
                    'score': score,
                    'episodes': list(getattr(edge, 'episodes', []) or []),
                    # The two entities this fact connects. An entity carries no
                    # provenance of its own, so this is the only honest way to say
                    # which conversations it came up in: the episodes of the facts
                    # that touch it.
                    'source_node_uuid': getattr(edge, 'source_node_uuid', None),
                    'target_node_uuid': getattr(edge, 'target_node_uuid', None),
                    'created_at': _iso(getattr(edge, 'created_at', None)),
                    'valid_at': _iso(getattr(edge, 'valid_at', None)),
                    'invalid_at': _iso(getattr(edge, 'invalid_at', None)),
                    'expired_at': _iso(getattr(edge, 'expired_at', None)),
                }
                for edge, score in zip(results.edges, _scores(results.edge_reranker_scores, results.edges))
            ],
            'entities': [
                {
                    'uuid': node.uuid,
                    'name': node.name,
                    'score': score,
                    'summary': getattr(node, 'summary', None),
                    'created_at': _iso(getattr(node, 'created_at', None)),
                }
                for node, score in zip(results.nodes, _scores(results.node_reranker_scores, results.nodes))
            ],
            'episodes': [
                {
                    'uuid': episode.uuid,
                    'name': episode.name,
                    'score': score,
                    'created_at': _iso(getattr(episode, 'created_at', None)),
                }
                for episode, score in zip(
                    results.episodes, _scores(results.episode_reranker_scores, results.episodes)
                )
            ],
        }


def _scores(scores: list[float], items: list[Any]) -> list[float | None]:
    """Pair every item with its score, tolerating a reranker that returned none.

    Some rerankers populate the score list and some leave it empty; zipping a
    short list would silently drop results, which is the one outcome a search must
    never produce.
    """
    if len(scores) == len(items):
        return list(scores)
    return [None] * len(items)


def _iso(value: Any) -> str | None:
    return value.isoformat() if hasattr(value, 'isoformat') else None


def _with_created_after(search_filter: Any, created_at_after: str) -> Any:
    """Add a created_at lower bound, creating the filter object if needed."""
    from graphiti_core.search.search_filters import ComparisonOperator, DateFilter, SearchFilters

    from utils.type_config import parse_reference_time

    parsed = parse_reference_time(created_at_after)
    if parsed is None:
        raise ValueError(f'could not parse {created_at_after!r} as an ISO-8601 timestamp')

    bound = [[DateFilter(date=parsed, comparison_operator=ComparisonOperator.greater_than_equal)]]
    if search_filter is None:
        return SearchFilters(created_at=bound)
    search_filter.created_at = bound
    return search_filter
