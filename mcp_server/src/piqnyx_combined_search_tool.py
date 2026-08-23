"""piqnyx read-only MCP tool: one search returning facts, entities and episodes with their scores.

Upstream splits this across `search_memory_facts` and `search_memory_nodes`, and
both discard the numbers. The engine does compute them — `SearchResults` carries a
reranker score beside every list — but the MCP layer keeps only the objects, so a
caller receives a ranked order with no way to say how strong a match is, and no
way to tell a near-certain hit from the last item that scraped in.

Returning the scores is what makes a two-step search possible: the agent picks
what is worth expanding by number rather than by position, and then reads the
conversation behind it.

That promise needs a number that means relevance, and RRF cannot carry one. Its
score is `1 / (rank + 1)` — the position restated — so the top hit reads 1.00
whether it answers the question or merely came first, and a query about something
the graph has never heard of still returns a confident-looking ten.

So retrieval stays RRF, which is arithmetic over BM25 and vector similarity with
no model in it, and a cross-encoder ranks what retrieval found. The reranker was
avoided here because it is a paid call per search; that trade was backwards.
Recall pays it on every single turn — 66 times in one measured session — while
search is invoked a few times a day. The frequent path had already bought the
expensive answer and the rare one was economising.

A floor then decides what is worth saying at all, on the same scale recall uses,
so a threshold learned there transfers here unchanged. Below it the search returns
nothing and says so, which is the answer: memory has no match, go look elsewhere.
Ten unrelated facts are worse than an empty list, because the caller has no way to
know they are unrelated.

The episode list is empty on this backend and that is structural, not a defect
here: `EpisodeSearchMethod` offers only bm25, an episode carries no embedding to
search by instead, and this fork's fulltext branch is dead on FalkorDB. Anchors
therefore come from the `episodes` field of the facts and entities that did rank,
which is where a caller should read them from.

`discriminates`, recall's refusal of a flat ranking, is deliberately not applied.
It guards against a ranker that returned numbers without judging, which recall
cannot detect any other way. An explicit search has both a floor and a caller who
asked, and refusing five equally good answers because they scored alike is the
worse error of the two.

Read-only, scoped to one physical graph, like the other fork tools.
"""

from __future__ import annotations

from typing import Any, Callable

from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

from models.response_types import ErrorResponse
from utils.type_config import build_fact_search_filters

DEFAULT_LIMIT = 10
MAX_LIMIT = 50

# How many candidates retrieval hands the ranker. The cross-encoder can only
# choose among what the pool contains, so a pool equal to the limit makes the
# ranking cosmetic: it reorders exactly the items that were going to be returned.
DEFAULT_POOL = 40
MAX_POOL = 200

# The floor on the cross-encoder's scale, measured on a live graph over 49 scores
# from 66 recall turns: correct hits ran 0.11 to 0.54 with a median of 0.25, and
# every wrong one sat between 0.08 and 0.105. The nearest correct hit above the
# floor scored 0.177, so the margin is wide in both directions.
DEFAULT_MIN_SCORE = 0.11

# A reranker is given a pair, not a document. A whole episode is thousands of
# characters and one such pair has already returned HTTP 500 from the local
# server, so passages are cut to something a cross-encoder is built to judge.
PASSAGE_MAX_CHARS = 2000


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
        min_score: float | None = None,
        pool: int | None = None,
        rerank: bool = True,
    ) -> dict[str, Any] | ErrorResponse:
        """Search one graph for facts, entities and episodes at once, with scores.

        Returns three ranked lists from a single retrieval pass. Facts answer what
        is known, entities answer who or what something is, and episodes are
        stretches of conversation whose own words matched — each already an anchor
        for reading the surrounding dialog.

        A hit's score is how well a cross-encoder judged it against the query, on
        the same scale recall uses. Anything below the floor is left out, so an
        empty list is an answer: memory has nothing on this.

        Args:
            query: What to look for.
            group_id: Graph to search; defaults to the configured group.
            limit: Maximum results per list.
            valid_at_after: Only facts that were true at or after this ISO-8601 time.
            valid_at_before: Only facts that were true at or before this time.
            created_at_after: Only facts recorded at or after this time — when the
                subject was discussed, as opposed to when it was true.
            min_score: Floor on the relevance score. Defaults to the measured one;
                lower it for a deliberately broad sweep.
            pool: How many candidates to rank before choosing `limit` of them.
            rerank: Rank by relevance. Off returns retrieval order with fusion
                scores, which are positions rather than relevance.
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
        candidate_limit = max(bounded, min(int(pool or DEFAULT_POOL), MAX_POOL))
        floor = DEFAULT_MIN_SCORE if min_score is None else float(min_score)

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
            # Retrieval asks for the pool, not the limit: the ranker below can only
            # choose among what this returns. Copied rather than rebuilt, since the
            # recipe already carries a limit and passing a second one is a duplicate
            # keyword.
            config = COMBINED_HYBRID_SEARCH_RRF.model_copy(
                deep=True, update={'limit': candidate_limit}
            )
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

        cross_encoder = getattr(client, 'cross_encoder', None) if rerank else None
        ranked_by = 'relevance' if cross_encoder is not None else 'fusion'

        if cross_encoder is not None:
            try:
                edges, edge_scores = await _rank(
                    cross_encoder, query, results.edges, _fact_passage, floor, bounded
                )
                nodes, node_scores = await _rank(
                    cross_encoder, query, results.nodes, _entity_passage, floor, bounded
                )
                episodes, episode_scores = await _rank(
                    cross_encoder, query, results.episodes, _episode_passage, floor, bounded
                )
            except Exception as exc:
                # A reranker that is down must not turn a working search into an
                # error: fall back to what retrieval already decided and say which
                # scale the numbers are on, so nobody reads a position as relevance.
                server.logger.warning(f'Reranking failed, falling back to fusion order: {exc}')
                cross_encoder = None
                ranked_by = 'fusion'

        if cross_encoder is None:
            edges = results.edges[:bounded]
            nodes = results.nodes[:bounded]
            episodes = results.episodes[:bounded]
            edge_scores = _scores(results.edge_reranker_scores, results.edges)[:bounded]
            node_scores = _scores(results.node_reranker_scores, results.nodes)[:bounded]
            episode_scores = _scores(results.episode_reranker_scores, results.episodes)[:bounded]

        # An entity does have provenance of its own -- (:Episodic)-[:MENTIONS]->(:Entity)
        # is how the graph records where it came up. Deriving it from the facts in
        # this result instead leaves an entity whose facts did not rank blank, which
        # is precisely the case a reader most wants to trace.
        entity_episodes: dict[str, list[str]] = {}
        entity_uuids = [node.uuid for node in nodes]
        if entity_uuids:
            try:
                rows, _, _ = await scoped_driver.execute_query(
                    """
                    MATCH (e:Episodic)-[:MENTIONS]->(n:Entity)
                    WHERE n.uuid IN $uuids
                    RETURN n.uuid AS entity_uuid, e.uuid AS episode_uuid
                    """,
                    uuids=entity_uuids,
                    routing_='r',
                )
                for row in rows:
                    entity_uuid = row.get('entity_uuid')
                    episode_uuid = row.get('episode_uuid')
                    if entity_uuid and episode_uuid:
                        entity_episodes.setdefault(str(entity_uuid), []).append(str(episode_uuid))
            except Exception as exc:
                # Provenance is an aid to reading, never the answer itself; a search
                # that found something must not fail because its sourcing did not.
                server.logger.warning(f'Could not resolve entity provenance: {exc}')

        found = len(edges) + len(nodes) + len(episodes)
        if found == 0:
            message = (
                f"Nothing in group '{effective_group_id}' matches that above a score of {floor}"
                if ranked_by == 'relevance'
                else f"Nothing in group '{effective_group_id}' matches that"
            )
        else:
            message = f"Combined search for group '{effective_group_id}' completed"

        return {
            'message': message,
            'group_id': effective_group_id,
            # Which scale the numbers are on. Relevance is comparable between
            # searches and against a floor; fusion is a position wearing a decimal
            # point, and reading one as the other is how a floor ends up three times
            # below its own noise.
            'ranked_by': ranked_by,
            'min_score': floor if ranked_by == 'relevance' else None,
            'facts': [
                {
                    'uuid': edge.uuid,
                    'fact': edge.fact,
                    'score': score,
                    'episodes': list(getattr(edge, 'episodes', []) or []),
                    # The two entities this fact connects, so a reader can follow a
                    # fact back to both of its ends.
                    'source_node_uuid': getattr(edge, 'source_node_uuid', None),
                    'target_node_uuid': getattr(edge, 'target_node_uuid', None),
                    'created_at': _iso(getattr(edge, 'created_at', None)),
                    'valid_at': _iso(getattr(edge, 'valid_at', None)),
                    'invalid_at': _iso(getattr(edge, 'invalid_at', None)),
                    'expired_at': _iso(getattr(edge, 'expired_at', None)),
                }
                for edge, score in zip(edges, edge_scores)
            ],
            'entities': [
                {
                    'uuid': node.uuid,
                    'name': node.name,
                    'score': score,
                    'summary': getattr(node, 'summary', None),
                    'episodes': entity_episodes.get(node.uuid, []),
                    'created_at': _iso(getattr(node, 'created_at', None)),
                }
                for node, score in zip(nodes, node_scores)
            ],
            'episodes': [
                {
                    'uuid': episode.uuid,
                    'name': episode.name,
                    'score': score,
                    'created_at': _iso(getattr(episode, 'created_at', None)),
                }
                for episode, score in zip(episodes, episode_scores)
            ],
        }


def _fact_passage(edge: Any) -> str:
    return str(getattr(edge, 'fact', '') or '')


def _entity_passage(node: Any) -> str:
    """A name alone is thin to judge; the summary is what says which thing it is."""
    name = str(getattr(node, 'name', '') or '')
    summary = str(getattr(node, 'summary', '') or '')
    return f'{name} — {summary}' if summary else name


def _episode_passage(episode: Any) -> str:
    return str(getattr(episode, 'content', '') or getattr(episode, 'name', '') or '')


async def _rank(
    cross_encoder: Any,
    query: str,
    items: list[Any],
    passage_of: Callable[[Any], str],
    floor: float,
    limit: int,
) -> tuple[list[Any], list[float]]:
    """The items a cross-encoder put above the floor, best first.

    The ranker answers with (passage, score) pairs sorted by score, not in the
    order it was asked, so items are matched back by their passage text. Two items
    that render to the same passage share a score, which is what they deserve.

    An item the ranker did not score is dropped rather than kept unranked: recall
    keeps those because a fact it already retrieved is worth showing even unjudged,
    while a search that was asked a question should answer with what it can stand
    behind.
    """
    if not items:
        return [], []

    passages = [passage_of(item)[:PASSAGE_MAX_CHARS] for item in items]
    scored = dict(await cross_encoder.rank(query, passages))

    kept: list[tuple[Any, float]] = []
    for item, passage in zip(items, passages):
        score = scored.get(passage)
        if score is None or float(score) < floor:
            continue
        kept.append((item, float(score)))

    kept.sort(key=lambda pair: pair[1], reverse=True)
    return [item for item, _ in kept][:limit], [score for _, score in kept][:limit]


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
