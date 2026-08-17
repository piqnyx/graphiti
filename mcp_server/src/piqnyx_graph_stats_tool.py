"""piqnyx read-only MCP tool for graph size, shape and integrity diagnostics.

Everything here is a read-only query against the physical FalkorDB graph selected
by group_id, so per-agent isolation is preserved exactly as elsewhere in the fork.

Each query is isolated: a query that fails — because a label is absent in a young
graph, or because a FalkorDB build lacks a function — records its error and leaves
the rest of the report intact. A diagnostic tool that returns nothing at the first
unsupported query would be worse than useless, since it fails precisely when the
graph is in an unusual state and the report is most needed.
"""

from __future__ import annotations

from typing import Any

from models.response_types import ErrorResponse

DEFAULT_TOP_ENTITIES = 10
MAX_TOP_ENTITIES = 50


def install_get_graph_stats_tool(server: Any) -> None:
    """Register get_graph_stats on the already-created FastMCP server."""

    @server.mcp.tool()
    async def get_graph_stats(
        group_id: str | None = None,
        top_entities: int = DEFAULT_TOP_ENTITIES,
        standalone_source_description: str | None = None,
    ) -> dict[str, Any] | ErrorResponse:
        """Return size, shape and integrity diagnostics for one isolated graph.

        Read-only and LLM-free. Reports node and edge counts, the most connected
        entities, memory age, and integrity checks: duplicated episode names,
        episodes detached from a saga or from any entity, broken NEXT_EPISODE
        chains, facts with no source episode, and isolated entities.

        Args:
            group_id: Graph to inspect; defaults to the configured group.
            top_entities: How many of the most connected entities to list.
            standalone_source_description: Episodes written with this
                source_description belong to no dialog by design — a standalone
                note must not join a dialog's chain — so they are excluded from
                the detached-episode count instead of being reported as damage.
        """
        if server.graphiti_service is None:
            return ErrorResponse(error='Graphiti service not initialized')

        effective_group_id = group_id or server.config.graphiti.group_id
        if not effective_group_id:
            return ErrorResponse(
                error='No group_id provided and no default group_id is configured'
            )

        limit = max(1, min(int(top_entities or DEFAULT_TOP_ENTITIES), MAX_TOP_ENTITIES))

        try:
            client = await server.graphiti_service.get_client()
            scoped_driver = client.driver.clone(database=effective_group_id)
        except Exception as exc:
            server.logger.error(f'Error opening graph for stats: {exc}')
            return ErrorResponse(error=f'Error opening graph for stats: {exc}')

        errors: list[str] = []

        async def query(label: str, cypher: str, **params: Any) -> list[dict[str, Any]]:
            """Run one read-only query, recording rather than raising failures."""
            try:
                records, _, _ = await scoped_driver.execute_query(
                    cypher, routing_='r', **params
                )
                return [dict(record) for record in records]
            except Exception as exc:  # noqa: BLE001 - deliberately non-fatal
                errors.append(f'{label}: {exc}')
                return []

        def scalar(rows: list[dict[str, Any]], key: str, default: Any = 0) -> Any:
            if not rows:
                return default
            value = rows[0].get(key)
            return default if value is None else value

        entities = scalar(
            await query('entity_count', 'MATCH (n:Entity) RETURN count(n) AS value'), 'value'
        )
        episodes = scalar(
            await query('episode_count', 'MATCH (e:Episodic) RETURN count(e) AS value'), 'value'
        )
        sagas = scalar(
            await query('saga_count', 'MATCH (s:Saga) RETURN count(s) AS value'), 'value'
        )
        facts = scalar(
            await query('fact_count', 'MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS value'),
            'value',
        )
        mentions = scalar(
            await query('mention_count', 'MATCH ()-[r:MENTIONS]->() RETURN count(r) AS value'),
            'value',
        )

        top = await query(
            'top_entities',
            """
            MATCH (n:Entity)
            OPTIONAL MATCH (n)-[r:RELATES_TO]-()
            WITH n, count(r) AS degree
            WHERE degree > 0
            RETURN n.name AS name, degree AS degree
            ORDER BY degree DESC, name ASC
            LIMIT $limit
            """,
            limit=limit,
        )

        # Ordering beats min()/max() here: the driver hands back the property in
        # whatever form it was stored, and a row keeps the episode's name next to
        # its timestamp, which is what makes the answer readable.
        oldest = await query(
            'oldest_episode',
            """
            MATCH (e:Episodic)
            WHERE e.created_at IS NOT NULL
            RETURN e.name AS name, e.created_at AS created_at
            ORDER BY e.created_at ASC LIMIT 1
            """,
        )
        newest = await query(
            'newest_episode',
            """
            MATCH (e:Episodic)
            WHERE e.created_at IS NOT NULL
            RETURN e.name AS name, e.created_at AS created_at
            ORDER BY e.created_at DESC LIMIT 1
            """,
        )

        per_saga = await query(
            'episodes_per_saga',
            """
            MATCH (s:Saga)
            OPTIONAL MATCH (s)-[:HAS_EPISODE]->(e:Episodic)
            WITH s, count(e) AS episodes
            RETURN s.name AS saga, episodes AS episodes
            ORDER BY episodes DESC, saga ASC
            LIMIT $limit
            """,
            limit=limit,
        )

        duplicate_names = await query(
            'duplicate_episode_names',
            """
            MATCH (e:Episodic)
            WITH e.name AS name, count(*) AS copies
            WHERE copies > 1
            RETURN name AS name, copies AS copies
            ORDER BY copies DESC, name ASC
            LIMIT $limit
            """,
            limit=limit,
        )

        orphan_episodes = scalar(
            await query(
                'episodes_without_saga',
                """
                MATCH (e:Episodic)
                WHERE NOT (:Saga)-[:HAS_EPISODE]->(e)
                  AND ($standalone IS NULL OR e.source_description <> $standalone)
                RETURN count(e) AS value
                """,
                standalone=standalone_source_description,
            ),
            'value',
        )

        silent_episodes = scalar(
            await query(
                'episodes_without_entities',
                """
                MATCH (e:Episodic)
                WHERE NOT (e)-[:MENTIONS]->()
                RETURN count(e) AS value
                """,
            ),
            'value',
        )

        # One head per saga is correct: the chain starts somewhere. More than one
        # means the NEXT_EPISODE chain was broken and restarted, which is exactly
        # the failure the numbering check in the plugin cannot see.
        chain_heads = await query(
            'chain_heads_per_saga',
            """
            MATCH (s:Saga)-[:HAS_EPISODE]->(e:Episodic)
            WHERE NOT ()-[:NEXT_EPISODE]->(e)
            WITH s.name AS saga, count(e) AS heads
            WHERE heads > 1
            RETURN saga AS saga, heads AS heads
            ORDER BY heads DESC, saga ASC
            LIMIT $limit
            """,
            limit=limit,
        )

        facts_without_source = scalar(
            await query(
                'facts_without_provenance',
                """
                MATCH ()-[r:RELATES_TO]->()
                WHERE r.episodes IS NULL OR size(r.episodes) = 0
                RETURN count(r) AS value
                """,
            ),
            'value',
        )

        # Communities are built by a separate scheduled run, never on this path:
        # summarising a cluster calls an LLM per community and takes as long as it
        # takes, and a diagnostic must stay cheap. Here they are only read.
        communities = await query(
            'communities',
            """
            MATCH (c:Community)
            OPTIONAL MATCH (c)-[:HAS_MEMBER]->(m)
            WITH c, count(m) AS members
            RETURN c.name AS name, members AS members, c.summary AS summary,
                   c.created_at AS created_at
            ORDER BY members DESC, name ASC
            LIMIT $limit
            """,
            limit=limit,
        )
        community_count = scalar(
            await query('community_count', 'MATCH (c:Community) RETURN count(c) AS value'),
            'value',
        )
        community_built_at = scalar(
            await query(
                'communities_built_at',
                'MATCH (c:Community) RETURN max(c.created_at) AS value',
            ),
            'value',
            None,
        )
        # What has arrived since the last build is the honest measure of how stale
        # the summaries are — far more useful than the age of the build alone.
        episodes_since_build = scalar(
            await query(
                'episodes_since_communities',
                """
                MATCH (e:Episodic)
                WHERE $built IS NOT NULL AND e.created_at > $built
                RETURN count(e) AS value
                """,
                built=community_built_at,
            ),
            'value',
        )

        isolated_entities = scalar(
            await query(
                'isolated_entities',
                """
                MATCH (n:Entity)
                WHERE NOT (n)-[:RELATES_TO]-()
                RETURN count(n) AS value
                """,
            ),
            'value',
        )

        return {
            'message': f"Graph statistics for group '{effective_group_id}' retrieved successfully",
            'group_id': effective_group_id,
            'size': {
                'entities': entities,
                'episodes': episodes,
                'sagas': sagas,
                'facts': facts,
                'mentions': mentions,
            },
            'top_entities': top,
            'episodes_per_saga': per_saga,
            'communities': {
                'count': community_count,
                'built_at': community_built_at,
                'episodes_since_build': episodes_since_build,
                'largest': communities,
            },
            'oldest_episode': oldest[0] if oldest else None,
            'newest_episode': newest[0] if newest else None,
            'integrity': {
                'duplicate_episode_names': duplicate_names,
                'episodes_without_saga': orphan_episodes,
                'episodes_without_entities': silent_episodes,
                'sagas_with_broken_chain': chain_heads,
                'facts_without_provenance': facts_without_source,
                'isolated_entities': isolated_entities,
            },
            'query_errors': errors,
        }
