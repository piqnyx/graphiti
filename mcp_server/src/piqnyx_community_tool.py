"""piqnyx community rebuild that respects per-agent graph isolation.

The upstream `build_communities` tool calls the core without a driver, and the
core then falls back to the default database. That is correct for a deployment
where `group_id` is a label inside one shared graph, and wrong for this one,
where each agent is a separate physical FalkorDB graph: the rebuild would run
against `default_db`, find nothing, and report success while the agents' graphs
stayed empty of communities.

The core accepts a driver for exactly this reason, so the scoped driver is passed
explicitly — the same way every other fork tool reaches an agent's graph.
"""

from __future__ import annotations

from typing import Any

from models.response_types import ErrorResponse


async def _nothing_to_do(driver: Any, group: str) -> dict[str, Any] | None:
    """Report why a rebuild would be pointless, or None when it is worth doing.

    Two cases earn a skip, and both are ordinary rather than exceptional: a graph
    with no entities has nothing to cluster, and a graph whose newest episode
    predates the last build would produce the summaries it already has. A
    schedule can then name every agent — including ones that have not spoken in a
    week, or have never spoken at all — and pay nothing for the quiet ones.
    """
    records, _, _ = await driver.execute_query(
        """
        MATCH (n:Entity) WITH count(n) AS entities
        OPTIONAL MATCH (c:Community) WITH entities, max(c.created_at) AS built_at
        OPTIONAL MATCH (e:Episodic) WITH entities, built_at, max(e.created_at) AS newest
        RETURN entities AS entities, built_at AS built_at, newest AS newest
        """,
        routing_='r',
    )
    row = dict(records[0]) if records else {}
    entities = row.get('entities') or 0
    built_at = row.get('built_at')
    newest = row.get('newest')

    if not entities:
        return {
            'message': f"Nothing to cluster in group '{group}': the graph holds no entities",
            'group_id': group,
            'skipped': True,
            'reason': 'empty_graph',
        }
    if built_at is not None and newest is not None and newest <= built_at:
        return {
            'message': (
                f"Communities for group '{group}' are already current: "
                f'nothing recorded since they were built at {built_at}'
            ),
            'group_id': group,
            'skipped': True,
            'reason': 'no_new_episodes',
            'built_at': built_at,
        }
    return None


def install_build_communities_for_group_tool(server: Any) -> None:
    """Register build_communities_for_group on the already-created FastMCP server."""

    @server.mcp.tool()
    async def build_communities_for_group(
        group_id: str, force: bool = False
    ) -> dict[str, Any] | ErrorResponse:
        """Rebuild community summaries inside one agent's graph.

        Expensive: clustering runs over every entity in the graph, and the summary
        of a community is produced by pairwise LLM merges — roughly one call per
        entity in it. Intended for a scheduled run, not for a conversation.
        Communities of one agent are built from that agent's graph alone and can
        never mix data between agents.

        Skips the work when nothing has been recorded since the last build, so a
        schedule can name every agent and quiet graphs cost nothing.

        Args:
            group_id: The agent's graph to rebuild communities for.
            force: Rebuild even when nothing new has been recorded.
        """
        if server.graphiti_service is None:
            return ErrorResponse(error='Graphiti service not initialized')
        if not group_id or not group_id.strip():
            return ErrorResponse(error='group_id is required')

        group = group_id.strip()
        try:
            client = await server.graphiti_service.get_client()
            scoped_driver = client.driver.clone(database=group)

            if not force:
                skip = await _nothing_to_do(scoped_driver, group)
                if skip is not None:
                    return skip
            communities, community_edges = await client.build_communities(
                group_ids=[group], driver=scoped_driver
            )
            return {
                'message': f"Communities rebuilt for group '{group}'",
                'group_id': group,
                'communities': len(communities),
                'edges': len(community_edges),
                'names': [community.name for community in communities][:50],
                'skipped': False,
            }
        except Exception as exc:
            server.logger.error(f'Error building communities for {group}: {exc}')
            return ErrorResponse(error=f'Error building communities for {group}: {exc}')
