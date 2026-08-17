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


def install_build_communities_for_group_tool(server: Any) -> None:
    """Register build_communities_for_group on the already-created FastMCP server."""

    @server.mcp.tool()
    async def build_communities_for_group(group_id: str) -> dict[str, Any] | ErrorResponse:
        """Rebuild community summaries inside one agent's graph.

        Expensive: clustering runs over every entity in the graph and an LLM
        writes a summary per community. Intended for a scheduled run, not for a
        conversation. Communities of one agent are built from that agent's graph
        alone and can never mix data between agents.

        Args:
            group_id: The agent's graph to rebuild communities for.
        """
        if server.graphiti_service is None:
            return ErrorResponse(error='Graphiti service not initialized')
        if not group_id or not group_id.strip():
            return ErrorResponse(error='group_id is required')

        group = group_id.strip()
        try:
            client = await server.graphiti_service.get_client()
            scoped_driver = client.driver.clone(database=group)
            communities, community_edges = await client.build_communities(
                group_ids=[group], driver=scoped_driver
            )
            return {
                'message': f"Communities rebuilt for group '{group}'",
                'group_id': group,
                'communities': len(communities),
                'edges': len(community_edges),
                'names': [community.name for community in communities][:50],
            }
        except Exception as exc:
            server.logger.error(f'Error building communities for {group}: {exc}')
            return ErrorResponse(error=f'Error building communities for {group}: {exc}')
