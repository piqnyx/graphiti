"""piqnyx read-only MCP tool for fetching specific episodes by uuid or name.

Upstream exposes episodes only as "the most recent N", which cannot answer the
question this fork needs: given a fact, show the conversation that produced it.
Facts carry the uuids of their source episodes, and every episode carries a name
of the form `<saga tail>-<batch number>`, so a lookup by uuid finds the source and
a lookup by name reaches its neighbours in the chain.

Read-only, LLM-free, and scoped to one physical graph, like the other fork tools.
"""

from __future__ import annotations

from typing import Any

from models.response_types import ErrorResponse

MAX_EPISODES = 100


def install_get_episodes_by_ref_tool(server: Any) -> None:
    """Register get_episodes_by_ref on the already-created FastMCP server."""

    @server.mcp.tool()
    async def get_episodes_by_ref(
        uuids: list[str] | None = None,
        names: list[str] | None = None,
        group_id: str | None = None,
    ) -> dict[str, Any] | ErrorResponse:
        """Return specific episodes, with their full text, by uuid or by name.

        Use to read the conversation behind a fact: a fact lists the uuids of the
        episodes that produced it, and episode names carry a batch number whose
        neighbours are the surrounding conversation.

        Args:
            uuids: Episode UUIDs to fetch.
            names: Episode names to fetch, such as `8248439450-12`.
            group_id: Graph to read; defaults to the configured group.
        """
        if server.graphiti_service is None:
            return ErrorResponse(error='Graphiti service not initialized')

        wanted_uuids = [u for u in (uuids or []) if isinstance(u, str) and u.strip()]
        wanted_names = [n for n in (names or []) if isinstance(n, str) and n.strip()]
        if not wanted_uuids and not wanted_names:
            return ErrorResponse(error='Provide at least one uuid or name')

        effective_group_id = group_id or server.config.graphiti.group_id
        if not effective_group_id:
            return ErrorResponse(
                error='No group_id provided and no default group_id is configured'
            )

        try:
            client = await server.graphiti_service.get_client()
            scoped_driver = client.driver.clone(database=effective_group_id)
            records, _, _ = await scoped_driver.execute_query(
                """
                MATCH (e:Episodic)
                WHERE e.uuid IN $uuids OR e.name IN $names
                RETURN e.uuid AS uuid,
                       e.name AS name,
                       e.content AS content,
                       e.created_at AS created_at
                ORDER BY e.created_at ASC
                LIMIT $limit
                """,
                uuids=wanted_uuids[:MAX_EPISODES],
                names=wanted_names[:MAX_EPISODES],
                limit=MAX_EPISODES,
                routing_='r',
            )

            episodes = [dict(record) for record in records]
            return {
                'message': f'Retrieved {len(episodes)} episode(s)',
                'group_id': effective_group_id,
                'episodes': episodes,
            }
        except Exception as exc:
            server.logger.error(f'Error getting episodes by ref: {exc}')
            return ErrorResponse(error=f'Error getting episodes by ref: {exc}')
