"""piqnyx MCP tool: fold a duplicate entity into the one it is a name for.

Extraction names an entity the way the conversation did, so the same thing arrives
under several names -- "Викинг" beside "OpenViking", a genitive beside the
nominative. No amount of prompting removes this: the second name is genuinely
present in the text, and once both nodes exist, nothing in the ingest path can
join them. Facts about one are invisible when searching the other.

Writing a note cannot fix it either. A note is a message in the batch, so at best
it records that the two are the same; the graph still holds two vertices. Joining
them is an operation on the graph, which is what this tool is.

Destructive by nature, so it is deliberately narrow: it moves what points at the
duplicate onto the canonical node and removes the duplicate, and it refuses
anything ambiguous rather than guessing.
"""

from __future__ import annotations

from typing import Any

from models.response_types import ErrorResponse


def install_merge_entities_tool(server: Any) -> None:
    """Register the entity-merge tool on the already-created FastMCP server."""

    @server.mcp.tool()
    async def merge_entities(
        duplicate_name: str,
        canonical_name: str,
        group_id: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any] | ErrorResponse:
        """Fold one entity into another that is the same thing under a different name.

        Moves every MENTIONS and RELATES_TO attached to `duplicate_name` onto
        `canonical_name`, then deletes the duplicate. Fact sentences keep their
        original wording -- they are a record of what was said, not a description of
        the graph -- so a fact may still spell the old name while pointing at the
        surviving entity.

        Refuses when either name is missing, when a name matches more than one
        entity, or when both names are the same entity. `dry_run` reports what would
        move without touching anything.
        """
        if server.graphiti_service is None:
            return ErrorResponse(error='Graphiti service not initialized')

        duplicate = (duplicate_name or '').strip()
        canonical = (canonical_name or '').strip()
        if not duplicate or not canonical:
            return ErrorResponse(error='duplicate_name and canonical_name are both required')
        if duplicate == canonical:
            return ErrorResponse(error='duplicate_name and canonical_name are the same name')

        effective_group_id = group_id or server.config.graphiti.group_id
        if not effective_group_id:
            return ErrorResponse(
                error='No group_id provided and no default group_id is configured'
            )

        try:
            client = await server.graphiti_service.get_client()
            scoped_driver = client.driver.clone(database=effective_group_id)

            async def resolve(name: str) -> list[str]:
                rows, _, _ = await scoped_driver.execute_query(
                    'MATCH (n:Entity {name: $name}) RETURN n.uuid AS uuid',
                    name=name,
                    routing_='r',
                )
                return [str(row['uuid']) for row in rows if row.get('uuid')]

            duplicate_uuids = await resolve(duplicate)
            canonical_uuids = await resolve(canonical)

            # Every refusal below names what it saw, because the caller is an agent
            # that has to decide what to do next without seeing the graph.
            if not duplicate_uuids:
                return ErrorResponse(error=f'no entity named "{duplicate}" in {effective_group_id}')
            if not canonical_uuids:
                return ErrorResponse(error=f'no entity named "{canonical}" in {effective_group_id}')
            if len(duplicate_uuids) > 1 or len(canonical_uuids) > 1:
                return ErrorResponse(
                    error=(
                        f'ambiguous: "{duplicate}" matches {len(duplicate_uuids)} entities and '
                        f'"{canonical}" matches {len(canonical_uuids)}; refusing to guess which'
                    )
                )

            duplicate_uuid = duplicate_uuids[0]
            canonical_uuid = canonical_uuids[0]
            if duplicate_uuid == canonical_uuid:
                return ErrorResponse(error='both names already resolve to the same entity')

            counts, _, _ = await scoped_driver.execute_query(
                """
                MATCH (d:Entity {uuid: $duplicate})
                OPTIONAL MATCH (:Episodic)-[m:MENTIONS]->(d)
                WITH d, count(m) AS mentions
                OPTIONAL MATCH (d)-[out:RELATES_TO]->(:Entity)
                WITH d, mentions, count(out) AS outgoing
                OPTIONAL MATCH (:Entity)-[inc:RELATES_TO]->(d)
                RETURN mentions, outgoing, count(inc) AS incoming
                """,
                duplicate=duplicate_uuid,
                routing_='r',
            )
            row = counts[0] if counts else {}
            moving = {
                'mentions': int(row.get('mentions') or 0),
                'outgoing_facts': int(row.get('outgoing') or 0),
                'incoming_facts': int(row.get('incoming') or 0),
            }

            if dry_run:
                return {
                    'message': f'"{duplicate}" would be folded into "{canonical}"',
                    'group_id': effective_group_id,
                    'duplicate_uuid': duplicate_uuid,
                    'canonical_uuid': canonical_uuid,
                    'would_move': moving,
                    'applied': False,
                }

            # Endpoints are rewritten rather than the relationships recreated: a
            # RELATES_TO carries fact text, validity bounds and its episode list, and
            # rebuilding it by hand would quietly drop whichever property was
            # forgotten. FalkorDB has no SET endpoint, so each edge is recreated with
            # its whole property map copied across, then the original removed.
            await scoped_driver.execute_query(
                """
                MATCH (e:Episodic)-[m:MENTIONS]->(d:Entity {uuid: $duplicate})
                MATCH (c:Entity {uuid: $canonical})
                MERGE (e)-[:MENTIONS]->(c)
                DELETE m
                """,
                duplicate=duplicate_uuid,
                canonical=canonical_uuid,
            )
            await scoped_driver.execute_query(
                """
                MATCH (d:Entity {uuid: $duplicate})-[r:RELATES_TO]->(t:Entity)
                MATCH (c:Entity {uuid: $canonical})
                WHERE t.uuid <> $canonical
                CREATE (c)-[n:RELATES_TO]->(t)
                SET n = properties(r)
                DELETE r
                """,
                duplicate=duplicate_uuid,
                canonical=canonical_uuid,
            )
            await scoped_driver.execute_query(
                """
                MATCH (s:Entity)-[r:RELATES_TO]->(d:Entity {uuid: $duplicate})
                MATCH (c:Entity {uuid: $canonical})
                WHERE s.uuid <> $canonical
                CREATE (s)-[n:RELATES_TO]->(c)
                SET n = properties(r)
                DELETE r
                """,
                duplicate=duplicate_uuid,
                canonical=canonical_uuid,
            )
            # Anything still attached would have been a self-loop through the
            # canonical node, which is not a fact about anything.
            await scoped_driver.execute_query(
                'MATCH (d:Entity {uuid: $duplicate}) DETACH DELETE d',
                duplicate=duplicate_uuid,
            )

            server.logger.info(
                f'merged entity "{duplicate}" into "{canonical}" in {effective_group_id}: {moving}'
            )
            return {
                'message': f'"{duplicate}" folded into "{canonical}"',
                'group_id': effective_group_id,
                'duplicate_uuid': duplicate_uuid,
                'canonical_uuid': canonical_uuid,
                'moved': moving,
                'applied': True,
            }
        except Exception as exc:
            server.logger.error(f'Error merging entities: {exc}')
            return ErrorResponse(error=f'Error merging entities: {exc}')
