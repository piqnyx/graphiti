"""piqnyx read-only MCP tool for deterministic Saga recovery and integrity checks."""

from __future__ import annotations

from collections import Counter
from typing import Any

from graphiti_core.nodes import SagaNode

from models.response_types import ErrorResponse, SagaStateResponse


def _validate_chain(
    *,
    episode_rows: list[dict[str, Any]],
    next_rows: list[dict[str, Any]],
    first_episode_uuid: str | None,
    last_episode_uuid: str | None,
) -> tuple[bool, list[str], int]:
    members = {str(row['uuid']) for row in episode_rows if row.get('uuid')}
    errors: list[str] = []

    if not members:
        if first_episode_uuid is not None:
            errors.append('empty saga has first_episode_uuid')
        if last_episode_uuid is not None:
            errors.append('empty saga has last_episode_uuid')
        return not errors, errors, 0

    if first_episode_uuid not in members:
        errors.append('first_episode_uuid does not identify a saga episode')
    if last_episode_uuid not in members:
        errors.append('last_episode_uuid does not identify a saga episode')

    names = [str(row.get('name') or '') for row in episode_rows]
    duplicate_names = sorted(name for name, count in Counter(names).items() if name and count > 1)
    if duplicate_names:
        errors.append(f'duplicate episode names: {", ".join(duplicate_names[:10])}')

    outgoing: dict[str, list[str]] = {uuid: [] for uuid in members}
    incoming: dict[str, list[str]] = {uuid: [] for uuid in members}
    for row in next_rows:
        source = str(row.get('source_uuid') or '')
        target = str(row.get('target_uuid') or '')
        if not source or not target:
            continue
        if source in members:
            outgoing[source].append(target)
            if target not in members:
                errors.append(f'episode {source} points outside saga to {target}')
        if target in members:
            incoming[target].append(source)
            if source not in members:
                errors.append(f'episode {target} has predecessor outside saga {source}')

    for uuid in members:
        if len(outgoing[uuid]) > 1:
            errors.append(f'episode {uuid} has {len(outgoing[uuid])} NEXT_EPISODE successors')
        if len(incoming[uuid]) > 1:
            errors.append(f'episode {uuid} has {len(incoming[uuid])} NEXT_EPISODE predecessors')

    heads = sorted(uuid for uuid in members if len(incoming[uuid]) == 0)
    tails = sorted(uuid for uuid in members if len(outgoing[uuid]) == 0)
    if len(heads) != 1:
        errors.append(f'saga has {len(heads)} chain heads instead of 1')
    if len(tails) != 1:
        errors.append(f'saga has {len(tails)} chain tails instead of 1')
    if len(heads) == 1 and first_episode_uuid != heads[0]:
        errors.append('first_episode_uuid does not match the unique chain head')
    if len(tails) == 1 and last_episode_uuid != tails[0]:
        errors.append('last_episode_uuid does not match the unique chain tail')

    visited: set[str] = set()
    current = first_episode_uuid if first_episode_uuid in members else None
    while current is not None and current not in visited:
        visited.add(current)
        successors = [target for target in outgoing[current] if target in members]
        current = successors[0] if len(successors) == 1 else None

    if current is not None and current in visited:
        errors.append(f'NEXT_EPISODE cycle detected at {current}')
    if len(visited) != len(members):
        errors.append(
            f'chain from first_episode_uuid reaches {len(visited)} of {len(members)} saga episodes'
        )

    # Keep the response useful even for corrupted graphs: chain_count is what can
    # actually be reached from the declared head, not merely HAS_EPISODE count.
    return not errors, errors, len(visited)


def install_get_saga_tool(server: Any) -> None:
    """Register a read-only get_saga tool on the already-created FastMCP server."""

    @server.mcp.tool()
    async def get_saga(
        saga_name: str,
        group_id: str | None = None,
    ) -> SagaStateResponse | ErrorResponse:
        """Return persisted state and prove that one Saga is a single linear chain.

        Recovery code must not infer a next batch number from a graph with forks,
        detached episodes, stale first/last pointers, duplicate names, cycles, or
        NEXT_EPISODE edges crossing the Saga boundary. This tool therefore reports
        the structural proof together with the ordinary Saga fields and invokes no
        LLM.
        """
        if server.graphiti_service is None:
            return ErrorResponse(error='Graphiti service not initialized')

        try:
            client = await server.graphiti_service.get_client()
            effective_group_id = group_id or server.config.graphiti.group_id
            if not effective_group_id:
                return ErrorResponse(
                    error='No group_id provided and no default group_id is configured'
                )

            scoped_driver = client.driver.clone(database=effective_group_id)
            sagas = await SagaNode.get_by_group_ids(scoped_driver, [effective_group_id])
            match = next((saga for saga in sagas if saga.name == saga_name), None)
            if match is None:
                return ErrorResponse(
                    error=f"No saga named '{saga_name}' found in group '{effective_group_id}'"
                )

            saga = await SagaNode.get_by_uuid(scoped_driver, match.uuid)
            episode_rows, _, _ = await scoped_driver.execute_query(
                """
                MATCH (s:Saga {uuid: $uuid})-[:HAS_EPISODE]->(e:Episodic)
                RETURN e.uuid AS uuid, e.name AS name
                """,
                uuid=saga.uuid,
                routing_='r',
            )
            # Two plain queries rather than one aggregating pass. FalkorDB cannot carry
            # an aggregated alias across a following MATCH: the combined form failed with
            # "_AR_EXP_UpdateEntityIdx: Unable to locate a value with alias outgoing_rows
            # within the record", which surfaced as every flush failing to commit. The
            # union is trivially done here and the result is identical -- an OPTIONAL
            # MATCH whose rows were then discarded for being NULL is just a MATCH.
            outgoing_rows, _, _ = await scoped_driver.execute_query(
                """
                MATCH (s:Saga {uuid: $uuid})-[:HAS_EPISODE]->(member:Episodic)
                MATCH (member)-[:NEXT_EPISODE]->(outgoing:Episodic)
                RETURN DISTINCT member.uuid AS source_uuid, outgoing.uuid AS target_uuid
                """,
                uuid=saga.uuid,
                routing_='r',
            )
            incoming_rows, _, _ = await scoped_driver.execute_query(
                """
                MATCH (s:Saga {uuid: $uuid})-[:HAS_EPISODE]->(member:Episodic)
                MATCH (incoming:Episodic)-[:NEXT_EPISODE]->(member)
                RETURN DISTINCT incoming.uuid AS source_uuid, member.uuid AS target_uuid
                """,
                uuid=saga.uuid,
                routing_='r',
            )
            next_rows: list[dict[str, Any]] = []
            seen_edges: set[tuple[str, str]] = set()
            for row in (*outgoing_rows, *incoming_rows):
                source = row.get('source_uuid')
                target = row.get('target_uuid')
                if not source or not target:
                    continue
                edge = (str(source), str(target))
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                next_rows.append({'source_uuid': edge[0], 'target_uuid': edge[1]})

            integrity_ok, integrity_errors, chain_count = _validate_chain(
                episode_rows=episode_rows,
                next_rows=next_rows,
                first_episode_uuid=saga.first_episode_uuid,
                last_episode_uuid=saga.last_episode_uuid,
            )

            return SagaStateResponse(
                message=f"Saga '{saga_name}' retrieved successfully",
                uuid=saga.uuid,
                name=saga.name,
                group_id=saga.group_id,
                created_at=saga.created_at.isoformat() if saga.created_at else None,
                summary=saga.summary,
                first_episode_uuid=saga.first_episode_uuid,
                last_episode_uuid=saga.last_episode_uuid,
                episode_count=len(episode_rows),
                integrity_ok=integrity_ok,
                integrity_errors=integrity_errors,
                chain_count=chain_count,
            )
        except Exception as exc:
            server.logger.error(f'Error getting saga: {exc}')
            return ErrorResponse(error=f'Error getting saga: {exc}')
