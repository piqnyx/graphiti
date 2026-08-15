"""piqnyx read-only MCP tool for deterministic saga state recovery."""

from __future__ import annotations

from typing import Any

from graphiti_core.nodes import SagaNode

from models.response_types import ErrorResponse, SagaStateResponse


def install_get_saga_tool(server: Any) -> None:
    """Register a read-only get_saga tool on the already-created FastMCP server."""

    @server.mcp.tool()
    async def get_saga(
        saga_name: str,
        group_id: str | None = None,
    ) -> SagaStateResponse | ErrorResponse:
        """Return persisted state for one saga without invoking an LLM.

        Sagas are keyed by (name, group_id). The lookup is executed against the
        physical FalkorDB graph selected by group_id, preserving piqnyx's strict
        per-agent graph isolation.
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

            # Reload by UUID through the scoped driver so every persisted Saga
            # property is hydrated by the canonical SagaNode deserializer.
            saga = await SagaNode.get_by_uuid(scoped_driver, match.uuid)
            return SagaStateResponse(
                message=f"Saga '{saga_name}' retrieved successfully",
                uuid=saga.uuid,
                name=saga.name,
                group_id=saga.group_id,
                created_at=saga.created_at.isoformat() if saga.created_at else None,
                summary=saga.summary,
                first_episode_uuid=saga.first_episode_uuid,
                last_episode_uuid=saga.last_episode_uuid,
            )
        except Exception as exc:
            server.logger.error(f'Error getting saga: {exc}')
            return ErrorResponse(error=f'Error getting saga: {exc}')
