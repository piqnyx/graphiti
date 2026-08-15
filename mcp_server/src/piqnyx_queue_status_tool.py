"""piqnyx read-only MCP tool for reliable queue health."""

from __future__ import annotations

from typing import Any

from models.response_types import ErrorResponse


def install_get_queue_status_tool(server: Any) -> None:
    """Register get_queue_status on the already-created FastMCP server."""

    @server.mcp.tool()
    async def get_queue_status(group_id: str) -> dict[str, Any] | ErrorResponse:
        """Return safe in-memory queue health for one isolated group without invoking an LLM."""
        queue = server.queue_service
        if queue is None:
            return ErrorResponse(error='Queue service not initialized')

        try:
            if hasattr(queue, 'get_failure_status'):
                status = queue.get_failure_status(group_id)
            else:
                base = queue.get_queue_status(group_id)
                status = {
                    **base,
                    'blocked': False,
                    'attempts': 0,
                    'last_error': None,
                    'episode_uuid': None,
                    'episode_name': None,
                    'saga': None,
                }

            return {
                'message': f"Queue status for group '{group_id}' retrieved successfully",
                **status,
            }
        except Exception as exc:
            server.logger.error(f'Error getting queue status: {exc}')
            return ErrorResponse(error=f'Error getting queue status: {exc}')
