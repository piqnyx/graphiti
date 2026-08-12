"""Queue service for managing episode processing."""

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_EPISODE_QUEUE_MAX_SIZE = 100
EPISODE_QUEUE_MAX_SIZE_ENV = 'EPISODE_QUEUE_MAX_SIZE'


class EpisodeQueueFullError(RuntimeError):
    """Raised when a per-group episode queue has reached its configured bound."""


class QueueService:
    """Service for managing bounded sequential episode processing queues by group_id."""

    def __init__(self, max_size_per_group: int | None = None):
        """Initialize the queue service.

        Args:
            max_size_per_group: Maximum number of pending episodes per group. When
                omitted, EPISODE_QUEUE_MAX_SIZE is read from the environment and
                defaults to 100. The currently-processing episode is not counted.
        """
        if max_size_per_group is None:
            raw_max_size = os.getenv(
                EPISODE_QUEUE_MAX_SIZE_ENV, str(DEFAULT_EPISODE_QUEUE_MAX_SIZE)
            )
            try:
                max_size_per_group = int(raw_max_size)
            except ValueError as exc:
                raise ValueError(
                    f'{EPISODE_QUEUE_MAX_SIZE_ENV} must be a positive integer'
                ) from exc

        if max_size_per_group <= 0:
            raise ValueError('max_size_per_group must be a positive integer')

        self._max_size_per_group = max_size_per_group
        # Dictionary to store bounded queues for each group_id.
        self._episode_queues: dict[str, asyncio.Queue] = {}
        # Dictionary to track if a worker is running/scheduled for each group_id.
        self._queue_workers: dict[str, bool] = {}
        # Store the graphiti client after initialization.
        self._graphiti_client: Any = None

    def _get_or_create_queue(self, group_id: str) -> asyncio.Queue:
        queue = self._episode_queues.get(group_id)
        if queue is None:
            queue = asyncio.Queue(maxsize=self._max_size_per_group)
            self._episode_queues[group_id] = queue
        return queue

    async def add_episode_task(
        self, group_id: str, process_func: Callable[[], Awaitable[None]]
    ) -> int:
        """Add an episode processing task to the queue.

        Args:
            group_id: The group ID for the episode.
            process_func: The async function to process the episode.

        Returns:
            The position in the pending queue at enqueue time.

        Raises:
            EpisodeQueueFullError: If the group's pending queue is full.
        """
        queue = self._get_or_create_queue(group_id)

        # Never block an MCP request waiting for queue capacity. A bounded queue
        # should provide backpressure immediately so callers can retry later.
        try:
            queue.put_nowait(process_func)
        except asyncio.QueueFull as exc:
            logger.warning(
                'Episode queue full for group_id=%s size=%d max=%d',
                group_id,
                queue.qsize(),
                self._max_size_per_group,
            )
            raise EpisodeQueueFullError(
                f"episode queue is full for group '{group_id}' "
                f'(max pending={self._max_size_per_group})'
            ) from exc

        queue_position = queue.qsize()

        # Mark the worker as scheduled before yielding to the event loop. This
        # closes the race where concurrent enqueue calls could all observe False
        # and create multiple workers for one group.
        if not self._queue_workers.get(group_id, False):
            self._queue_workers[group_id] = True
            try:
                asyncio.create_task(self._process_episode_queue(group_id))
            except Exception:
                self._queue_workers[group_id] = False
                raise

        logger.info(
            'Queued episode for group_id=%s pending=%d max=%d worker=%s',
            group_id,
            queue_position,
            self._max_size_per_group,
            self._queue_workers.get(group_id, False),
        )
        return queue_position

    async def _process_episode_queue(self, group_id: str) -> None:
        """Process episodes for a specific group_id sequentially.

        This function runs as a long-lived task that processes episodes from the
        queue one at a time.
        """
        logger.info(f'Starting episode queue worker for group_id: {group_id}')

        try:
            while True:
                # Get the next episode processing function from the queue. This
                # waits without consuming CPU when the queue is empty.
                process_func = await self._episode_queues[group_id].get()

                try:
                    await process_func()
                except Exception as e:
                    logger.error(
                        f'Error processing queued episode for group_id {group_id}: {str(e)}'
                    )
                finally:
                    self._episode_queues[group_id].task_done()
        except asyncio.CancelledError:
            logger.info(f'Episode queue worker for group_id {group_id} was cancelled')
        except Exception as e:
            logger.error(f'Unexpected error in queue worker for group_id {group_id}: {str(e)}')
        finally:
            self._queue_workers[group_id] = False
            logger.info(f'Stopped episode queue worker for group_id: {group_id}')

    def get_queue_size(self, group_id: str) -> int:
        """Get the current pending queue size for group_id."""
        queue = self._episode_queues.get(group_id)
        return queue.qsize() if queue is not None else 0

    def get_queue_capacity(self) -> int:
        """Get the configured maximum pending queue size per group."""
        return self._max_size_per_group

    def get_queue_status(self, group_id: str) -> dict[str, int | bool | str]:
        """Return safe operational queue metadata without episode content."""
        return {
            'group_id': group_id,
            'pending': self.get_queue_size(group_id),
            'max_pending': self._max_size_per_group,
            'worker_running': self.is_worker_running(group_id),
        }

    def is_worker_running(self, group_id: str) -> bool:
        """Check if a worker is running or already scheduled for group_id."""
        return self._queue_workers.get(group_id, False)

    async def initialize(self, graphiti_client: Any) -> None:
        """Initialize the queue service with a graphiti client.

        Args:
            graphiti_client: The Graphiti client instance used for processing episodes.
        """
        self._graphiti_client = graphiti_client
        logger.info(
            'Queue service initialized with graphiti client (max_pending_per_group=%d)',
            self._max_size_per_group,
        )

    async def add_episode(
        self,
        group_id: str,
        name: str,
        content: str,
        source_description: str,
        episode_type: Any,
        entity_types: Any,
        uuid: str | None,
        reference_time: datetime | None = None,
        edge_types: Any = None,
        edge_type_map: Any = None,
        excluded_entity_types: list[str] | None = None,
        previous_episode_uuids: list[str] | None = None,
        custom_extraction_instructions: str | None = None,
        update_communities: bool = False,
        saga: str | None = None,
        saga_previous_episode_uuid: str | None = None,
    ) -> int:
        """Add an episode for background processing."""
        if self._graphiti_client is None:
            raise RuntimeError('Queue service not initialized. Call initialize() first.')

        async def process_episode():
            """Process the episode using the Graphiti client."""
            try:
                logger.info(f'Processing episode {uuid} for group {group_id}')

                await self._graphiti_client.add_episode(
                    name=name,
                    episode_body=content,
                    source_description=source_description,
                    source=episode_type,
                    group_id=group_id,
                    reference_time=reference_time or datetime.now(timezone.utc),
                    entity_types=entity_types,
                    edge_types=edge_types,
                    edge_type_map=edge_type_map,
                    excluded_entity_types=excluded_entity_types,
                    previous_episode_uuids=previous_episode_uuids,
                    custom_extraction_instructions=custom_extraction_instructions,
                    update_communities=update_communities,
                    saga=saga,
                    saga_previous_episode_uuid=saga_previous_episode_uuid,
                    uuid=uuid,
                )

                logger.info(f'Successfully processed episode {uuid} for group {group_id}')

            except Exception as e:
                logger.error(f'Failed to process episode {uuid} for group {group_id}: {str(e)}')
                raise

        return await self.add_episode_task(group_id, process_episode)
