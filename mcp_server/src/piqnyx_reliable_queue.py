"""piqnyx reliable per-group episode queue overlay.

This layer retries the current episode in place and blocks only that group after
terminal failure. Pending episodes are never allowed to leapfrog a failed
predecessor.
"""

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from datetime import datetime
from typing import Any

from services import queue_service as queue_module

logger = logging.getLogger(__name__)

_BaseQueueService = queue_module.QueueService
_installed = False

DEFAULT_PROCESS_MAX_ATTEMPTS = 5
DEFAULT_RETRY_BASE_SECONDS = 2.0
DEFAULT_RETRY_MAX_SECONDS = 30.0


class EpisodeQueueBlockedError(RuntimeError):
    """Raised when a group is blocked behind a terminally failed episode."""


class ReliableQueueService(_BaseQueueService):
    """QueueService with retry-in-place and stop-on-terminal-failure semantics."""

    def __init__(
        self,
        max_size_per_group: int | None = None,
        process_max_attempts: int | None = None,
        retry_base_seconds: float | None = None,
        retry_max_seconds: float | None = None,
    ):
        super().__init__(max_size_per_group=max_size_per_group)

        self._process_max_attempts = self._read_positive_int(
            process_max_attempts,
            'EPISODE_PROCESS_MAX_ATTEMPTS',
            DEFAULT_PROCESS_MAX_ATTEMPTS,
        )
        self._retry_base_seconds = self._read_nonnegative_float(
            retry_base_seconds,
            'EPISODE_PROCESS_RETRY_BASE_SECONDS',
            DEFAULT_RETRY_BASE_SECONDS,
        )
        self._retry_max_seconds = self._read_nonnegative_float(
            retry_max_seconds,
            'EPISODE_PROCESS_RETRY_MAX_SECONDS',
            DEFAULT_RETRY_MAX_SECONDS,
        )
        if self._retry_max_seconds < self._retry_base_seconds:
            raise ValueError(
                'EPISODE_PROCESS_RETRY_MAX_SECONDS must be greater than or equal to '
                'EPISODE_PROCESS_RETRY_BASE_SECONDS'
            )

        self._blocked_groups: dict[str, str] = {}
        self._failed_tasks: dict[str, Callable[[], Awaitable[None]]] = {}
        self._failed_attempts: dict[str, int] = {}
        self._failed_metadata: dict[str, dict[str, Any]] = {}
        self._task_metadata: dict[int, dict[str, Any]] = {}
        self._enqueue_metadata: ContextVar[dict[str, Any] | None] = ContextVar(
            f'piqnyx_queue_metadata_{id(self)}', default=None
        )

    @staticmethod
    def _read_positive_int(value: int | None, env_name: str, default: int) -> int:
        raw: Any = value if value is not None else os.getenv(env_name, str(default))
        try:
            parsed = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{env_name} must be a positive integer') from exc
        if parsed <= 0:
            raise ValueError(f'{env_name} must be a positive integer')
        return parsed

    @staticmethod
    def _read_nonnegative_float(value: float | None, env_name: str, default: float) -> float:
        raw: Any = value if value is not None else os.getenv(env_name, str(default))
        try:
            parsed = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{env_name} must be a non-negative number') from exc
        if parsed < 0:
            raise ValueError(f'{env_name} must be a non-negative number')
        return parsed

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
        """Attach safe episode identity metadata while preserving upstream enqueue behavior."""
        token = self._enqueue_metadata.set(
            {
                'episode_uuid': uuid,
                'episode_name': name,
                'saga': saga,
            }
        )
        try:
            return await super().add_episode(
                group_id=group_id,
                name=name,
                content=content,
                source_description=source_description,
                episode_type=episode_type,
                entity_types=entity_types,
                uuid=uuid,
                reference_time=reference_time,
                edge_types=edge_types,
                edge_type_map=edge_type_map,
                excluded_entity_types=excluded_entity_types,
                previous_episode_uuids=previous_episode_uuids,
                custom_extraction_instructions=custom_extraction_instructions,
                update_communities=update_communities,
                saga=saga,
                saga_previous_episode_uuid=saga_previous_episode_uuid,
            )
        finally:
            self._enqueue_metadata.reset(token)

    async def add_episode_task(
        self, group_id: str, process_func: Callable[[], Awaitable[None]]
    ) -> int:
        if self.is_group_blocked(group_id):
            last_error = self._blocked_groups[group_id]
            raise EpisodeQueueBlockedError(
                f"episode queue for group '{group_id}' is blocked by a failed predecessor: "
                f'{last_error}'
            )

        metadata = self._enqueue_metadata.get()
        if metadata is not None:
            self._task_metadata[id(process_func)] = dict(metadata)
        try:
            return await super().add_episode_task(group_id, process_func)
        except Exception:
            self._task_metadata.pop(id(process_func), None)
            raise

    def is_group_blocked(self, group_id: str) -> bool:
        return group_id in self._blocked_groups

    def get_failure_status(self, group_id: str) -> dict[str, Any]:
        metadata = self._failed_metadata.get(group_id, {})
        return {
            'group_id': group_id,
            'blocked': self.is_group_blocked(group_id),
            'attempts': self._failed_attempts.get(group_id, 0),
            'last_error': self._blocked_groups.get(group_id),
            'pending': self.get_queue_size(group_id),
            'episode_uuid': metadata.get('episode_uuid'),
            'episode_name': metadata.get('episode_name'),
            'saga': metadata.get('saga'),
        }

    def _retry_delay(self, failed_attempt_number: int) -> float:
        delay = self._retry_base_seconds * (2 ** max(failed_attempt_number - 1, 0))
        return min(delay, self._retry_max_seconds)

    async def _run_with_retries(
        self,
        group_id: str,
        process_func: Callable[[], Awaitable[None]],
    ) -> tuple[bool, Exception | None, int]:
        last_error: Exception | None = None

        for attempt in range(1, self._process_max_attempts + 1):
            try:
                await process_func()
                if attempt > 1:
                    logger.info(
                        'Episode processing recovered for group_id=%s on attempt=%d',
                        group_id,
                        attempt,
                    )
                return True, None, attempt
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                logger.warning(
                    'Episode processing failed for group_id=%s attempt=%d/%d: %s',
                    group_id,
                    attempt,
                    self._process_max_attempts,
                    str(exc),
                )

                if attempt < self._process_max_attempts:
                    delay = self._retry_delay(attempt)
                    if delay > 0:
                        await asyncio.sleep(delay)

        return False, last_error, self._process_max_attempts

    async def _process_episode_queue(
        self,
        group_id: str,
        initial_task: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        logger.info('Starting reliable episode queue worker for group_id: %s', group_id)
        initial = initial_task

        try:
            while True:
                from_queue = initial is None
                process_func = (
                    await self._episode_queues[group_id].get() if from_queue else initial
                )
                initial = None

                try:
                    success, last_error, attempts = await self._run_with_retries(
                        group_id, process_func
                    )

                    if success:
                        self._task_metadata.pop(id(process_func), None)
                        self._blocked_groups.pop(group_id, None)
                        self._failed_tasks.pop(group_id, None)
                        self._failed_attempts.pop(group_id, None)
                        self._failed_metadata.pop(group_id, None)
                        continue

                    error_text = str(last_error) if last_error is not None else 'unknown error'
                    self._blocked_groups[group_id] = error_text
                    self._failed_tasks[group_id] = process_func
                    self._failed_attempts[group_id] = attempts
                    self._failed_metadata[group_id] = dict(
                        self._task_metadata.get(id(process_func), {})
                    )
                    logger.error(
                        'Blocking episode queue for group_id=%s after %d failed attempts; '
                        'pending=%d episode_uuid=%s error=%s',
                        group_id,
                        attempts,
                        self.get_queue_size(group_id),
                        self._failed_metadata[group_id].get('episode_uuid'),
                        error_text,
                    )
                    return
                finally:
                    if from_queue:
                        self._episode_queues[group_id].task_done()
        except asyncio.CancelledError:
            logger.info('Episode queue worker for group_id %s was cancelled', group_id)
        except Exception as exc:
            logger.error(
                'Unexpected error in reliable queue worker for group_id %s: %s',
                group_id,
                str(exc),
            )
        finally:
            self._queue_workers[group_id] = False
            logger.info('Stopped reliable episode queue worker for group_id: %s', group_id)

    async def retry_blocked_group(self, group_id: str) -> bool:
        """Retry the failed predecessor before any already-pending episodes."""
        process_func = self._failed_tasks.get(group_id)
        if process_func is None or not self.is_group_blocked(group_id):
            return False
        if self._queue_workers.get(group_id, False):
            return False

        self._blocked_groups.pop(group_id, None)
        self._failed_tasks.pop(group_id, None)
        self._failed_attempts.pop(group_id, None)
        self._failed_metadata.pop(group_id, None)

        self._queue_workers[group_id] = True
        try:
            asyncio.create_task(self._process_episode_queue(group_id, initial_task=process_func))
        except Exception:
            self._queue_workers[group_id] = False
            self._blocked_groups[group_id] = 'failed to schedule retry worker'
            self._failed_tasks[group_id] = process_func
            self._failed_metadata[group_id] = dict(
                self._task_metadata.get(id(process_func), {})
            )
            raise
        return True


def install_reliable_queue_patch() -> None:
    """Expose ReliableQueueService before graphiti_mcp_server imports QueueService."""
    global _installed
    if _installed:
        return

    queue_module.QueueService = ReliableQueueService  # type: ignore[assignment]
    queue_module.EpisodeQueueBlockedError = EpisodeQueueBlockedError  # type: ignore[attr-defined]
    _installed = True
