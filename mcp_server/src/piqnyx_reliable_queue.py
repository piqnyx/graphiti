"""piqnyx sequential per-group episode executor.

Durability belongs to the caller-side spool. This server queue has one job: execute
accepted work in FIFO order while the process is alive. A failed head is retried in
place with capped exponential backoff and later episodes never leapfrog it.

There is deliberately no terminal "blocked" state holding the only copy of a task
in RAM. If this process restarts, the caller notices that its episode is absent and
resubmits the same durable queue head. Repeated submissions of an episode UUID that
is already active or queued are acknowledged without enqueuing a second execution.
"""

import asyncio
import logging
import math
import os
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from datetime import datetime
from typing import Any

from services import queue_service as queue_module

logger = logging.getLogger(__name__)

_BaseQueueService = queue_module.QueueService
_installed = False

DEFAULT_RETRY_BASE_SECONDS = 2.0
DEFAULT_RETRY_MAX_SECONDS = 900.0


class EpisodeQueueBlockedError(RuntimeError):
    """Compatibility alias retained for callers compiled against the old overlay."""


class ReliableQueueService(_BaseQueueService):
    """FIFO queue that retries the current head until it succeeds or the process stops."""

    def __init__(
        self,
        max_size_per_group: int | None = None,
        process_max_attempts: int | None = None,  # kept for config compatibility; ignored
        retry_base_seconds: float | None = None,
        retry_max_seconds: float | None = None,
    ):
        super().__init__(max_size_per_group=max_size_per_group)
        del process_max_attempts

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

        self._task_metadata: dict[int, dict[str, Any]] = {}
        self._enqueue_metadata: ContextVar[dict[str, Any] | None] = ContextVar(
            f'piqnyx_queue_metadata_{id(self)}', default=None
        )
        self._current_task: dict[str, Callable[[], Awaitable[None]]] = {}
        self._current_attempts: dict[str, int] = {}
        self._current_errors: dict[str, str] = {}
        self._current_failure_kind: dict[str, str] = {}
        self._current_retry_at: dict[str, float] = {}

    @staticmethod
    def _read_nonnegative_float(value: float | None, env_name: str, default: float) -> float:
        raw: Any = value if value is not None else os.getenv(env_name, str(default))
        try:
            parsed = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{env_name} must be a finite non-negative number') from exc
        if not math.isfinite(parsed) or parsed < 0:
            raise ValueError(f'{env_name} must be a finite non-negative number')
        return parsed

    def _ensure_worker(self, group_id: str) -> None:
        """Start a worker for queued work when bookkeeping says none is alive.

        The base service stores only a boolean, not the worker Task. An unexpected
        exception in the worker therefore used to strand queued UUIDs forever: the
        caller saw its UUID in get_queue_status and correctly refused to submit a
        duplicate, while no worker existed to consume it. Status/duplicate checks
        now self-heal that state on the event-loop thread.
        """
        queue = self._episode_queues.get(group_id)
        if queue is None or queue.empty() or self._queue_workers.get(group_id, False):
            return
        self._queue_workers[group_id] = True
        try:
            asyncio.get_running_loop().create_task(self._process_episode_queue(group_id))
        except Exception:
            self._queue_workers[group_id] = False
            raise
        logger.warning(
            'Restarted missing reliable episode queue worker group_id=%s pending=%d',
            group_id,
            queue.qsize(),
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

    def _metadata_for_current(self, group_id: str) -> dict[str, Any]:
        process_func = self._current_task.get(group_id)
        return self._task_metadata.get(id(process_func), {}) if process_func else {}

    def _queued_episode_uuids(self, group_id: str) -> list[str]:
        queue = self._episode_queues.get(group_id)
        if queue is None:
            return []
        queued = getattr(queue, '_queue', ())
        result: list[str] = []
        for process_func in list(queued):
            metadata = self._task_metadata.get(id(process_func), {})
            episode_uuid = metadata.get('episode_uuid')
            if isinstance(episode_uuid, str) and episode_uuid:
                result.append(episode_uuid)
        return result

    async def add_episode_task(
        self, group_id: str, process_func: Callable[[], Awaitable[None]]
    ) -> int:
        metadata = self._enqueue_metadata.get()
        episode_uuid = metadata.get('episode_uuid') if metadata is not None else None

        if isinstance(episode_uuid, str) and episode_uuid:
            if self._metadata_for_current(group_id).get('episode_uuid') == episode_uuid:
                logger.info(
                    'Ignoring duplicate submission already processing group_id=%s episode_uuid=%s',
                    group_id,
                    episode_uuid,
                )
                return 0
            queued_uuids = self._queued_episode_uuids(group_id)
            if episode_uuid in queued_uuids:
                self._ensure_worker(group_id)
                logger.info(
                    'Ignoring duplicate submission already queued group_id=%s episode_uuid=%s',
                    group_id,
                    episode_uuid,
                )
                return queued_uuids.index(episode_uuid) + 1

        if metadata is not None:
            self._task_metadata[id(process_func)] = dict(metadata)
        try:
            return await super().add_episode_task(group_id, process_func)
        except Exception:
            self._task_metadata.pop(id(process_func), None)
            raise

    def is_group_blocked(self, group_id: str) -> bool:  # noqa: ARG002
        return False

    def get_failure_status(self, group_id: str) -> dict[str, Any]:
        self._ensure_worker(group_id)
        metadata = self._metadata_for_current(group_id)
        retry_at = self._current_retry_at.get(group_id)
        retry_in = None if retry_at is None else max(0.0, retry_at - time.monotonic())
        return {
            'group_id': group_id,
            'blocked': False,
            'attempts': self._current_attempts.get(group_id, 0),
            'last_error': self._current_errors.get(group_id),
            'failure_kind': self._current_failure_kind.get(group_id),
            'retry_in_seconds': round(retry_in, 3) if retry_in is not None else None,
            'pending': self.get_queue_size(group_id),
            'worker_running': self.is_worker_running(group_id),
            'episode_uuid': metadata.get('episode_uuid'),
            'episode_name': metadata.get('episode_name'),
            'saga': metadata.get('saga'),
            'queued_episode_uuids': self._queued_episode_uuids(group_id),
        }

    @staticmethod
    def _failure_policy(exc: Exception) -> tuple[str, float]:
        """Classify a failed processing attempt and choose a sensible retry floor.

        The durable caller never drops the head, so even configuration/billing failures
        remain retryable. The distinction only controls how aggressively we poke the
        provider. Expensive deterministic failures deliberately cool down much longer
        than a dropped TCP connection.
        """
        name = type(exc).__name__
        message = str(exc).lower()
        status_code = getattr(exc, 'status_code', None)

        if name == 'OutputLimitError':
            return 'output_limit', 300.0
        if name == 'RateLimitError' or status_code == 429 or 'rate limit' in message:
            return 'rate_limit', 30.0
        if status_code in {401, 402, 403} or any(
            marker in message
            for marker in (
                'insufficient balance',
                'insufficient credit',
                'payment required',
                'quota exhausted',
                'invalid api key',
                'authentication',
            )
        ):
            return 'credentials_or_balance', 300.0
        if (
            isinstance(status_code, int)
            and status_code >= 500
            or any(marker in name.lower() for marker in ('connection', 'timeout'))
            or any(
                marker in message
                for marker in ('connection reset', 'connection refused', 'timed out', 'temporarily unavailable')
            )
        ):
            return 'provider_unavailable', 10.0
        if name in {'JSONDecodeError', 'ValidationError'}:
            # These already went through the bounded per-call retry wrapper. Re-running
            # the entire episode immediately would mostly repeat successful earlier calls.
            return 'malformed_response', 60.0
        if name == 'EmptyResponseError':
            return 'empty_response', 10.0
        return 'unexpected', 2.0

    def _retry_delay(self, failed_attempt_number: int, exc: Exception) -> tuple[str, float]:
        failure_kind, floor = self._failure_policy(exc)
        if self._retry_base_seconds == 0 or self._retry_max_seconds == 0:
            exponential = 0.0
        else:
            # Cap the exponent before evaluating it. Computing 2**attempt first can
            # overflow after a long outage even though the final delay is capped.
            max_exponent = max(
                0,
                math.ceil(
                    math.log2(self._retry_max_seconds)
                    - math.log2(self._retry_base_seconds)
                ),
            )
            exponent = min(max(failed_attempt_number - 1, 0), max_exponent)
            exponential = min(
                math.ldexp(self._retry_base_seconds, exponent),
                self._retry_max_seconds,
            )
        delay = max(exponential, floor)
        return failure_kind, min(delay, self._retry_max_seconds)

    async def _process_episode_queue(self, group_id: str) -> None:
        logger.info('Starting reliable episode queue worker for group_id: %s', group_id)

        try:
            while True:
                process_func = await self._episode_queues[group_id].get()
                self._current_task[group_id] = process_func
                self._current_attempts[group_id] = 0
                self._current_errors.pop(group_id, None)
                self._current_failure_kind.pop(group_id, None)
                self._current_retry_at.pop(group_id, None)
                metadata = self._task_metadata.get(id(process_func), {})

                try:
                    while True:
                        try:
                            await process_func()
                            failures = self._current_attempts.get(group_id, 0)
                            if failures > 0:
                                logger.info(
                                    'Episode processing recovered for group_id=%s '
                                    'episode_uuid=%s after_failures=%d',
                                    group_id,
                                    metadata.get('episode_uuid'),
                                    failures,
                                )
                            break
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            failures = self._current_attempts.get(group_id, 0) + 1
                            self._current_attempts[group_id] = failures
                            self._current_errors[group_id] = str(exc)
                            failure_kind, delay = self._retry_delay(failures, exc)
                            self._current_failure_kind[group_id] = failure_kind
                            self._current_retry_at[group_id] = time.monotonic() + delay
                            logger.warning(
                                'Episode processing failed for group_id=%s episode_uuid=%s '
                                'attempt=%d failure_kind=%s retry_in=%.3fs error=%s',
                                group_id,
                                metadata.get('episode_uuid'),
                                failures,
                                failure_kind,
                                delay,
                                str(exc),
                            )
                            if delay > 0:
                                await asyncio.sleep(delay)
                            self._current_retry_at.pop(group_id, None)
                finally:
                    self._task_metadata.pop(id(process_func), None)
                    self._current_task.pop(group_id, None)
                    self._current_attempts.pop(group_id, None)
                    self._current_errors.pop(group_id, None)
                    self._current_failure_kind.pop(group_id, None)
                    self._current_retry_at.pop(group_id, None)
                    self._episode_queues[group_id].task_done()
        except asyncio.CancelledError:
            logger.info('Episode queue worker for group_id %s was cancelled', group_id)
        except Exception as exc:
            logger.exception(
                'Unexpected error in reliable queue worker for group_id %s: %s',
                group_id,
                str(exc),
            )
        finally:
            self._queue_workers[group_id] = False
            self._current_task.pop(group_id, None)
            self._current_attempts.pop(group_id, None)
            self._current_errors.pop(group_id, None)
            self._current_failure_kind.pop(group_id, None)
            self._current_retry_at.pop(group_id, None)
            logger.info('Stopped episode queue worker for group_id: %s', group_id)


def install_reliable_queue_patch() -> None:
    global _installed
    if _installed:
        return

    queue_module.QueueService = ReliableQueueService  # type: ignore[assignment]
    queue_module.EpisodeQueueBlockedError = EpisodeQueueBlockedError  # type: ignore[attr-defined]
    _installed = True
