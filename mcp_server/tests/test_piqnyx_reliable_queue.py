"""Regression tests for piqnyx durable-caller / transient-server queue semantics."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from piqnyx_reliable_queue import (  # noqa: E402
    DEFAULT_RETRY_MAX_SECONDS,
    ReliableQueueService,
)


async def _wait_until(predicate, timeout: float = 1.0) -> None:
    async def waiter():
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(waiter(), timeout=timeout)


@pytest.mark.asyncio
async def test_failed_head_retries_past_old_terminal_limit_before_successor_runs():
    service = ReliableQueueService(
        max_size_per_group=10,
        # Legacy setting is intentionally ignored. The old implementation stopped
        # after this many attempts and stranded the only failed task in RAM.
        process_max_attempts=1,
        retry_base_seconds=0,
        retry_max_seconds=0,
    )

    attempts = 0
    allow_success = False
    successor_ran = asyncio.Event()

    async def predecessor():
        nonlocal attempts
        attempts += 1
        if not allow_success:
            raise RuntimeError('provider unavailable')

    async def successor():
        successor_ran.set()

    await service.add_episode_task('main', predecessor)
    await service.add_episode_task('main', successor)

    await _wait_until(lambda: attempts >= 5)
    assert successor_ran.is_set() is False
    assert service.is_group_blocked('main') is False
    assert service.get_failure_status('main')['attempts'] >= 5

    allow_success = True
    await asyncio.wait_for(successor_ran.wait(), timeout=1)
    assert attempts >= 5


@pytest.mark.asyncio
async def test_status_exposes_active_identity_and_pending_uuid_without_content():
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowGraphiti:
        async def add_episode(self, **kwargs):
            if kwargs['uuid'] == 'episode-1':
                started.set()
                await release.wait()

    service = ReliableQueueService(
        max_size_per_group=10,
        retry_base_seconds=0,
        retry_max_seconds=0,
    )
    await service.initialize(SlowGraphiti())

    await service.add_episode(
        group_id='main',
        name='batch-1',
        content='secret conversation body one',
        source_description='OpenClaw conversation batch',
        episode_type='json',
        entity_types=None,
        uuid='episode-1',
        saga='session-1',
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    await service.add_episode(
        group_id='main',
        name='batch-2',
        content='secret conversation body two',
        source_description='OpenClaw conversation batch',
        episode_type='json',
        entity_types=None,
        uuid='episode-2',
        saga='session-1',
    )

    status = service.get_failure_status('main')
    assert status == {
        'group_id': 'main',
        'blocked': False,
        'attempts': 0,
        'last_error': None,
        'failure_kind': None,
        'retry_in_seconds': None,
        'pending': 1,
        'worker_running': True,
        'episode_uuid': 'episode-1',
        'episode_name': 'batch-1',
        'saga': 'session-1',
        'queued_episode_uuids': ['episode-2'],
    }
    assert 'secret' not in repr(status)

    release.set()
    await _wait_until(lambda: service.get_queue_size('main') == 0)


@pytest.mark.asyncio
async def test_duplicate_uuid_is_not_enqueued_while_active_or_pending():
    current_started = asyncio.Event()
    release_current = asyncio.Event()
    calls: list[str] = []

    class SlowGraphiti:
        async def add_episode(self, **kwargs):
            uuid = kwargs['uuid']
            calls.append(uuid)
            if uuid == 'episode-1':
                current_started.set()
                await release_current.wait()

    service = ReliableQueueService(max_size_per_group=10, retry_base_seconds=0, retry_max_seconds=0)
    await service.initialize(SlowGraphiti())

    common = {
        'group_id': 'main',
        'source_description': 'OpenClaw conversation batch',
        'episode_type': 'json',
        'entity_types': None,
        'saga': 'session-1',
    }

    await service.add_episode(name='batch-1', content='one', uuid='episode-1', **common)
    await asyncio.wait_for(current_started.wait(), timeout=1)

    # Lost caller response, same UUID comes back while it is already running.
    duplicate_active_position = await service.add_episode(
        name='batch-1', content='one', uuid='episode-1', **common
    )
    assert duplicate_active_position == 0
    assert service.get_queue_size('main') == 0

    await service.add_episode(name='batch-2', content='two', uuid='episode-2', **common)
    assert service.get_queue_size('main') == 1

    # Same for an item already waiting behind the active head.
    duplicate_pending_position = await service.add_episode(
        name='batch-2', content='two', uuid='episode-2', **common
    )
    assert duplicate_pending_position == 1
    assert service.get_queue_size('main') == 1

    release_current.set()
    await _wait_until(lambda: calls == ['episode-1', 'episode-2'])


@pytest.mark.asyncio
async def test_failures_are_isolated_per_group():
    service = ReliableQueueService(
        max_size_per_group=10,
        retry_base_seconds=0.001,
        retry_max_seconds=0.001,
    )

    main_attempts = 0
    release_main = False
    igor_ran = asyncio.Event()

    async def fail_main():
        nonlocal main_attempts
        main_attempts += 1
        if not release_main:
            raise RuntimeError('main provider failed')

    async def run_igor():
        igor_ran.set()

    await service.add_episode_task('main', fail_main)
    await service.add_episode_task('igor', run_igor)

    await asyncio.wait_for(igor_ran.wait(), timeout=1)
    await _wait_until(lambda: main_attempts >= 3)
    assert service.get_failure_status('main')['last_error'] == 'main provider failed'
    assert service.get_failure_status('igor')['last_error'] is None

    release_main = True
    await _wait_until(lambda: service.get_failure_status('main')['episode_uuid'] is None)


def test_retry_policy_distinguishes_expensive_and_transient_failures():
    service = ReliableQueueService(
        retry_base_seconds=2,
        retry_max_seconds=3600,
    )

    OutputLimitError = type('OutputLimitError', (RuntimeError,), {})
    RateLimitError = type('RateLimitError', (RuntimeError,), {})
    APIConnectionError = type('APIConnectionError', (RuntimeError,), {})

    assert service._retry_delay(1, OutputLimitError('budget exhausted')) == (
        'output_limit',
        300.0,
    )
    assert service._retry_delay(1, RateLimitError('rate limit')) == ('rate_limit', 30.0)
    assert service._retry_delay(1, APIConnectionError('connection reset')) == (
        'provider_unavailable',
        10.0,
    )

    payment_error = RuntimeError('402 payment required: insufficient balance')
    payment_error.status_code = 402  # type: ignore[attr-defined]
    assert service._retry_delay(1, payment_error) == ('credentials_or_balance', 300.0)

    # Ordinary bugs still back off exponentially rather than being treated as a
    # provider outage. They are retained by the durable caller exactly the same way.
    assert service._retry_delay(1, RuntimeError('boom')) == ('unexpected', 2.0)
    assert service._retry_delay(5, RuntimeError('boom')) == ('unexpected', 32.0)


def test_retry_delay_saturates_without_overflow_after_extreme_outage():
    service = ReliableQueueService(retry_base_seconds=2, retry_max_seconds=900)

    assert service._retry_delay(1_000_000, RuntimeError('still unavailable')) == (
        'unexpected',
        900.0,
    )


def test_default_retry_ceiling_is_fifteen_minutes():
    assert DEFAULT_RETRY_MAX_SECONDS == 900.0
    service = ReliableQueueService()
    assert service._retry_max_seconds == 900.0


def test_retry_configuration_validation_and_legacy_attempt_limit_is_ignored():
    # Old deployments may still carry EPISODE_PROCESS_MAX_ATTEMPTS. It no longer
    # controls correctness and even zero is harmless because the parameter is ignored.
    ReliableQueueService(process_max_attempts=0)

    with pytest.raises(ValueError, match='EPISODE_PROCESS_RETRY_BASE_SECONDS'):
        ReliableQueueService(retry_base_seconds=-1)

    with pytest.raises(ValueError, match='EPISODE_PROCESS_RETRY_MAX_SECONDS'):
        ReliableQueueService(retry_base_seconds=2, retry_max_seconds=1)

    with pytest.raises(ValueError, match='EPISODE_PROCESS_RETRY_BASE_SECONDS'):
        ReliableQueueService(retry_base_seconds=float('inf'))

    with pytest.raises(ValueError, match='EPISODE_PROCESS_RETRY_MAX_SECONDS'):
        ReliableQueueService(retry_max_seconds=float('nan'))
