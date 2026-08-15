"""Regression tests for piqnyx retry-and-block episode queue behavior."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from piqnyx_reliable_queue import (  # noqa: E402
    EpisodeQueueBlockedError,
    ReliableQueueService,
)


async def _wait_until(predicate, timeout: float = 1.0) -> None:
    async def waiter():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(waiter(), timeout=timeout)


@pytest.mark.asyncio
async def test_current_episode_retries_before_next_episode_runs():
    service = ReliableQueueService(
        max_size_per_group=10,
        process_max_attempts=3,
        retry_base_seconds=0,
        retry_max_seconds=0,
    )

    attempts = 0
    events: list[str] = []

    async def transient_episode():
        nonlocal attempts
        attempts += 1
        events.append(f'first:{attempts}')
        if attempts < 3:
            raise RuntimeError('transient')
        events.append('first:success')

    async def terminal_second_episode():
        events.append('second')
        raise RuntimeError('terminal second')

    await service.add_episode_task('main', transient_episode)
    await service.add_episode_task('main', terminal_second_episode)

    await _wait_until(lambda: service.is_group_blocked('main'))

    assert attempts == 3
    assert events[:4] == ['first:1', 'first:2', 'first:3', 'first:success']
    assert events[4:] == ['second', 'second', 'second']
    assert service.get_failure_status('main')['attempts'] == 3


@pytest.mark.asyncio
async def test_terminal_failure_blocks_group_and_preserves_pending_fifo():
    service = ReliableQueueService(
        max_size_per_group=10,
        process_max_attempts=2,
        retry_base_seconds=0,
        retry_max_seconds=0,
    )

    pending_ran = False

    async def failed_predecessor():
        raise RuntimeError('broken predecessor')

    async def pending_successor():
        nonlocal pending_ran
        pending_ran = True

    await service.add_episode_task('main', failed_predecessor)
    await service.add_episode_task('main', pending_successor)

    await _wait_until(lambda: service.is_group_blocked('main'))

    assert pending_ran is False
    assert service.get_queue_size('main') == 1
    assert service.get_failure_status('main') == {
        'group_id': 'main',
        'blocked': True,
        'attempts': 2,
        'last_error': 'broken predecessor',
        'pending': 1,
        'episode_uuid': None,
        'episode_name': None,
        'saga': None,
    }

    with pytest.raises(EpisodeQueueBlockedError, match='failed predecessor'):
        await service.add_episode_task('main', pending_successor)


@pytest.mark.asyncio
async def test_blocked_status_identifies_failed_episode_without_content():
    class FailingGraphiti:
        async def add_episode(self, **_kwargs):
            raise RuntimeError('provider exploded')

    service = ReliableQueueService(
        max_size_per_group=10,
        process_max_attempts=1,
        retry_base_seconds=0,
        retry_max_seconds=0,
    )
    await service.initialize(FailingGraphiti())

    await service.add_episode(
        group_id='main',
        name='6bc2a77c6957-7',
        content='secret conversation body must not appear in status',
        source_description='OpenClaw conversation batch',
        episode_type='json',
        entity_types=None,
        uuid='episode-uuid-7',
        saga='agent:main:web:1d8d5bfd-de0e-4877-82cb-6bc2a77c6957',
    )
    await _wait_until(lambda: service.is_group_blocked('main'))

    assert service.get_failure_status('main') == {
        'group_id': 'main',
        'blocked': True,
        'attempts': 1,
        'last_error': 'provider exploded',
        'pending': 0,
        'episode_uuid': 'episode-uuid-7',
        'episode_name': '6bc2a77c6957-7',
        'saga': 'agent:main:web:1d8d5bfd-de0e-4877-82cb-6bc2a77c6957',
    }


@pytest.mark.asyncio
async def test_blocking_is_isolated_per_group():
    service = ReliableQueueService(
        max_size_per_group=10,
        process_max_attempts=1,
        retry_base_seconds=0,
        retry_max_seconds=0,
    )

    igor_ran = asyncio.Event()

    async def fail_main():
        raise RuntimeError('main failed')

    async def run_igor_then_block():
        igor_ran.set()
        raise RuntimeError('igor terminal')

    await service.add_episode_task('main', fail_main)
    await service.add_episode_task('igor', run_igor_then_block)

    await asyncio.wait_for(igor_ran.wait(), timeout=1)
    await _wait_until(lambda: service.is_group_blocked('main'))

    assert service.is_group_blocked('main') is True
    assert service.is_group_blocked('igor') is True


@pytest.mark.asyncio
async def test_operator_retry_runs_failed_predecessor_before_pending_successor():
    service = ReliableQueueService(
        max_size_per_group=10,
        process_max_attempts=1,
        retry_base_seconds=0,
        retry_max_seconds=0,
    )

    allow_predecessor = False
    order: list[str] = []

    async def predecessor():
        order.append('predecessor')
        if not allow_predecessor:
            raise RuntimeError('not yet')

    async def successor_then_stop():
        order.append('successor')
        raise RuntimeError('stop after successor')

    await service.add_episode_task('main', predecessor)
    await service.add_episode_task('main', successor_then_stop)
    await _wait_until(lambda: service.is_group_blocked('main'))

    assert order == ['predecessor']

    allow_predecessor = True
    assert await service.retry_blocked_group('main') is True

    await _wait_until(
        lambda: service.is_group_blocked('main')
        and service.get_failure_status('main')['last_error'] == 'stop after successor'
    )

    assert order == ['predecessor', 'predecessor', 'successor']


def test_retry_configuration_validation():
    with pytest.raises(ValueError, match='EPISODE_PROCESS_MAX_ATTEMPTS'):
        ReliableQueueService(process_max_attempts=0)

    with pytest.raises(ValueError, match='EPISODE_PROCESS_RETRY_BASE_SECONDS'):
        ReliableQueueService(retry_base_seconds=-1)

    with pytest.raises(ValueError, match='EPISODE_PROCESS_RETRY_MAX_SECONDS'):
        ReliableQueueService(retry_base_seconds=2, retry_max_seconds=1)
