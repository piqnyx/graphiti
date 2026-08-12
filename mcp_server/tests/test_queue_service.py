"""Regression tests for the MCP episode QueueService."""

import asyncio

import pytest

from services.queue_service import EpisodeQueueFullError, QueueService


@pytest.mark.asyncio
async def test_queue_is_bounded_per_group_and_fails_fast_when_full():
    service = QueueService(max_size_per_group=2)
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking_task():
        started.set()
        await release.wait()

    async def noop_task():
        return None

    await service.add_episode_task('main', blocking_task)
    await asyncio.wait_for(started.wait(), timeout=1)

    # The active task is not pending, so two additional episodes fit.
    await service.add_episode_task('main', noop_task)
    await service.add_episode_task('main', noop_task)

    assert service.get_queue_status('main') == {
        'group_id': 'main',
        'pending': 2,
        'max_pending': 2,
        'worker_running': True,
    }

    with pytest.raises(EpisodeQueueFullError, match='max pending=2'):
        await service.add_episode_task('main', noop_task)

    # A different agent/group has an independent queue budget.
    await service.add_episode_task('igor', noop_task)
    assert service.get_queue_size('igor') <= 1

    release.set()


@pytest.mark.asyncio
async def test_concurrent_enqueues_schedule_only_one_worker_per_group():
    class CountingQueueService(QueueService):
        def __init__(self):
            super().__init__(max_size_per_group=100)
            self.worker_starts = 0
            self.release = asyncio.Event()

        async def _process_episode_queue(self, group_id: str) -> None:
            self.worker_starts += 1
            await self.release.wait()

    service = CountingQueueService()

    async def noop_task():
        return None

    await asyncio.gather(
        *(service.add_episode_task('main', noop_task) for _ in range(20))
    )
    await asyncio.sleep(0)

    assert service.worker_starts == 1
    assert service.is_worker_running('main') is True
    assert service.get_queue_size('main') == 20

    service.release.set()


def test_queue_capacity_must_be_positive(monkeypatch):
    with pytest.raises(ValueError, match='positive integer'):
        QueueService(max_size_per_group=0)

    monkeypatch.setenv('EPISODE_QUEUE_MAX_SIZE', 'not-an-int')
    with pytest.raises(ValueError, match='EPISODE_QUEUE_MAX_SIZE'):
        QueueService()


def test_queue_capacity_defaults_to_100(monkeypatch):
    monkeypatch.delenv('EPISODE_QUEUE_MAX_SIZE', raising=False)
    service = QueueService()
    assert service.get_queue_capacity() == 100
