"""Index building runs once per graph and connection, not once per driver (no live database).

Every ``clone()`` for a group builds a new FalkorDriver, and each new driver used to send
all index-creation commands again. Those commands are writes: on a graph busy with a long
read they waited behind it, and every query queued after them waited too. Measured on the
piqnyx main graph: a 0.1 s read took 27 s behind one such index command.

A graph is marked indexed only after its build succeeded, so a failed or cancelled build
is retried by the next driver. Drivers created while a build runs do not start another.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.driver.falkordb_driver import FalkorDriver


@pytest.fixture
def build(monkeypatch):
    # raising=False: before the fix the registries do not exist, and the tests must fail on
    # the repeated builds, not on the fixture.
    monkeypatch.setattr(FalkorDriver, '_indexed_graphs', set(), raising=False)
    monkeypatch.setattr(FalkorDriver, '_indexing_graphs', set(), raising=False)
    build = AsyncMock()
    monkeypatch.setattr(FalkorDriver, 'build_indices_and_constraints', build)
    return build


@pytest.mark.asyncio
async def test_indices_built_once_per_graph_and_connection(build):
    client = MagicMock()

    first = FalkorDriver(falkor_db=client, database='main')
    assert first._init_task is not None
    await first._init_task

    again = FalkorDriver(falkor_db=client, database='main')
    assert again._init_task is None

    other_graph = FalkorDriver(falkor_db=client, database='igor')
    assert other_graph._init_task is not None
    await other_graph._init_task

    other_connection = FalkorDriver(falkor_db=MagicMock(), database='main')
    assert other_connection._init_task is not None
    await other_connection._init_task

    assert build.await_count == 3


@pytest.mark.asyncio
async def test_failed_index_build_is_retried_by_next_driver(build):
    build.side_effect = [RuntimeError('Query timed out'), None]
    client = MagicMock()

    failed = FalkorDriver(falkor_db=client, database='main')
    with pytest.raises(RuntimeError):
        await failed._init_task

    retried = FalkorDriver(falkor_db=client, database='main')
    assert retried._init_task is not None
    await retried._init_task

    after_success = FalkorDriver(falkor_db=client, database='main')
    assert after_success._init_task is None
    assert build.await_count == 2


@pytest.mark.asyncio
async def test_drivers_created_during_a_build_do_not_start_another(build):
    release = asyncio.Event()

    async def slow_build():
        await release.wait()

    build.side_effect = slow_build
    client = MagicMock()

    drivers = [FalkorDriver(falkor_db=client, database='main') for _ in range(5)]
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(*(d._init_task for d in drivers if d._init_task is not None))

    assert build.await_count == 1
    assert FalkorDriver(falkor_db=client, database='main')._init_task is None
