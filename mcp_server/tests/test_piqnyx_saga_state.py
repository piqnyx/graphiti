"""Regression tests for piqnyx Saga state preservation."""

from datetime import datetime, timezone

import pytest

from graphiti_core import Graphiti
from graphiti_core.nodes import SagaNode


class ExistingSagaDriver:
    async def execute_query(self, query, **kwargs):
        assert kwargs['name'] == 'session-1'
        assert kwargs['group_id'] == 'main'
        return ([{'uuid': 'saga-uuid'}], None, None)


@pytest.mark.asyncio
async def test_existing_saga_is_loaded_through_canonical_node_loader(monkeypatch):
    expected = SagaNode(
        uuid='saga-uuid',
        name='session-1',
        group_id='main',
        created_at=datetime(2026, 8, 15, tzinfo=timezone.utc),
        first_episode_uuid='episode-1',
        last_episode_uuid='episode-5',
        summary='existing summary',
    )

    async def fake_get_by_uuid(cls, driver, uuid):
        assert uuid == 'saga-uuid'
        return expected

    monkeypatch.setattr(SagaNode, 'get_by_uuid', classmethod(fake_get_by_uuid))

    graphiti = object.__new__(Graphiti)
    result = await graphiti._get_or_create_saga(
        'session-1',
        'main',
        datetime(2026, 8, 16, tzinfo=timezone.utc),
        driver=ExistingSagaDriver(),
    )

    assert result is expected
    assert result.first_episode_uuid == 'episode-1'
    assert result.last_episode_uuid == 'episode-5'
    assert result.summary == 'existing summary'
