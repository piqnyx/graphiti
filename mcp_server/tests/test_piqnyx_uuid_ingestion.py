"""Regression tests for piqnyx caller-visible episode UUID behavior."""

import inspect
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest

from graphiti_core import Graphiti
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.nodes import EpisodeType, EpisodicNode

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from piqnyx_uuid_tool_patch import wrap_add_memory_with_uuid  # noqa: E402


@pytest.mark.asyncio
async def test_add_memory_wrapper_generates_and_returns_uuid():
    seen: dict[str, str | None] = {}

    async def fake_add_memory(name: str, episode_body: str, uuid: str | None = None):
        seen['uuid'] = uuid
        return {'message': f"Episode '{name}' queued"}

    wrapped = wrap_add_memory_with_uuid(fake_add_memory)
    result = await wrapped(name='episode-1', episode_body='body')

    generated = result['uuid']
    UUID(generated)
    assert seen['uuid'] == generated
    assert inspect.signature(wrapped) == inspect.signature(fake_add_memory)


@pytest.mark.asyncio
async def test_add_memory_wrapper_preserves_caller_uuid():
    async def fake_add_memory(name: str, episode_body: str, uuid: str | None = None):
        return {'message': 'queued', 'seen_uuid': uuid}

    wrapped = wrap_add_memory_with_uuid(fake_add_memory)
    result = await wrapped(name='episode-1', episode_body='body', uuid='caller-uuid')

    assert result['uuid'] == 'caller-uuid'
    assert result['seen_uuid'] == 'caller-uuid'


@pytest.mark.asyncio
async def test_add_memory_wrapper_does_not_attach_uuid_to_error():
    async def fake_add_memory(name: str, episode_body: str, uuid: str | None = None):
        return {'error': 'queue full'}

    wrapped = wrap_add_memory_with_uuid(fake_add_memory)
    result = await wrapped(name='episode-1', episode_body='body')

    assert result == {'error': 'queue full'}


@pytest.mark.asyncio
async def test_new_caller_uuid_constructs_unsaved_episode_with_requested_uuid(monkeypatch):
    async def missing_episode(_driver, uuid):
        raise NodeNotFoundError(uuid)

    monkeypatch.setattr(EpisodicNode, 'get_by_uuid', missing_episode)

    reference_time = datetime(2026, 8, 15, 1, 2, 3, tzinfo=timezone.utc)
    created_at = datetime(2026, 8, 15, 1, 2, 4, tzinfo=timezone.utc)
    episode = await Graphiti._get_or_create_caller_episode(
        driver=object(),
        uuid='reserved-uuid',
        name='episode-1',
        group_id='main',
        source=EpisodeType.text,
        episode_body='body',
        source_description='test',
        created_at=created_at,
        reference_time=reference_time,
    )

    assert episode.uuid == 'reserved-uuid'
    assert episode.name == 'episode-1'
    assert episode.group_id == 'main'
    assert episode.content == 'body'
    assert episode.created_at == created_at
    assert episode.valid_at == reference_time


@pytest.mark.asyncio
async def test_existing_caller_uuid_keeps_upstream_lookup_semantics(monkeypatch):
    existing = EpisodicNode(
        uuid='existing-uuid',
        name='existing',
        group_id='main',
        source=EpisodeType.text,
        source_description='test',
        content='existing body',
        valid_at=datetime(2026, 8, 15, tzinfo=timezone.utc),
    )

    async def get_existing(_driver, uuid):
        assert uuid == 'existing-uuid'
        return existing

    monkeypatch.setattr(EpisodicNode, 'get_by_uuid', get_existing)

    result = await Graphiti._get_or_create_caller_episode(
        driver=object(),
        uuid='existing-uuid',
        name='replacement name',
        group_id='main',
        source=EpisodeType.json,
        episode_body='replacement body',
        source_description='replacement description',
        created_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
        reference_time=datetime(2026, 8, 16, tzinfo=timezone.utc),
    )

    assert result is existing


@pytest.mark.asyncio
async def test_missing_uuid_uses_normal_generated_episode_uuid(monkeypatch):
    async def must_not_lookup(_driver, uuid):
        raise AssertionError(f'unexpected UUID lookup: {uuid}')

    monkeypatch.setattr(EpisodicNode, 'get_by_uuid', must_not_lookup)

    episode = await Graphiti._get_or_create_caller_episode(
        driver=object(),
        uuid=None,
        name='episode-1',
        group_id='main',
        source=EpisodeType.text,
        episode_body='body',
        source_description='test',
        created_at=datetime(2026, 8, 15, tzinfo=timezone.utc),
        reference_time=datetime(2026, 8, 15, tzinfo=timezone.utc),
    )

    UUID(episode.uuid)
