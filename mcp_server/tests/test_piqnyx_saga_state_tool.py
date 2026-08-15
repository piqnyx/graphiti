from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import piqnyx_saga_state_tool as patch


class FakeMcp:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def register(fn):
            self.tools[fn.__name__] = fn
            return fn

        return register


class FakeDriver:
    def __init__(self):
        self.database = None

    def clone(self, *, database):
        cloned = FakeDriver()
        cloned.database = database
        return cloned


@pytest.mark.asyncio
async def test_get_saga_uses_group_scoped_driver_and_returns_persisted_state(monkeypatch):
    driver = FakeDriver()
    client = SimpleNamespace(driver=driver)

    class FakeService:
        async def get_client(self):
            return client

    server = SimpleNamespace(
        mcp=FakeMcp(),
        graphiti_service=FakeService(),
        config=SimpleNamespace(graphiti=SimpleNamespace(group_id=None)),
        logger=SimpleNamespace(error=lambda _message: None),
    )

    seen = {}
    discovered = SimpleNamespace(uuid="saga-uuid", name="session-1")
    hydrated = SimpleNamespace(
        uuid="saga-uuid",
        name="session-1",
        group_id="main",
        created_at=datetime(2026, 8, 15, tzinfo=timezone.utc),
        summary="summary",
        first_episode_uuid="ep-1",
        last_episode_uuid="ep-6",
    )

    async def fake_get_by_group_ids(_cls, scoped_driver, group_ids, *args, **kwargs):
        seen["group_database"] = scoped_driver.database
        seen["group_ids"] = group_ids
        return [discovered]

    async def fake_get_by_uuid(_cls, scoped_driver, uuid):
        seen["uuid_database"] = scoped_driver.database
        seen["uuid"] = uuid
        return hydrated

    monkeypatch.setattr(patch.SagaNode, "get_by_group_ids", classmethod(fake_get_by_group_ids))
    monkeypatch.setattr(patch.SagaNode, "get_by_uuid", classmethod(fake_get_by_uuid))

    patch.install_get_saga_tool(server)
    result = await server.mcp.tools["get_saga"]("session-1", "main")

    assert seen == {
        "group_database": "main",
        "group_ids": ["main"],
        "uuid_database": "main",
        "uuid": "saga-uuid",
    }
    assert result["uuid"] == "saga-uuid"
    assert result["name"] == "session-1"
    assert result["group_id"] == "main"
    assert result["first_episode_uuid"] == "ep-1"
    assert result["last_episode_uuid"] == "ep-6"
    assert result["summary"] == "summary"


@pytest.mark.asyncio
async def test_get_saga_returns_error_when_name_is_missing(monkeypatch):
    client = SimpleNamespace(driver=FakeDriver())

    class FakeService:
        async def get_client(self):
            return client

    server = SimpleNamespace(
        mcp=FakeMcp(),
        graphiti_service=FakeService(),
        config=SimpleNamespace(graphiti=SimpleNamespace(group_id=None)),
        logger=SimpleNamespace(error=lambda _message: None),
    )

    async def fake_get_by_group_ids(_cls, _driver, _group_ids, *args, **kwargs):
        return []

    monkeypatch.setattr(patch.SagaNode, "get_by_group_ids", classmethod(fake_get_by_group_ids))

    patch.install_get_saga_tool(server)
    result = await server.mcp.tools["get_saga"]("missing", "main")

    assert "No saga named 'missing'" in result["error"]
