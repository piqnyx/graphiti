"""Regression tests for crash-safe ordered Falkor ingestion."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType, EpisodicNode, SagaNode
from graphiti_core.piqnyx_graphiti import Graphiti


NOW = datetime(2026, 8, 18, 1, 2, 3, tzinfo=timezone.utc)


class FakeNode:
    def __init__(self, uuid: str, name: str):
        self.uuid = uuid
        self.name = name
        self.group_id = "main"
        self.labels = ["Entity"]
        self.summary = ""
        self.created_at = NOW
        self.name_embedding = [0.1, 0.2]
        self.attributes = {}

    async def generate_name_embedding(self, _embedder):
        raise AssertionError("embedding must be prepared before the commit test")


class NewSagaDriver:
    provider = GraphProvider.FALKORDB

    def __init__(self, final_records=None):
        self.calls = []
        self.final_records = final_records

    async def execute_query(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if "MATCH (s:Saga {name: $name, group_id: $group_id})" in query:
            return ([], None, None)
        records = self.final_records
        if records is None:
            records = [
                {
                    "committed_uuid": kwargs["episode_uuid"],
                    "first_episode_uuid": kwargs["episode_uuid"],
                    "last_episode_uuid": kwargs["episode_uuid"],
                }
            ]
        return (records, None, None)


class ExistingSagaDriver:
    provider = GraphProvider.FALKORDB

    def __init__(self, saga: SagaNode, final_records=None):
        self.saga = saga
        self.calls = []
        self.final_records = final_records

    async def execute_query(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if "MATCH (s:Saga {name: $name, group_id: $group_id})" in query:
            return ([{"uuid": self.saga.uuid}], None, None)
        records = self.final_records
        if records is None:
            records = [
                {
                    "committed_uuid": kwargs["episode_uuid"],
                    "first_episode_uuid": self.saga.first_episode_uuid,
                    "last_episode_uuid": kwargs["episode_uuid"],
                }
            ]
        return (records, None, None)


def make_graphiti() -> Graphiti:
    graphiti = object.__new__(Graphiti)
    graphiti.store_raw_episode_content = True
    graphiti.max_coroutines = 10
    return graphiti


def make_episode(uuid: str = "episode-1", name: str = "dialog-1") -> EpisodicNode:
    return EpisodicNode(
        uuid=uuid,
        name=name,
        group_id="main",
        source=EpisodeType.json,
        source_description="OpenClaw conversation batch",
        content='{"messages":[{"role":"user","text":"hello"}]}',
        created_at=NOW,
        valid_at=NOW,
    )


@pytest.mark.asyncio
async def test_falkor_saga_commit_is_one_mutating_query():
    driver = NewSagaDriver()
    clients = SimpleNamespace(driver=driver, embedder=object())
    graphiti = make_graphiti()
    episode = make_episode()
    left = FakeNode("entity-left", "Left")
    right = FakeNode("entity-right", "Right")
    fact = EntityEdge(
        uuid="fact-1",
        group_id="main",
        source_node_uuid=left.uuid,
        target_node_uuid=right.uuid,
        name="KNOWS",
        fact="Left knows Right",
        fact_embedding=[0.3, 0.4],
        episodes=[episode.uuid],
        created_at=NOW,
        valid_at=NOW,
        reference_time=NOW,
    )

    episodic_edges, committed = await graphiti._process_episode_data(
        episode,
        [left, right],
        [fact],
        NOW,
        "main",
        saga="session-1",
        saga_previous_episode_uuid=None,
        node_episode_index_map=None,
        clients=clients,
    )

    assert committed.uuid == episode.uuid
    assert len(episodic_edges) == 2

    # One read resolves the Saga; every graph mutation is then one Falkor query.
    assert len(driver.calls) == 2
    query, params = driver.calls[-1]
    assert "MERGE (ep:Episodic" in query
    assert "MERGE (n:Entity" in query
    assert "MERGE (source)-[fact:RELATES_TO" in query
    assert "MERGE (mention_episode)-[mention:MENTIONS" in query
    assert "MERGE (saga)-[has_episode:HAS_EPISODE" in query
    assert "SET saga = $saga_data" in query
    assert "NEXT_EPISODE" not in query
    assert params["episodes"][0]["uuid"] == episode.uuid
    assert params["saga_data"]["first_episode_uuid"] == episode.uuid
    assert params["saga_data"]["last_episode_uuid"] == episode.uuid


@pytest.mark.asyncio
async def test_non_first_commit_checks_predecessor_before_any_mutation(monkeypatch):
    saga = SagaNode(
        uuid="saga-1",
        name="session-1",
        group_id="main",
        created_at=NOW,
        first_episode_uuid="episode-1",
        last_episode_uuid="episode-1",
    )
    driver = ExistingSagaDriver(saga)
    clients = SimpleNamespace(driver=driver, embedder=object())
    graphiti = make_graphiti()
    episode = make_episode("episode-2", "dialog-2")

    async def load_saga(_driver, uuid):
        assert uuid == saga.uuid
        return saga

    monkeypatch.setattr(SagaNode, "get_by_uuid", load_saga)

    await graphiti._process_episode_data(
        episode,
        [],
        [],
        NOW,
        "main",
        saga="session-1",
        saga_previous_episode_uuid="episode-1",
        clients=clients,
    )

    query, params = driver.calls[-1]
    prerequisite = query.index("MATCH (saga:Saga")
    episode_write = query.index("MERGE (ep:Episodic")
    assert prerequisite < episode_write
    assert "saga.last_episode_uuid = $previous_episode_uuid" in query
    assert "MATCH (saga)-[:HAS_EPISODE]->(previous)" in query
    assert "OPTIONAL MATCH (previous)-[:NEXT_EPISODE]->(existing_successor:Episodic)" in query
    assert "MERGE (previous)-[next_episode:NEXT_EPISODE" in query
    assert params["previous_episode_uuid"] == "episode-1"


@pytest.mark.asyncio
async def test_failed_atomic_precondition_is_a_hard_failure(monkeypatch):
    saga = SagaNode(
        uuid="saga-1",
        name="session-1",
        group_id="main",
        created_at=NOW,
        first_episode_uuid="episode-1",
        last_episode_uuid="episode-1",
    )
    driver = ExistingSagaDriver(saga, final_records=[])
    clients = SimpleNamespace(driver=driver, embedder=object())
    graphiti = make_graphiti()

    async def load_saga(_driver, uuid):
        return saga

    monkeypatch.setattr(SagaNode, "get_by_uuid", load_saga)

    with pytest.raises(RuntimeError, match="chronology was not mutated"):
        await graphiti._process_episode_data(
            make_episode("episode-2", "dialog-2"),
            [],
            [],
            NOW,
            "main",
            saga="session-1",
            saga_previous_episode_uuid="episode-1",
            clients=clients,
        )


@pytest.mark.asyncio
async def test_existing_committed_tail_is_idempotent_replay(monkeypatch):
    graphiti = make_graphiti()
    episode = make_episode("episode-9", "dialog-9")

    class Driver:
        async def execute_query(self, query, **kwargs):
            if "different UUID" in query:
                raise AssertionError("query text is not an error message")
            if "{name: $saga_name" in query and "{name: $episode_name}" in query:
                return ([], None, None)
            return (
                [{"first_episode_uuid": "episode-1", "last_episode_uuid": episode.uuid}],
                None,
                None,
            )

    async def load_episode(_driver, uuid):
        assert uuid == episode.uuid
        return episode

    monkeypatch.setattr(EpisodicNode, "get_by_uuid", load_episode)
    result = await graphiti._existing_committed_caller_episode(
        driver=Driver(),
        uuid=episode.uuid,
        name=episode.name,
        group_id="main",
        saga="session-1",
    )
    assert result is episode


@pytest.mark.asyncio
async def test_existing_detached_uuid_fails_closed(monkeypatch):
    graphiti = make_graphiti()
    episode = make_episode("episode-bad", "dialog-bad")

    class Driver:
        async def execute_query(self, query, **kwargs):
            return ([], None, None)

    async def load_episode(_driver, uuid):
        return episode

    monkeypatch.setattr(EpisodicNode, "get_by_uuid", load_episode)
    with pytest.raises(RuntimeError, match="partial or divergent episode"):
        await graphiti._existing_committed_caller_episode(
            driver=Driver(),
            uuid=episode.uuid,
            name=episode.name,
            group_id="main",
            saga="session-1",
        )
