#!/usr/bin/env python3
"""The fact search must read the graph its group actually writes to."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

import graphiti_mcp_server


def _install_client(monkeypatch):
    """A client that records what the search was handed."""
    scoped_driver = Mock(name='scoped_driver')
    client = Mock()
    client.driver.clone = Mock(return_value=scoped_driver)
    client.search_ = AsyncMock(return_value=SimpleNamespace(edges=[]))

    service = Mock()
    service.get_client = AsyncMock(return_value=client)
    monkeypatch.setattr(graphiti_mcp_server, 'graphiti_service', service)
    monkeypatch.setattr(
        graphiti_mcp_server,
        'config',
        SimpleNamespace(graphiti=SimpleNamespace(group_id='main')),
        raising=False,
    )
    return client, scoped_driver


@pytest.mark.asyncio
async def test_a_named_group_is_searched_in_its_own_database(monkeypatch):
    """Episodes are written with driver.clone(database=group_id).

    Searching without that clone reads whichever database the shared driver was
    built with, and the group_id filter then matches nothing there -- recall for
    every agent but the configured one comes back empty, and says nothing about it.
    """
    client, scoped_driver = _install_client(monkeypatch)

    await graphiti_mcp_server.search_memory_facts(query='кукуруза', group_ids='igor')

    client.driver.clone.assert_called_once_with(database='igor')
    assert client.search_.await_args.kwargs['driver'] is scoped_driver
    assert client.search_.await_args.kwargs['group_ids'] == ['igor']


@pytest.mark.asyncio
async def test_several_groups_keep_the_shared_driver(monkeypatch):
    """A database is one graph, so a search across two cannot be scoped to either."""
    client, _ = _install_client(monkeypatch)

    await graphiti_mcp_server.search_memory_facts(query='кукуруза', group_ids=['igor', 'red'])

    client.driver.clone.assert_not_called()
    assert client.search_.await_args.kwargs['driver'] is None


@pytest.mark.asyncio
async def test_the_shared_recipe_survives_a_search(monkeypatch):
    """The recipe is a module-level object every concurrent request reads.

    Assigning the caller's limit onto it lets a manual search asking for fifty
    rewrite the limit a recall asking for eight is about to use.
    """
    client, _ = _install_client(monkeypatch)
    before = EDGE_HYBRID_SEARCH_RRF.limit

    await graphiti_mcp_server.search_memory_facts(query='кукуруза', group_ids='main', max_facts=50)

    assert EDGE_HYBRID_SEARCH_RRF.limit == before
    assert client.search_.await_args.kwargs['config'].limit == 50
