"""Regression tests for physical Falkor graph isolation by group_id."""

from types import SimpleNamespace

from graphiti_core.graphiti import Graphiti


class _Clients:
    def __init__(self, driver):
        self.driver = driver

    def model_copy(self, *, update):
        return _Clients(update['driver'])


class _Driver:
    provider = 'falkordb'

    def __init__(self, database: str):
        self._database = database
        self.clone_calls: list[str] = []

    def clone(self, database: str):
        self.clone_calls.append(database)
        return _Driver(database)


class _Host:
    def __init__(self):
        self.driver = _Driver('default_db')
        self.clients = _Clients(self.driver)


def test_request_scope_uses_call_local_driver_without_rebinding_shared_state():
    host = _Host()

    main_group, main_driver, main_clients = Graphiti._resolve_request_scope(host, 'main')
    igor_group, igor_driver, igor_clients = Graphiti._resolve_request_scope(host, 'igor')

    assert main_group == 'main'
    assert igor_group == 'igor'
    assert main_driver._database == 'main'
    assert igor_driver._database == 'igor'
    assert main_clients.driver is main_driver
    assert igor_clients.driver is igor_driver

    # This is the invariant that prevents cross-agent leakage during awaits: a
    # request may clone a target graph, but may never mutate the Graphiti instance
    # shared by another concurrent request.
    assert host.driver._database == 'default_db'
    assert host.clients.driver is host.driver
    assert host.driver.clone_calls == ['main', 'igor']


def test_same_group_reuses_matching_driver_without_cross_scope_clone():
    host = _Host()
    host.driver = _Driver('main')
    host.clients = _Clients(host.driver)

    group, driver, clients = Graphiti._resolve_request_scope(host, 'main')

    assert group == 'main'
    assert driver is host.driver
    assert clients is host.clients
    assert host.driver.clone_calls == []
