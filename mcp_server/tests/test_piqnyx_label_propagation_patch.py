from __future__ import annotations

from types import SimpleNamespace

import pytest

import piqnyx_label_propagation_patch as patch


def neighbours(*pairs):
    return [SimpleNamespace(node_uuid=uuid, edge_count=count) for uuid, count in pairs]


def test_a_converging_graph_is_clustered_exactly_as_before():
    # Two triangles joined by nothing: the answer is unambiguous, so the ceiling
    # must not change it.
    projection = {
        'a': neighbours(('b', 2), ('c', 2)),
        'b': neighbours(('a', 2), ('c', 2)),
        'c': neighbours(('a', 2), ('b', 2)),
        'x': neighbours(('y', 2), ('z', 2)),
        'y': neighbours(('x', 2), ('z', 2)),
        'z': neighbours(('x', 2), ('y', 2)),
    }

    clusters = [sorted(cluster) for cluster in patch.bounded_label_propagation(projection)]
    assert sorted(clusters) == [['a', 'b', 'c'], ['x', 'y', 'z']]


def test_an_oscillating_graph_returns_instead_of_spinning_forever():
    # A bipartite pair swaps labels every sweep and never settles: this is the
    # shape that pinned a core with no way out.
    projection = {
        'left': neighbours(('right', 5)),
        'right': neighbours(('left', 5)),
    }

    clusters = patch.bounded_label_propagation(projection)

    # It returns, and every node is accounted for exactly once — a partition is
    # valid after any sweep, which is why stopping early is sound.
    assigned = sorted(uuid for cluster in clusters for uuid in cluster)
    assert assigned == ['left', 'right']


def test_the_patch_replaces_the_unbounded_function():
    # graphiti_core pulls in the graph drivers, which are absent from a bare
    # checkout; the behaviour above is what matters and is covered without them.
    community_operations = pytest.importorskip(
        'graphiti_core.utils.maintenance.community_operations'
    )

    original = community_operations.label_propagation
    try:
        patch.install_bounded_label_propagation_patch()
        assert community_operations.label_propagation is patch.bounded_label_propagation
    finally:
        community_operations.label_propagation = original
