"""Bound the community clustering loop so it cannot spin forever.

`label_propagation` updates every node from the previous sweep's assignment and
repeats `while True` until a sweep changes nothing. Synchronous label
propagation is not guaranteed to reach that state: two assignments can map onto
each other and alternate indefinitely. There is no iteration limit, so when that
happens the process pins a core and never returns — observed on a 177-entity
graph, where the clustering itself is sub-second work.

The fix is a ceiling, not a new algorithm. Behaviour is identical wherever the
loop converges, which is the normal case; where it does not, the sweep count runs
out and the current assignment is used. That is a sound result rather than a
compromise: label propagation produces a valid partition after every sweep, and
the disagreement that keeps it oscillating is between two partitions that are
equally good — which is precisely why neither wins.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

# Convergence, when it happens, takes a handful of sweeps; the literature puts
# it at around five for most graphs. A hundred is far beyond anything genuine and
# still returns in milliseconds on a graph of any size we run.
MAX_SWEEPS = 100


def bounded_label_propagation(projection: dict[str, list[Any]]) -> list[list[str]]:
    """Upstream's algorithm, with a limit on the number of sweeps."""
    community_map = {uuid: i for i, uuid in enumerate(projection.keys())}

    for sweep in range(MAX_SWEEPS):
        no_change = True
        new_community_map: dict[str, int] = {}

        for uuid, neighbors in projection.items():
            curr_community = community_map[uuid]

            community_candidates: dict[int, int] = defaultdict(int)
            for neighbor in neighbors:
                community_candidates[community_map[neighbor.node_uuid]] += neighbor.edge_count
            community_lst = [
                (count, community) for community, count in community_candidates.items()
            ]

            community_lst.sort(reverse=True)
            candidate_rank, community_candidate = community_lst[0] if community_lst else (0, -1)
            if community_candidate != -1 and candidate_rank > 1:
                new_community = community_candidate
            else:
                new_community = max(community_candidate, curr_community)

            new_community_map[uuid] = new_community

            if new_community != curr_community:
                no_change = False

        community_map = new_community_map

        if no_change:
            break
    else:
        logger.warning(
            'label propagation did not settle within %s sweeps; using the current partition. '
            'This means two assignments are alternating, not that the result is invalid.',
            MAX_SWEEPS,
        )

    community_cluster_map = defaultdict(list)
    for uuid, community in community_map.items():
        community_cluster_map[community].append(uuid)

    return list(community_cluster_map.values())


def install_bounded_label_propagation_patch() -> None:
    """Replace the unbounded loop before any community build can reach it."""
    from graphiti_core.utils.maintenance import community_operations

    community_operations.label_propagation = bounded_label_propagation
