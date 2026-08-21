"""The pool is only as wide as the floor under it allows."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from graphiti_core.search.search_config_recipes import (  # noqa: E402
    EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    EDGE_HYBRID_SEARCH_RRF,
)

from graphiti_mcp_server import widen_recipe  # noqa: E402


def test_limit_is_applied_and_the_recipe_is_left_alone():
    widened = widen_recipe(EDGE_HYBRID_SEARCH_RRF, 40, None)
    assert widened.limit == 40
    assert EDGE_HYBRID_SEARCH_RRF.limit != 40 or widened is not EDGE_HYBRID_SEARCH_RRF


def test_without_asking_the_library_floor_stands():
    original = EDGE_HYBRID_SEARCH_RRF.edge_config.sim_min_score
    widened = widen_recipe(EDGE_HYBRID_SEARCH_RRF, 40, None)
    assert widened.edge_config.sim_min_score == original


def test_the_vector_floor_reaches_the_edge_config():
    # Asking for a pool of forty is meaningless while a cosine floor of 0.6 lets two
    # edges out of the database: measured on a live graph, that is what happened.
    widened = widen_recipe(EDGE_HYBRID_SEARCH_RRF, 40, 0.25)
    assert widened.edge_config.sim_min_score == 0.25
    assert widened.limit == 40


def test_lowering_the_floor_never_touches_the_shared_recipe():
    before = EDGE_HYBRID_SEARCH_RRF.edge_config.sim_min_score
    widen_recipe(EDGE_HYBRID_SEARCH_RRF, 40, 0.1)
    assert EDGE_HYBRID_SEARCH_RRF.edge_config.sim_min_score == before


def test_extra_updates_ride_along_for_the_cross_encoder_recipe():
    widened = widen_recipe(
        EDGE_HYBRID_SEARCH_CROSS_ENCODER, 40, 0.3, {'reranker_min_score': 0.08}
    )
    assert widened.reranker_min_score == 0.08
    assert widened.edge_config.sim_min_score == 0.3
    assert widened.limit == 40


class _Edge:
    def __init__(self, uuid, fact):
        self.uuid = uuid
        self.fact = fact


def test_merging_keeps_the_first_appearance_and_drops_repeats():
    from graphiti_mcp_server import merge_candidates

    remark = [_Edge('a', 'про Марину'), _Edge('b', 'про Антона')]
    conversation = [_Edge('b', 'про Антона'), _Edge('c', 'про C++')]
    merged = merge_candidates(remark, conversation)
    assert [edge.uuid for edge in merged] == ['a', 'b', 'c']


def test_two_edges_sharing_a_sentence_stay_two():
    from graphiti_mcp_server import merge_candidates

    # Identity is the uuid, not the text: this graph holds duplicated facts, and
    # collapsing them here would hide a defect rather than fix it.
    merged = merge_candidates([_Edge('a', 'одно и то же'), _Edge('b', 'одно и то же')])
    assert len(merged) == 2


def test_merging_nothing_is_nothing():
    from graphiti_mcp_server import merge_candidates

    assert merge_candidates([], []) == []


def test_a_ranking_that_separates_is_kept():
    from graphiti_mcp_server import discriminates

    # Asked which car someone drives: the answer second, the tail far below.
    assert discriminates([0.6178, 0.6083, 0.513, 0.379, 0.2752, 0.1354], 0.2)


def test_a_flat_ranking_is_refused():
    from graphiti_mcp_server import discriminates

    # Asked how far away something was, and asked about a fridge nobody mentioned.
    assert not discriminates([0.8088, 0.8041, 0.8009, 0.8003, 0.7998, 0.7922], 0.2)
    assert not discriminates([0.6192, 0.6122, 0.6108, 0.6061, 0.603, 0.5818], 0.2)


def test_too_few_results_to_read_a_spread():
    from graphiti_mcp_server import discriminates

    # Two facts sitting close may both be right; there is no distribution to judge.
    assert discriminates([0.28, 0.27], 0.2)
    assert discriminates([], 0.2)


def test_the_test_stands_aside_unless_asked():
    from graphiti_mcp_server import discriminates

    assert discriminates([0.80, 0.79, 0.79, 0.79], None)
