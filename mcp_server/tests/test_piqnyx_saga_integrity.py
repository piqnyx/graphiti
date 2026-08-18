from piqnyx_saga_state_tool import _validate_chain


def rows(*uuids: str):
    return [{'uuid': uuid, 'name': f'batch-{index + 1}'} for index, uuid in enumerate(uuids)]


def edge(source: str, target: str):
    return {'source_uuid': source, 'target_uuid': target}


def test_linear_chain_is_valid():
    ok, errors, count = _validate_chain(
        episode_rows=rows('a', 'b', 'c'),
        next_rows=[edge('a', 'b'), edge('b', 'c')],
        first_episode_uuid='a',
        last_episode_uuid='c',
    )
    assert ok is True
    assert errors == []
    assert count == 3


def test_fork_is_rejected():
    ok, errors, count = _validate_chain(
        episode_rows=rows('a', 'b', 'c'),
        next_rows=[edge('a', 'b'), edge('a', 'c')],
        first_episode_uuid='a',
        last_episode_uuid='c',
    )
    assert ok is False
    assert any('2 NEXT_EPISODE successors' in item for item in errors)
    assert count == 1


def test_detached_episode_is_rejected():
    ok, errors, count = _validate_chain(
        episode_rows=rows('a', 'b', 'c'),
        next_rows=[edge('a', 'b')],
        first_episode_uuid='a',
        last_episode_uuid='b',
    )
    assert ok is False
    assert any('2 chain tails' in item for item in errors)
    assert any('reaches 2 of 3' in item for item in errors)
    assert count == 2


def test_cycle_is_rejected():
    ok, errors, count = _validate_chain(
        episode_rows=rows('a', 'b'),
        next_rows=[edge('a', 'b'), edge('b', 'a')],
        first_episode_uuid='a',
        last_episode_uuid='b',
    )
    assert ok is False
    assert any('0 chain heads' in item for item in errors)
    assert any('0 chain tails' in item for item in errors)
    assert any('cycle detected' in item for item in errors)
    assert count == 2


def test_cross_saga_next_edge_is_rejected():
    ok, errors, _ = _validate_chain(
        episode_rows=rows('a', 'b'),
        next_rows=[edge('a', 'b'), edge('b', 'outside')],
        first_episode_uuid='a',
        last_episode_uuid='b',
    )
    assert ok is False
    assert any('points outside saga' in item for item in errors)


def test_duplicate_episode_names_are_rejected():
    ok, errors, _ = _validate_chain(
        episode_rows=[
            {'uuid': 'a', 'name': 'same'},
            {'uuid': 'b', 'name': 'same'},
        ],
        next_rows=[edge('a', 'b')],
        first_episode_uuid='a',
        last_episode_uuid='b',
    )
    assert ok is False
    assert any('duplicate episode names' in item for item in errors)


def test_empty_saga_requires_empty_pointers():
    ok, errors, count = _validate_chain(
        episode_rows=[],
        next_rows=[],
        first_episode_uuid=None,
        last_episode_uuid=None,
    )
    assert ok is True
    assert errors == []
    assert count == 0
