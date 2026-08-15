from tools.falkor_validate import (
    Episode,
    Report,
    Saga,
    SemanticExpectation,
    validate_chain,
    validate_semantics,
)


def episode(number: int) -> Episode:
    return Episode(
        uuid=f'ep-{number}',
        name=f'saga-1-{number}',
        group_id='main',
        created_at=f'2026-08-15T00:00:0{number}+00:00',
        valid_at=f'2026-08-15T00:00:0{number}+00:00',
    )


def saga(count: int = 3) -> Saga:
    return Saga(
        uuid='saga-uuid',
        name='agent:main:test:saga-1',
        group_id='main',
        first_episode_uuid='ep-1' if count else None,
        last_episode_uuid=f'ep-{count}' if count else None,
        created_at='2026-08-15T00:00:00+00:00',
    )


def test_valid_directed_chain_is_ordered_predecessor_to_successor():
    report = Report()
    ordered = validate_chain(
        saga(),
        [episode(1), episode(2), episode(3)],
        [('ep-1', 'ep-2'), ('ep-2', 'ep-3')],
        report,
    )

    assert [item.uuid for item in ordered] == ['ep-1', 'ep-2', 'ep-3']
    assert report.failures == []


def test_reciprocal_next_episode_edge_is_rejected_as_cycle_and_bad_degree():
    report = Report()
    validate_chain(
        saga(2),
        [episode(1), episode(2)],
        [('ep-1', 'ep-2'), ('ep-2', 'ep-1')],
        report,
    )

    assert report.failures
    assert any('NEXT_EPISODE edges' in item for item in report.failures)
    assert any('predecessor' in item or 'successor' in item for item in report.failures)


def test_cross_saga_next_episode_edge_is_rejected():
    report = Report()
    validate_chain(
        saga(2),
        [episode(1), episode(2)],
        [('ep-1', 'foreign-episode'), ('ep-1', 'ep-2')],
        report,
    )

    assert any('crosses saga boundary' in item for item in report.failures)


def test_entityless_episode_is_warning_not_structural_failure():
    report = Report()
    ordered = [episode(1), episode(2)]
    validate_semantics(
        saga(2),
        ordered,
        {'ep-1': ['Вит'], 'ep-2': []},
        {'ep-1': ['Вит uses Graphiti'], 'ep-2': []},
        [],
        report,
    )

    assert report.failures == []
    assert any('zero MENTIONS' in item and 'saga-1-2' in item for item in report.warnings)


def test_required_semantic_expectation_fails_when_entity_or_fact_is_missing():
    report = Report()
    validate_semantics(
        saga(1),
        [episode(1)],
        {'ep-1': ['VS Code']},
        {'ep-1': ['Вит compares editors']},
        [
            SemanticExpectation(
                episode_selector='saga-1-1',
                entity_terms=('Cursor',),
                fact_terms=('Graphiti',),
            )
        ],
        report,
    )

    assert len(report.failures) == 2
    assert any('Cursor' in item for item in report.failures)
    assert any('Graphiti' in item for item in report.failures)


def test_semantic_terms_match_case_insensitive_substrings():
    report = Report()
    validate_semantics(
        saga(1),
        [episode(1)],
        {'ep-1': ['Visual Studio Code']},
        {'ep-1': ['Вит рассматривает Visual Studio Code как редактор']},
        [
            SemanticExpectation(
                episode_selector='ep-1',
                entity_terms=('studio code',),
                fact_terms=('РЕДАКТОР',),
            )
        ],
        report,
    )

    assert report.failures == []
