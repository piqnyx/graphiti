"""Entity deduplication with each entity's own candidates (fork, 13.11).

Why: the candidates are searched for each extracted entity apart -- the fifteen nearest names --
but the model was handed them as one shared list of up to 343 rows, numbered from zero just as
the entities are. On long lists it lost track of whose candidate was whose and wrote an entity's
own number as the candidate's: "Пентагон -> Чернобыльская АЭС", "Почта России -> SpaceX". The
code took every such answer as it came. Measured on the recorded trace, 60 such merges in 25 of
529 calls since August; replaying the 20 largest calls, 30 of them at high, none with each
entity's own candidates under it and candidate numbers that never coincide with entity numbers.

The answer is checked as well: a merge is taken only with a candidate from that entity's own
list, named as that candidate is named. Anything else is asked again once, for those entities
alone; what still fails stays a new entity rather than a wrong merge.
"""

import json
import logging
import re
from types import SimpleNamespace

import pytest

from graphiti_core.llm_client.errors import RefusalError
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.prompts.dedupe_nodes import NodeResolutions
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.node_operations import resolve_extracted_nodes

ENTITIES = re.compile(r'<ENTITIES>\n(.*?)\n</ENTITIES>', re.S)


class ScriptedLLM:
    """Answers each dedupe call with the next scripted answer and keeps what was asked."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    async def generate_response(self, messages, response_model=None, **kwargs):
        self.calls.append({'messages': messages, 'response_model': response_model, **kwargs})
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _entities_in(call):
    return json.loads(ENTITIES.search(call['messages'][-1].content).group(1))


def _answer(*rows):
    return {
        'entity_resolutions': [
            {'id': i, 'name': name, 'duplicate_candidate_id': k, 'duplicate_candidate_name': echo}
            for i, name, k, echo in rows
        ]
    }


def _episode():
    return EpisodicNode(
        name='episode',
        group_id='group',
        source=EpisodeType.message,
        source_description='test',
        content='content',
        valid_at=utc_now(),
    )


def _node(name):
    return EntityNode(name=name, group_id='group', labels=['Entity'])


@pytest.fixture
def world(monkeypatch):
    """Two extracted entities, each with its own two candidates; one candidate is shared.

    Short names keep both away from the exact and fuzzy passes, so both go to the model.
    The shared list then reads joseph, shared, annie -- so with two entities in the call
    the candidate numbers are 2, 3 and 4.
    """
    joe, ann = _node('Joe'), _node('Ann')
    joseph, shared, annie = _node('Joseph'), _node('Shared Thing'), _node('Annie')

    async def search(*_, **__):
        return [[joseph, shared], [annie, shared]]

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search', search
    )
    return SimpleNamespace(joe=joe, ann=ann, joseph=joseph, shared=shared, annie=annie)


async def _resolve(llm, world):
    clients = SimpleNamespace(llm_client=llm)
    return await resolve_extracted_nodes(
        clients, [world.joe, world.ann], episode=_episode(), previous_episodes=[]
    )


@pytest.mark.asyncio
async def test_each_entity_is_shown_with_its_own_candidates_only(world):
    llm = ScriptedLLM([_answer((0, 'Joe', -1, ''), (1, 'Ann', -1, ''))])
    await _resolve(llm, world)

    prompt = llm.calls[0]['messages'][-1].content
    assert '<EXISTING ENTITIES>' not in prompt
    entities = _entities_in(llm.calls[0])
    assert [e['name'] for e in entities] == ['Joe', 'Ann']
    assert [(c['candidate_id'], c['name']) for c in entities[0]['candidates']] == [
        (2, 'Joseph'),
        (3, 'Shared Thing'),
    ]
    assert [(c['candidate_id'], c['name']) for c in entities[1]['candidates']] == [
        (4, 'Annie'),
        (3, 'Shared Thing'),
    ]


@pytest.mark.asyncio
async def test_candidate_numbers_never_coincide_with_entity_numbers(world):
    llm = ScriptedLLM([_answer((0, 'Joe', -1, ''), (1, 'Ann', -1, ''))])
    await _resolve(llm, world)

    entities = _entities_in(llm.calls[0])
    entity_ids = {e['id'] for e in entities}
    candidate_ids = {c['candidate_id'] for e in entities for c in e['candidates']}
    assert entity_ids == {0, 1}
    assert not entity_ids & candidate_ids


@pytest.mark.asyncio
async def test_the_answer_names_the_candidate_it_picks(world):
    llm = ScriptedLLM([_answer((0, 'Joe', -1, ''), (1, 'Ann', -1, ''))])
    await _resolve(llm, world)

    assert llm.calls[0]['response_model'] is NodeResolutions
    item = NodeResolutions.model_json_schema()['$defs']['NodeDuplicate']
    assert 'duplicate_candidate_name' in item['required']
    assert '`duplicate_candidate_name`' in llm.calls[0]['messages'][-1].content


@pytest.mark.asyncio
async def test_a_merge_with_its_own_candidate_by_its_name_is_taken_at_once(world):
    llm = ScriptedLLM([_answer((0, 'Joe', 2, 'Joseph'), (1, 'Ann', -1, ''))])
    resolved, uuid_map, pairs = await _resolve(llm, world)

    assert len(llm.calls) == 1
    assert resolved[0].uuid == world.joseph.uuid
    assert uuid_map[world.joe.uuid] == world.joseph.uuid
    assert resolved[1].uuid == world.ann.uuid
    assert pairs == [(world.joe, world.joseph)]


@pytest.mark.asyncio
async def test_the_name_check_ignores_case_spaces_and_yo(world, monkeypatch):
    hedgehog = _node('ежик  в тумане')

    async def search(*_, **__):
        return [[hedgehog], [world.annie]]

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search', search
    )
    llm = ScriptedLLM([_answer((0, 'Joe', 2, ' Ёжик в Тумане '), (1, 'Ann', -1, ''))])
    resolved, _, _ = await _resolve(llm, world)

    assert len(llm.calls) == 1
    assert resolved[0].uuid == hedgehog.uuid


@pytest.mark.asyncio
async def test_a_candidate_of_another_entity_is_asked_again_for_that_entity_alone(world):
    # Annie (4) is Ann's candidate, not Joe's: the answer cannot be taken as it is.
    llm = ScriptedLLM(
        [
            _answer((0, 'Joe', 4, 'Annie'), (1, 'Ann', -1, '')),
            _answer((0, 'Joe', 2, 'Joseph')),
        ]
    )
    resolved, _, _ = await _resolve(llm, world)

    assert len(llm.calls) == 2
    again = _entities_in(llm.calls[1])
    assert [(e['id'], e['name']) for e in again] == [(0, 'Joe')]
    assert [c['candidate_id'] for c in again[0]['candidates']] == [2, 3]
    assert llm.calls[1]['prompt_name'] == 'dedupe_nodes.nodes_again'
    assert resolved[0].uuid == world.joseph.uuid
    assert resolved[1].uuid == world.ann.uuid


@pytest.mark.asyncio
async def test_an_entitys_own_number_is_not_a_candidate(world):
    # The failure the shared list produced: Ann's own number written as the candidate's.
    llm = ScriptedLLM(
        [
            _answer((0, 'Joe', -1, ''), (1, 'Ann', 1, 'Ann')),
            _answer((0, 'Ann', -1, '')),
        ]
    )
    resolved, _, pairs = await _resolve(llm, world)

    assert len(llm.calls) == 2
    # Ann was entity 1 of the first question; alone in the second, she is entity 0 of it.
    assert [(e['id'], e['name']) for e in _entities_in(llm.calls[1])] == [(0, 'Ann')]
    assert resolved[1].uuid == world.ann.uuid
    assert pairs == []


@pytest.mark.asyncio
async def test_a_wrong_name_twice_leaves_the_entity_new_and_says_so(world, caplog):
    llm = ScriptedLLM(
        [
            _answer((0, 'Joe', 2, 'Josephine'), (1, 'Ann', -1, '')),
            _answer((0, 'Joe', 2, 'Josephine')),
        ]
    )
    with caplog.at_level(logging.WARNING):
        resolved, uuid_map, pairs = await _resolve(llm, world)

    assert len(llm.calls) == 2
    assert resolved[0].uuid == world.joe.uuid
    assert uuid_map[world.joe.uuid] == world.joe.uuid
    assert pairs == []
    assert 'Joe' in caplog.text and 'Josephine' in caplog.text


@pytest.mark.asyncio
async def test_a_missing_entity_is_asked_again(world):
    llm = ScriptedLLM(
        [
            _answer((0, 'Joe', -1, '')),
            _answer((0, 'Ann', 4, 'Annie')),
        ]
    )
    resolved, _, _ = await _resolve(llm, world)

    assert len(llm.calls) == 2
    assert [(e['id'], e['name']) for e in _entities_in(llm.calls[1])] == [(0, 'Ann')]
    assert resolved[1].uuid == world.annie.uuid


@pytest.mark.asyncio
async def test_no_duplicate_is_taken_without_asking_again(world):
    llm = ScriptedLLM([_answer((0, 'Joe', -1, ''), (1, 'Ann', -1, ''))])
    resolved, _, pairs = await _resolve(llm, world)

    assert len(llm.calls) == 1
    assert [n.uuid for n in resolved] == [world.joe.uuid, world.ann.uuid]
    assert pairs == []


@pytest.mark.asyncio
async def test_the_question_is_asked_again_once_at_most(world):
    llm = ScriptedLLM(
        [
            _answer((0, 'Joe', 4, 'Annie'), (1, 'Ann', 2, 'Joseph')),
            _answer((0, 'Joe', 4, 'Annie'), (1, 'Ann', 2, 'Joseph')),
            _answer((0, 'Joe', 2, 'Joseph'), (1, 'Ann', 4, 'Annie')),
        ]
    )
    resolved, _, pairs = await _resolve(llm, world)

    assert len(llm.calls) == 2
    assert [n.uuid for n in resolved] == [world.joe.uuid, world.ann.uuid]
    assert pairs == []


@pytest.mark.asyncio
async def test_two_entities_merged_into_one_node_land_on_one_copy_of_it(monkeypatch):
    # Each search returns its own copy of a graph node. Both merges must land on the one copy
    # the shared list holds, as they always did, so the type one entity brings reaches the
    # node the other entity resolves to as well.
    joe = EntityNode(name='Joe', group_id='group', labels=['Entity', 'Person'])
    ann = _node('Ann')
    found_by_joe = _node('Joseph')
    found_by_ann = EntityNode(
        uuid=found_by_joe.uuid, name='Joseph', group_id='group', labels=['Entity']
    )

    async def search(*_, **__):
        return [[found_by_joe], [found_by_ann]]

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search', search
    )
    llm = ScriptedLLM([_answer((0, 'Joe', 2, 'Joseph'), (1, 'Ann', 2, 'Joseph'))])
    resolved, _, _ = await resolve_extracted_nodes(
        SimpleNamespace(llm_client=llm), [joe, ann], episode=_episode(), previous_episodes=[]
    )

    assert len(llm.calls) == 1
    assert resolved[0] is resolved[1]
    assert set(resolved[1].labels) == {'Entity', 'Person'}


@pytest.mark.asyncio
async def test_an_entity_left_out_twice_is_named_in_the_log(world, caplog):
    llm = ScriptedLLM([_answer((0, 'Joe', -1, '')), _answer()])
    with caplog.at_level(logging.WARNING):
        resolved, _, pairs = await _resolve(llm, world)

    assert len(llm.calls) == 2
    assert resolved[1].uuid == world.ann.uuid
    assert pairs == []
    assert any('Ann' in r.getMessage() and 'twice' in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_failed_second_question_keeps_the_first_answer(world, caplog):
    # Asking again is insurance: when it fails, the entities it was for stay new and the
    # episode goes on with what the first answer settled.
    llm = ScriptedLLM(
        [
            _answer((0, 'Joe', 2, 'Joseph'), (1, 'Ann', 2, 'Joseph')),
            RefusalError('provider refused the prompt'),
        ]
    )
    with caplog.at_level(logging.WARNING):
        resolved, _, pairs = await _resolve(llm, world)

    assert len(llm.calls) == 2
    assert resolved[0].uuid == world.joseph.uuid
    assert resolved[1].uuid == world.ann.uuid
    assert pairs == [(world.joe, world.joseph)]
    assert any('Ann' in r.getMessage() and 'RefusalError' in r.getMessage() for r in caplog.records)
