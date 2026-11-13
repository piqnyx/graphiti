"""Undoing the old dedupe's wrong merges (fork, 13.11).

The pure helpers run everywhere. The rest runs against a real FalkorDB, the version the
server uses, and is skipped unless UNMERGE_TEST_FALKOR_PORT names one:

    docker run -d --rm -p 127.0.0.1:16379:6379 falkordb/falkordb:v4.2.1
    UNMERGE_TEST_FALKOR_PORT=16379 pytest tests/tools/test_unmerge_old_dedupe.py

The graph it builds is saved with Graphiti's own functions, so what the tool meets has
exactly the shape add_episode leaves.
"""

import asyncio
import copy
import json
import os
import re
from datetime import datetime, timedelta, timezone

import pytest

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from tools.unmerge_old_dedupe import checks_of, compare, norm, remove_fact_lines

PORT = os.environ.get('UNMERGE_TEST_FALKOR_PORT')
needs_db = pytest.mark.skipif(not PORT, reason='UNMERGE_TEST_FALKOR_PORT is not set')


# ---------------------------------------------------------------- pure helpers


def test_remove_fact_lines_takes_one_line_per_fact_from_the_end():
    summary = 'Немецкое СМИ.\nКанада ввела санкции.\nКанада ввела санкции.\nКанада ввела санкции.'
    after, missing = remove_fact_lines(summary, ['Канада ввела санкции.'] * 3)
    assert after == 'Немецкое СМИ.'
    assert missing == []


def test_remove_fact_lines_leaves_a_line_that_was_not_moved():
    summary = 'Свой факт.\nЧужой факт.\nЧужой факт.'
    after, missing = remove_fact_lines(summary, ['Чужой факт.'])
    assert after == 'Свой факт.\nЧужой факт.'
    assert missing == []


def test_remove_fact_lines_reports_a_fact_with_no_line_and_keeps_text_mentioning_it():
    summary = 'NetEase создал эмулятор MuMu.'
    after, missing = remove_fact_lines(summary, ['MuMu вышел на Android 12'])
    assert after == summary
    assert missing == ['MuMu вышел на Android 12']


def test_norm_treats_case_yo_and_spaces_alike():
    assert norm('  Ёжик   в Тумане ') == norm('ежик в тумане')


def snap(**over):
    base = {
        'labels': {"['Episodic']": 2, "['Entity']": 2},
        'types': {'MENTIONS': 1, 'RELATES_TO': 1, 'NEXT_EPISODE': 1, 'HAS_EPISODE': 2},
        'entities': {
            'w': {
                'name': 'W',
                'summary': 'S',
                'group_id': 'main',
                'labels': ['Entity'],
                'embedding': 'Vectorf32',
            }
        },
        'edges': {
            'r': {
                'source': 'w',
                'target': 'o',
                'source_prop': 'w',
                'target_prop': 'o',
                'fact': 'f',
                'episodes': ['e'],
                'group_id': 'main',
                'name': 'N',
                'valid_at': None,
                'invalid_at': None,
                'expired_at': None,
                'has_embedding': 'Vectorf32',
            }
        },
        'mentions': {
            'm': {
                'episode': 'e',
                'entity': 'w',
                'from_episodic': True,
                'to_entity': True,
                'has_uuid': True,
            }
        },
        'episodes': ['e'],
    }
    base.update(over)
    base.setdefault('checks', checks_of(base, 'main'))
    return base


NO_OPS = {
    'new_nodes': [],
    'summary_edits': [],
    'edge_moves': [],
    'mentions_add': [],
    'mentions_drop': [],
}


def test_compare_accepts_identical_snapshots():
    assert compare(snap(), snap(), NO_OPS, 'main') == []


def test_compare_names_a_summary_change_the_plan_did_not_make():
    after = snap()
    after['entities'] = {'w': dict(after['entities']['w'], summary='other')}
    assert any(p.startswith('entity w:') for p in compare(snap(), after, NO_OPS, 'main'))


def test_compare_names_a_fact_moved_without_the_plan():
    after = snap()
    after['edges'] = {'r': dict(after['edges']['r'], source='x', source_prop='x')}
    assert any(p.startswith('fact r:') for p in compare(snap(), after, NO_OPS, 'main'))


def test_compare_names_a_lost_episode():
    after = snap(labels={"['Episodic']": 1, "['Entity']": 2})
    assert any('Episodic' in p for p in compare(snap(), after, NO_OPS, 'main'))


def test_compare_names_a_new_orphan_entity():
    after = snap(mentions={})
    problems = compare(snap(), after, NO_OPS, 'main')
    assert any('entity_without_mention' in p for p in problems)


# ---------------------------------------------------------------- the graph

GROUP = 'main'
DIM = 32
AXES: dict[str, int] = {}


def vector(name: str) -> list[float]:
    """Each name its own axis; names meant to be dedupe candidates of one another share it."""
    shared = {
        'BBC News': 'BBC',
        'Канада': 'Канада',
        'Swift': 'Swift',
        'Google Mail': 'Gmail',
        'Prague': 'Прага',
        'Brno': 'Брно',
        'ФСИН России': 'ФСИН',
        'Чемезов': 'Ростех',
    }
    key = shared.get(name, name)
    axis = AXES.setdefault(key, len(AXES) % DIM)
    v = [0.0] * DIM
    v[axis] = 1.0
    if name != key:
        v[(axis + 1) % DIM] = 0.2
    return v


class Embedder(EmbedderClient):
    async def create(self, input_data):
        return vector(input_data[0] if isinstance(input_data, list) else input_data)

    async def create_batch(self, input_data_list):
        return [vector(x) for x in input_data_list]


class CrossEncoder(CrossEncoderClient):
    async def rank(self, query, passages):
        return [(p, 1.0) for p in passages]


ENTITIES = re.compile(r'<ENTITIES>\s*(.*?)\s*</ENTITIES>', re.S)


class LLM(LLMClient):
    """Dedupe answers from a name -> chosen candidate map; summaries for whatever is asked."""

    def __init__(self, choices):
        super().__init__(LLMConfig(api_key='test'), cache=False)
        self.choices = choices
        self.calls = []
        self.summarized = []

    async def _generate_response(self, *args, **kwargs):
        raise AssertionError('generate_response is answered directly')

    async def generate_response(self, messages, response_model=None, **kwargs):
        name = kwargs.get('prompt_name', '')
        self.calls.append(name)
        if name.startswith('dedupe_nodes'):
            rows = []
            for e in json.loads(ENTITIES.search(messages[-1].content).group(1)):
                want = self.choices.get(e['name'])
                want, echo = want if isinstance(want, tuple) else (want, want)
                hit = next((c for c in e.get('candidates', []) if c['name'] == want), None)
                rows.append(
                    {
                        'id': e['id'],
                        'name': e['name'],
                        'duplicate_candidate_id': hit['candidate_id'] if hit else -1,
                        'duplicate_candidate_name': echo if hit else '',
                    }
                )
            return {'entity_resolutions': rows}
        if 'summar' in name:
            asked = json.loads(ENTITIES.search(messages[-1].content).group(1))
            self.summarized += [e['name'] for e in asked]
            return {
                'summaries': [
                    {'name': e['name'], 'summary': f'Сводка: {e["name"]}.'} for e in asked
                ]
            }
        raise AssertionError(f'unexpected call {name}')


def services(choices=None):
    return {
        'llm': LLM(
            choices
            or {
                'BBC News': 'BBC',
                'Google Mail': 'Gmail',
                'Prague': 'Прага',
                'Brno': 'Брно',
                'ФСИН России': 'ФСИН',
                # Names Ростех wrongly every time: the check fails on every question.
                'Чемезов': ('Ростех', 'Ростех-не-тот'),
            }
        ),
        'embedder': Embedder(),
        'cross_encoder': CrossEncoder(),
        'entity_types': None,
    }


@pytest.fixture
def db(monkeypatch):
    from falkordb import FalkorDB

    monkeypatch.setenv('UNMERGE_DB_HOST', '127.0.0.1')
    monkeypatch.setenv('UNMERGE_DB_PORT', PORT or '0')
    client = FalkorDB(host='127.0.0.1', port=int(PORT))
    for name in client.list_graphs():
        episodes = (
            client.select_graph(name)
            .ro_query('MATCH (e:Episodic) RETURN count(e)')
            .result_set[0][0]
        )
        assert episodes < 50, (
            f'graph {name!r} holds {episodes} episodes: not a test database, not wiping it'
        )
    for name in client.list_graphs():
        client.select_graph(name).delete()
    yield client
    for name in client.list_graphs():
        client.select_graph(name).delete()


T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)


async def seed():
    """A graph with the wrong merges the trace showed, saved the way add_episode saves."""
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from graphiti_core.edges import EntityEdge, EpisodicEdge
    from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
    from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk

    driver = FalkorDriver(host='127.0.0.1', port=int(PORT), database=GROUP)
    await driver.build_indices_and_constraints()
    emb = Embedder()
    made: dict[str, EntityNode] = {}
    episodes: dict[str, EpisodicNode] = {}

    async def episode(key, minutes, text, mentions, facts=(), summaries=None):
        ep = EpisodicNode(
            name=key,
            group_id=GROUP,
            labels=[],
            source=EpisodeType.json,
            source_description='chat',
            content=json.dumps({'messages': [{'text': text}]}, ensure_ascii=False),
            created_at=T0 + timedelta(minutes=minutes, seconds=7),
            valid_at=T0 + timedelta(minutes=minutes),
        )
        episodes[key] = ep
        new_nodes = []
        for n in mentions:
            if n not in made:
                made[n] = EntityNode(
                    name=n,
                    group_id=GROUP,
                    labels=['Entity'],
                    summary='',
                    created_at=ep.created_at,
                    name_embedding=vector(n),
                )
                new_nodes.append(made[n])
        for n, s in (summaries or {}).items():
            made[n].summary = s
        edges = [
            EntityEdge(
                group_id=GROUP,
                source_node_uuid=made[a].uuid,
                target_node_uuid=made[b].uuid,
                created_at=ep.created_at,
                name='RELATES',
                fact=f,
                episodes=[ep.uuid],
                fact_embedding=vector(f),
                valid_at=ep.valid_at,
            )
            for a, b, f in facts
        ]
        mention_edges = [
            EpisodicEdge(
                source_node_uuid=ep.uuid,
                target_node_uuid=made[n].uuid,
                group_id=GROUP,
                created_at=ep.created_at,
            )
            for n in mentions
        ]
        await add_nodes_and_edges_bulk(
            driver,
            [ep],
            mention_edges,
            new_nodes + [made[n] for n in (summaries or {}) if made[n] not in new_nodes],
            edges,
            emb,
        )
        return ep

    sanctions = 'Канада ввела санкции против банков Яндекса, Ozon и Wildberries.'
    await episode(
        'e0',
        0,
        'старые новости',
        [
            'Чернобыльская АЭС',
            'Deutsche Welle',
            'Euronews',
            'Яндекс',
            'Ozon',
            'Wildberries',
            'Red Hat',
            'Виктор Орехов',
            'Донецк',
            'ViT',
            'BBC',
            'Киев',
            'Оксфорд',
            'Gmail',
            'Прага',
            'Брно',
            'Росатом',
            'Минфин',
            'Ростех',
            'Минобороны',
            'ФСИН',
            'МВД2',
        ],
        facts=[
            ('Чернобыльская АЭС', 'Киев', 'Чернобыльская АЭС находится рядом с Киевом'),
            ('Донецк', 'Киев', 'Донецк связан с Киевом'),
        ],
        summaries={
            'Deutsche Welle': 'Немецкое СМИ.',
            'Euronews': 'Европейское СМИ.',
            'Оксфорд': 'Университет.',
            'Росатом': 'Госкорпорация.',
            'Минфин': 'Министерство.',
            'Минобороны': 'Ведомство.',
            'ФСИН': 'Служба.',
        },
    )
    # Пентагон wrongly merged into Чернобыльская АЭС; no facts.
    await episode('e1', 10, 'Пентагон заявил', ['Чернобыльская АЭС'])
    # Канада into Deutsche Welle, Яндекс into Euronews: one fact has both ends wrong.
    await episode(
        'e2',
        20,
        'Канада ввела санкции',
        ['Deutsche Welle', 'Euronews', 'Ozon', 'Wildberries'],
        facts=[
            ('Deutsche Welle', 'Euronews', sanctions),
            ('Deutsche Welle', 'Ozon', sanctions),
            ('Deutsche Welle', 'Wildberries', sanctions),
        ],
        summaries={
            'Deutsche Welle': f'Немецкое СМИ.\n{sanctions}\n{sanctions}\n{sanctions}',
            'Euronews': f'Европейское СМИ.\n{sanctions}',
        },
    )
    # Swift twice, into two different wrong nodes.
    await episode('e3', 30, 'Swift в новостях', ['Red Hat'])
    await episode('e4', 40, 'снова Swift', ['Виктор Орехов'])
    # ВСУ into Донецк, while the episode named Донецк itself.
    await episode('e5', 50, 'ВСУ и Донецк', ['Донецк'])
    # BBC News into ViT; the new dedupe finds BBC, which the episode already named itself.
    # Its fact about BBC then becomes a loop on BBC, as add_episode would have made it.
    quote = 'BBC News цитирует BBC'
    await episode(
        'e6',
        60,
        'BBC News сообщает, BBC подтверждает',
        ['ViT', 'BBC'],
        facts=[('ViT', 'BBC', quote)],
        summaries={'ViT': quote, 'BBC': quote},
    )
    # Яндекс into Оксфорд with a fact: Яндекс gains facts from two episodes.
    await episode(
        'e7',
        70,
        'Яндекс купил долю',
        ['Оксфорд', 'Ozon'],
        facts=[('Оксфорд', 'Ozon', 'Яндекс купил долю Ozon')],
        summaries={'Оксфорд': 'Университет.\nЯндекс купил долю Ozon'},
    )
    # Google Mail into Gmail: an old merge the new dedupe makes too.
    await episode('e8', 80, 'Google Mail', ['Gmail'])
    # Prague into Брно and Brno into Прага in one episode: each wrong node is the other's twin.
    await episode('e9', 90, 'Prague и Brno', ['Брно', 'Прага'])
    # Two episodes with the same text: neither can be told apart from the other.
    await episode('e10', 100, 'повтор', ['ViT'])
    await episode('e11', 110, 'повтор', ['ViT'])
    # Шойгу into Росатом, which the episode named itself: his fact became a loop on Росатом.
    loop = 'Шойгу посетил Росатом'
    await episode(
        'e12',
        120,
        'Шойгу и Росатом',
        ['Росатом'],
        facts=[('Росатом', 'Росатом', loop)],
        summaries={'Росатом': f'Госкорпорация.\n{loop}\n{loop}'},
    )
    # Мишустин and Набиуллина both into Минфин: a loop between them and one more fact.
    order, ozon = 'Мишустин поручил Набиуллиной', 'Мишустин упомянул Ozon'
    await episode(
        'e13',
        130,
        'Мишустин и Набиуллина',
        ['Минфин', 'Минфин', 'Ozon'],
        facts=[('Минфин', 'Минфин', order), ('Минфин', 'Ozon', ozon)],
        summaries={'Минфин': f'Министерство.\n{order}\n{order}\n{ozon}'},
    )
    # Луганск into Донецк, its fact judged the same as an older one: e14 only appended itself to it.
    e14 = await episode('e14', 140, 'Луганск', ['Донецк'])
    await driver.execute_query(
        'MATCH ()-[r:RELATES_TO]->() WHERE r.fact = $f SET r.episodes = r.episodes + [$e]',
        f='Донецк связан с Киевом',
        e=e14.uuid,
    )
    # Чемезов into Ростех, which the episode named itself: two mentions of Ростех.
    await episode('e15', 150, 'Чемезов и Ростех', ['Ростех', 'Ростех'])
    # A case the trace could not explain.
    await episode('e16', 160, 'неясно', ['ViT'])
    # Росгвардия into Минобороны, ФСИН России into МВД2 and found to be ФСИН: the moved fact's other
    # end is a target that already holds it. Later e5 joined that fact as well.
    guard = 'Росгвардия передала дела ФСИН'
    await episode(
        'e17',
        170,
        'Росгвардия и ФСИН',
        ['Минобороны', 'ФСИН', 'МВД2'],
        facts=[('Минобороны', 'ФСИН', guard)],
        summaries={'Минобороны': f'Ведомство.\n{guard}', 'ФСИН': f'Служба.\n{guard}'},
    )
    await driver.execute_query(
        'MATCH ()-[r:RELATES_TO]->() WHERE r.fact = $f SET r.episodes = r.episodes + [$e]',
        f=guard,
        e=episodes['e5'].uuid,
    )
    return driver, episodes, made, sanctions


def cases(episodes, sanctions):
    text = lambda key: episodes[key].content  # noqa: E731
    return [
        {
            'entity': 'Пентагон',
            'wrong': 'Чернобыльская АЭС',
            'episode_content': text('e1'),
            'entity_facts': [],
        },
        {
            'entity': 'Канада',
            'wrong': 'Deutsche Welle',
            'episode_content': text('e2'),
            'entity_labels': ['Entity', 'Organization'],
            'entity_facts': [
                {'source': 'Канада', 'target': x, 'fact': sanctions}
                for x in ('Яндекс', 'Ozon', 'Wildberries')
            ],
        },
        {
            'entity': 'Яндекс',
            'wrong': 'Euronews',
            'episode_content': text('e2'),
            'entity_facts': [{'source': 'Канада', 'target': 'Яндекс', 'fact': sanctions}],
        },
        {'entity': 'Swift', 'wrong': 'Red Hat', 'episode_content': text('e3'), 'entity_facts': []},
        {
            'entity': 'Swift',
            'wrong': 'Виктор Орехов',
            'episode_content': text('e4'),
            'entity_facts': [],
        },
        {
            'entity': 'ВСУ',
            'wrong': 'Донецк',
            'episode_content': text('e5'),
            'entity_facts': [],
            'keep_mention': True,
        },
        {
            'entity': 'BBC News',
            'wrong': 'ViT',
            'episode_content': text('e6'),
            'entity_facts': [
                {'source': 'BBC News', 'target': 'BBC', 'fact': 'BBC News цитирует BBC'}
            ],
        },
        {
            'entity': 'Нет такого',
            'wrong': 'ViT',
            'episode_content': 'текста нет нигде',
            'entity_facts': [],
        },
        {
            'entity': 'Яндекс',
            'wrong': 'Оксфорд',
            'episode_content': text('e7'),
            'entity_facts': [
                {'source': 'Яндекс', 'target': 'Ozon', 'fact': 'Яндекс купил долю Ozon'}
            ],
        },
        {
            'entity': 'Google Mail',
            'wrong': 'Gmail',
            'episode_content': text('e8'),
            'entity_facts': [],
        },
        # Gmail Inc into Gmail too, in the same episode; it is undone while Google Mail stays.
        {
            'entity': 'Gmail Inc',
            'wrong': 'Gmail',
            'episode_content': text('e8'),
            'entity_facts': [],
        },
        {'entity': 'Prague', 'wrong': 'Брно', 'episode_content': text('e9'), 'entity_facts': []},
        {'entity': 'Brno', 'wrong': 'Прага', 'episode_content': text('e9'), 'entity_facts': []},
        {'entity': 'Повтор', 'wrong': 'ViT', 'episode_content': text('e10'), 'entity_facts': []},
        {
            'entity': 'Шойгу',
            'wrong': 'Росатом',
            'episode_content': text('e12'),
            'keep_mention': True,
            'entity_facts': [
                {'source': 'Шойгу', 'target': 'Росатом', 'fact': 'Шойгу посетил Росатом'}
            ],
        },
        {
            'entity': 'Мишустин',
            'wrong': 'Минфин',
            'episode_content': text('e13'),
            'entity_facts': [
                {
                    'source': 'Мишустин',
                    'target': 'Набиуллина',
                    'fact': 'Мишустин поручил Набиуллиной',
                },
                {'source': 'Мишустин', 'target': 'Ozon', 'fact': 'Мишустин упомянул Ozon'},
            ],
        },
        {
            'entity': 'Набиуллина',
            'wrong': 'Минфин',
            'episode_content': text('e13'),
            'entity_facts': [
                {
                    'source': 'Мишустин',
                    'target': 'Набиуллина',
                    'fact': 'Мишустин поручил Набиуллиной',
                }
            ],
        },
        {
            'entity': 'Луганск',
            'wrong': 'Донецк',
            'episode_content': text('e14'),
            'entity_facts': [
                {'source': 'Луганск', 'target': 'Киев', 'fact': 'Донецк связан с Киевом'}
            ],
        },
        {
            'entity': 'Чемезов',
            'wrong': 'Ростех',
            'episode_content': text('e15'),
            'keep_mention': True,
            'entity_facts': [],
        },
        {
            'entity': 'Неясно',
            'wrong': 'ViT',
            'episode_content': text('e16'),
            'entity_facts': [],
            'unknown': True,
        },
        {
            'entity': 'Росгвардия',
            'wrong': 'Минобороны',
            'episode_content': text('e17'),
            'entity_facts': [
                {'source': 'Росгвардия', 'target': 'ФСИН', 'fact': 'Росгвардия передала дела ФСИН'}
            ],
        },
        {
            'entity': 'ФСИН России',
            'wrong': 'МВД2',
            'episode_content': text('e17'),
            'entity_facts': [],
        },
    ]


async def made_plan(choices=None):
    from tools.unmerge_old_dedupe import make_plan

    driver, episodes, made, sanctions = await seed()
    plan = await make_plan(cases(episodes, sanctions), [GROUP], services(choices))
    return plan, episodes, made, sanctions


def by_entity(plan):
    return {(c['entity'], c['wrong']): c for c in plan['cases']}


@needs_db
def test_plan_sends_each_entity_where_the_new_dedupe_puts_it(db):
    plan, _, made, _ = asyncio.run(made_plan())
    c = by_entity(plan)
    assert c[('Пентагон', 'Чернобыльская АЭС')]['decision'] == 'new'
    assert c[('Канада', 'Deutsche Welle')]['decision'] == 'new'
    assert c[('Яндекс', 'Euronews')]['decision'] == 'existing'
    assert c[('Яндекс', 'Euronews')]['target_uuid'] == made['Яндекс'].uuid
    assert c[('BBC News', 'ViT')]['decision'] == 'existing'
    assert c[('BBC News', 'ViT')]['target_uuid'] == made['BBC'].uuid
    first, second = c[('Swift', 'Red Hat')], c[('Swift', 'Виктор Орехов')]
    assert (first['decision'], second['decision']) == ('new', 'new_planned_earlier')
    assert first['target_uuid'] == second['target_uuid']
    assert {n['name'] for n in plan['new_nodes']} == {
        'ВСУ',
        'Канада',
        'Пентагон',
        'Swift',
        'Чемезов',
        'Gmail Inc',
        'Росгвардия',
    }
    # Every new node carries the attributes Graphiti gave it, empty or not, for write to keep.
    assert all(isinstance(n['attributes'], dict) for n in plan['new_nodes'])
    assert sorted(p['entity'] for p in plan['problems'] if 'entity' in p) == [
        'Луганск',
        'Мишустин',
        'Набиуллина',
        'Нет такого',
        'Неясно',
        'Повтор',
        'Шойгу',
    ]
    assert [p['why'] for p in plan['problems'] if p.get('entity') == 'Повтор'] == [
        '2 episodes carry this text'
    ]


@needs_db
def test_plan_moves_a_fact_with_both_ends_wrong_once_and_cleans_both_summaries(db):
    plan, _, made, sanctions = asyncio.run(made_plan())
    kanada = by_entity(plan)[('Канада', 'Deutsche Welle')]['target_uuid']
    moves = {
        (m['old_source'], m['old_target']): (m['new_source'], m['new_target'])
        for m in plan['edge_moves']
    }
    assert len([m for m in plan['edge_moves'] if m['fact'] == sanctions]) == 3
    assert moves[(made['Deutsche Welle'].uuid, made['Euronews'].uuid)] == (
        kanada,
        made['Яндекс'].uuid,
    )
    assert moves[(made['Deutsche Welle'].uuid, made['Ozon'].uuid)] == (kanada, made['Ozon'].uuid)
    edits = {e['node_uuid']: e for e in plan['summary_edits']}
    assert edits[made['Deutsche Welle'].uuid]['after'] == 'Немецкое СМИ.'
    assert edits[made['Euronews'].uuid]['after'] == 'Европейское СМИ.'
    assert not [p for p in plan['problems'] if 'summary has no line' in p.get('why', '')]
    new = {n['name']: n for n in plan['new_nodes']}
    assert new['Канада']['summary'].count(sanctions) == 3


@needs_db
def test_plan_keeps_the_mention_the_episode_made_itself(db):
    plan, episodes, made, _ = asyncio.run(made_plan())
    dropped = {(m['episode'], m['entity']) for m in plan['mentions_drop']}
    assert (episodes['e5'].uuid, made['Донецк'].uuid) not in dropped
    assert (episodes['e1'].uuid, made['Чернобыльская АЭС'].uuid) in dropped
    assert (episodes['e2'].uuid, made['Deutsche Welle'].uuid) in dropped
    added = {(m['episode'], m['entity']) for m in plan['mentions_add']}
    assert (episodes['e2'].uuid, made['Яндекс'].uuid) in added
    assert len(added) == len(plan['mentions_add'])


@needs_db
def test_plan_writes_nothing(db):
    from tools.unmerge_old_dedupe import snapshot

    async def run():
        driver, episodes, made, sanctions = await seed()
        before = snapshot(db.select_graph(GROUP), GROUP)
        from tools.unmerge_old_dedupe import make_plan

        await make_plan(cases(episodes, sanctions), [GROUP], services())
        return before, snapshot(db.select_graph(GROUP), GROUP)

    before, after = asyncio.run(run())
    assert before == after


def copy_graph(db, source, target):
    db.connection.execute_command('GRAPH.COPY', source, target)


@needs_db
def test_apply_on_a_copy_changes_exactly_the_plan_and_leaves_the_source_alone(db, capsys):
    from tools.unmerge_old_dedupe import apply, snapshot

    plan, episodes, made, _ = asyncio.run(made_plan())
    plan = json.loads(json.dumps(plan))  # what a plan file gives back
    copy_graph(db, GROUP, 'main_copy')
    source_before = snapshot(db.select_graph(GROUP), GROUP)
    assert asyncio.run(apply(plan, GROUP, 'main_copy', None)) == 0
    out = capsys.readouterr().out
    assert 'every difference is the planned one' in out
    assert 'a second pass found nothing left to do' in out
    assert snapshot(db.select_graph(GROUP), GROUP) == source_before
    g = db.select_graph('main_copy')
    kanada = by_entity(plan)[('Канада', 'Deutsche Welle')]['target_uuid']
    assert g.ro_query(
        'MATCH (e:Episodic {uuid: $e})-[:MENTIONS]->(n:Entity {uuid: $n}) RETURN count(*)',
        {'e': episodes['e2'].uuid, 'n': kanada},
    ).result_set == [[1]]
    assert g.ro_query(
        'MATCH (n:Entity {uuid: $n}) RETURN n.summary', {'n': made['Deutsche Welle'].uuid}
    ).result_set == [['Немецкое СМИ.']]


@needs_db
def test_a_moved_fact_keeps_every_property_and_its_vector(db, capsys):
    from tools.unmerge_old_dedupe import apply

    plan, _, made, _ = asyncio.run(made_plan())
    copy_graph(db, GROUP, 'main_copy')
    move = next(m for m in plan['edge_moves'] if m['old_target'] == made['Ozon'].uuid)
    query = 'MATCH ()-[r:RELATES_TO {uuid: $u}]->() RETURN properties(r)'
    before = db.select_graph('main_copy').ro_query(query, {'u': move['uuid']}).result_set[0][0]
    assert asyncio.run(apply(json.loads(json.dumps(plan)), GROUP, 'main_copy', None)) == 0
    after = db.select_graph('main_copy').ro_query(query, {'u': move['uuid']}).result_set[0][0]
    assert after.pop('source_node_uuid') == move['new_source']
    assert before.pop('source_node_uuid') == move['old_source']
    assert after == before
    same = (
        db.select_graph('main_copy')
        .ro_query(
            'MATCH ()-[r:RELATES_TO {uuid: $u}]->() RETURN vec.cosineDistance(r.fact_embedding, vecf32($v))',
            {'u': move['uuid'], 'v': vector(move['fact'])},
        )
        .result_set[0][0]
    )
    assert same < 1e-6


@needs_db
def test_a_created_node_has_the_shape_graphiti_gives_its_own(db, capsys):
    from tools.unmerge_old_dedupe import apply

    plan, _, made, _ = asyncio.run(made_plan())
    copy_graph(db, GROUP, 'main_copy')
    assert asyncio.run(apply(json.loads(json.dumps(plan)), GROUP, 'main_copy', None)) == 0
    g = db.select_graph('main_copy')

    def shape(u):
        keys, labels = g.ro_query(
            'MATCH (n:Entity {uuid: $u}) RETURN keys(n), labels(n)', {'u': u}
        ).result_set[0]
        return sorted(keys), sorted(labels)

    new = next(n for n in plan['new_nodes'] if n['name'] == 'Пентагон')
    assert shape(new['uuid']) == shape(made['Яндекс'].uuid)


@needs_db
def test_apply_refuses_the_source_without_a_matching_backup(db, capsys):
    from tools.unmerge_old_dedupe import apply

    plan, _, _, _ = asyncio.run(made_plan())
    plan = json.loads(json.dumps(plan))
    assert asyncio.run(apply(plan, GROUP, GROUP, None)) == 2
    assert asyncio.run(apply(plan, GROUP, GROUP, 'no_such_graph')) == 2
    copy_graph(db, GROUP, 'main_backup')
    db.select_graph('main_backup').query("CREATE (:Entity {uuid: 'extra', name: 'extra'})")
    assert asyncio.run(apply(plan, GROUP, GROUP, 'main_backup')) == 2


@needs_db
def test_apply_on_the_source_with_a_fresh_backup_works(db, capsys):
    from tools.unmerge_old_dedupe import apply, snapshot

    plan, _, _, _ = asyncio.run(made_plan())
    copy_graph(db, GROUP, 'main_backup')
    backup = snapshot(db.select_graph('main_backup'), GROUP)
    assert asyncio.run(apply(json.loads(json.dumps(plan)), GROUP, GROUP, 'main_backup')) == 2
    assert (
        asyncio.run(
            apply(json.loads(json.dumps(plan)), GROUP, GROUP, 'main_backup', server_stopped=True)
        )
        == 0
    )
    assert snapshot(db.select_graph('main_backup'), GROUP) == backup


@needs_db
def test_apply_writes_nothing_when_the_graph_moved_on_since_the_plan(db, capsys):
    from tools.unmerge_old_dedupe import apply, snapshot

    plan, _, made, _ = asyncio.run(made_plan())
    copy_graph(db, GROUP, 'main_copy')
    g = db.select_graph('main_copy')
    g.query(
        "MATCH (n:Entity {uuid: $u}) SET n.summary = 'переписано позже'",
        {'u': made['Euronews'].uuid},
    )
    before = snapshot(g, GROUP)
    assert asyncio.run(apply(json.loads(json.dumps(plan)), GROUP, 'main_copy', None)) == 3
    assert snapshot(g, GROUP) == before


@needs_db
def test_apply_finishes_a_run_that_stopped_half_way(db, capsys):
    from tools.unmerge_old_dedupe import apply, config_and_connection, ops_for, write

    plan, _, _, _ = asyncio.run(made_plan())
    plan = json.loads(json.dumps(plan))
    copy_graph(db, GROUP, 'main_copy')
    half = copy.deepcopy(ops_for(plan, GROUP))
    half['edge_moves'] = half['edge_moves'][:1]
    half['mentions_drop'] = []
    half['summary_edits'] = []
    asyncio.run(write(config_and_connection()[1], 'main_copy', half))
    # The half-done graph no longer equals the plan's starting point everywhere, yet every
    # remaining step is either still to do or already done, so preflight lets it finish.
    code = asyncio.run(apply(plan, GROUP, 'main_copy', None))
    out = capsys.readouterr().out
    assert code == 0, out


@needs_db
def test_plan_leaves_alone_a_merge_the_new_dedupe_makes_too(db):
    plan, episodes, made, _ = asyncio.run(made_plan())
    case = by_entity(plan)[('Google Mail', 'Gmail')]
    assert case['decision'] == 'same_as_before'
    gmail = made['Gmail'].uuid
    assert not [m for m in plan['mentions_drop'] if m['entity'] == gmail]
    inc = by_entity(plan)[('Gmail Inc', 'Gmail')]
    assert inc['decision'] == 'new'
    added = [m['entity'] for m in plan['mentions_add'] if m['episode'] == episodes['e8'].uuid]
    assert added == [inc['target_uuid']]


@needs_db
def test_plan_adds_no_second_mention_of_a_node_the_episode_named(db):
    plan, episodes, made, _ = asyncio.run(made_plan())
    assert by_entity(plan)[('BBC News', 'ViT')]['target_uuid'] == made['BBC'].uuid
    assert not [m for m in plan['mentions_add'] if m['episode'] == episodes['e6'].uuid]
    assert [m['entity'] for m in plan['mentions_drop'] if m['episode'] == episodes['e6'].uuid] == [
        made['ViT'].uuid
    ]


@needs_db
def test_a_node_gaining_facts_in_two_episodes_keeps_both(db):
    plan, _, made, sanctions = asyncio.run(made_plan())
    edits = {e['node_uuid']: e for e in plan['summary_edits']}
    yandex = edits[made['Яндекс'].uuid]
    assert yandex['before'] == ''
    assert yandex['after'].split('\n') == [sanctions, 'Яндекс купил долю Ozon']
    assert edits[made['Оксфорд'].uuid]['after'] == 'Университет.'


def test_compare_names_a_mention_the_plan_did_not_add():
    after = snap()
    after['mentions'] = dict(
        after['mentions'],
        x={
            'episode': 'e',
            'entity': 'z',
            'from_episodic': True,
            'to_entity': True,
            'has_uuid': True,
        },
    )
    assert any(p.startswith('mention x:') for p in compare(snap(), after, NO_OPS, 'main'))


@needs_db
def test_wrong_nodes_that_are_each_others_twin_keep_their_mentions(db):
    plan, episodes, made, _ = asyncio.run(made_plan())
    c = by_entity(plan)
    assert c[('Prague', 'Брно')]['target_uuid'] == made['Прага'].uuid
    assert c[('Brno', 'Прага')]['target_uuid'] == made['Брно'].uuid
    e9 = episodes['e9'].uuid
    assert not [m for m in plan['mentions_drop'] if m['episode'] == e9]
    assert not [m for m in plan['mentions_add'] if m['episode'] == e9]


@needs_db
def test_a_new_node_is_dated_by_its_episode(db):
    plan, episodes, _, _ = asyncio.run(made_plan())
    new = next(n for n in plan['new_nodes'] if n['name'] == 'Пентагон')
    assert new['created_at'] == episodes['e1'].created_at.isoformat()


@needs_db
def test_snapshot_counts_a_fact_whose_recorded_ends_disagree(db):
    from tools.unmerge_old_dedupe import snapshot

    g = db.select_graph('probe')
    g.query(
        "CREATE (:Entity {uuid: 'a'})-[:RELATES_TO {uuid: 'r', source_node_uuid: 'a', "
        "target_node_uuid: 'x', episodes: ['e']}]->(:Entity {uuid: 'b'})"
    )
    assert snapshot(g, 'probe')['checks']['edge_endpoint_props_disagree'] == 1


@needs_db
def test_apply_fails_when_a_second_pass_still_writes(db, capsys, monkeypatch):
    import collections

    import tools.unmerge_old_dedupe as tool

    plan, _, _, _ = asyncio.run(made_plan())
    copy_graph(db, GROUP, 'main_copy')
    real = tool.write
    passes = []

    async def flaky(conn, graph, ops):
        done = await real(conn, graph, ops)
        passes.append(1)
        return done if len(passes) == 1 else collections.Counter({'facts moved': 1})

    monkeypatch.setattr(tool, 'write', flaky)
    assert asyncio.run(tool.apply(json.loads(json.dumps(plan)), GROUP, 'main_copy', None)) == 4
    assert 'SECOND PASS WROTE' in capsys.readouterr().out


def by_fact(plan, fact):
    return [m for m in plan['edge_moves'] if m['fact'] == fact]


@needs_db
def test_a_loop_left_by_the_old_merge_is_left_whole(db):
    # A loop E->W needs W named by the episode itself, so the old run filed W twice and wrote the
    # fact into its summary more than once: how many lines it left is not known.
    plan, _, made, _ = asyncio.run(made_plan())
    assert by_entity(plan)[('Шойгу', 'Росатом')]['decision'] == 'skipped'
    assert not by_fact(plan, 'Шойгу посетил Росатом')
    assert made['Росатом'].uuid not in {e['node_uuid'] for e in plan['summary_edits']}


@needs_db
def test_two_merges_into_one_node_with_facts_are_left_whole(db):
    # The old run filed that node twice, so its facts went into its summary twice; how often is not
    # known for sure, so neither merge is undone and both are reported.
    plan, episodes, made, _ = asyncio.run(made_plan())
    c = by_entity(plan)
    assert c[('Мишустин', 'Минфин')]['decision'] == 'skipped'
    assert c[('Набиуллина', 'Минфин')]['decision'] == 'skipped'
    whys = [p for p in plan['problems'] if p.get('entity') in ('Мишустин', 'Набиуллина')]
    assert len(whys) == 2 and all(
        p['skipped'] and 'one node in one episode' in p['why'] for p in whys
    )
    assert not by_fact(plan, 'Мишустин поручил Набиуллиной')
    assert not by_fact(plan, 'Мишустин упомянул Ozon')
    assert made['Минфин'].uuid not in {e['node_uuid'] for e in plan['summary_edits']}
    e13 = episodes['e13'].uuid
    assert not [m for m in plan['mentions_drop'] + plan['mentions_add'] if m['episode'] == e13]


@needs_db
def test_a_fact_folded_into_an_older_one_is_not_moved_and_its_case_is_left_whole(db):
    plan, episodes, made, _ = asyncio.run(made_plan())
    assert not by_fact(plan, 'Донецк связан с Киевом')
    (why,) = [p['why'] for p in plan['problems'] if p.get('entity') == 'Луганск']
    assert why.startswith('fact joined an older one')
    assert 'Луганск' not in {n['name'] for n in plan['new_nodes']}
    e14 = episodes['e14'].uuid
    assert not [m for m in plan['mentions_drop'] if m['episode'] == e14]
    assert not [m for m in plan['mentions_add'] if m['episode'] == e14]


@needs_db
def test_a_node_the_episode_named_keeps_one_mention_not_two(db):
    plan, episodes, made, _ = asyncio.run(made_plan())
    e15 = episodes['e15'].uuid
    dropped = [m for m in plan['mentions_drop'] if m['episode'] == e15]
    assert [m['entity'] for m in dropped] == [made['Ростех'].uuid]


@needs_db
def test_a_case_the_trace_could_not_explain_is_left_whole(db):
    plan, episodes, _, _ = asyncio.run(made_plan())
    e16 = episodes['e16'].uuid
    assert 'Неясно' not in {n['name'] for n in plan['new_nodes']}
    assert not [m for m in plan['mentions_drop'] if m['episode'] == e16]


@needs_db
def test_the_whole_plan_applies_on_a_copy_with_the_new_cases(db, capsys):
    from tools.unmerge_old_dedupe import apply

    plan, _, made, _ = asyncio.run(made_plan())
    copy_graph(db, GROUP, 'main_copy')
    code = asyncio.run(apply(json.loads(json.dumps(plan)), GROUP, 'main_copy', None))
    assert code == 0, capsys.readouterr().out
    g = db.select_graph('main_copy')
    got = g.ro_query('MATCH (n:Entity {uuid: $u}) RETURN n.summary', {'u': made['Минфин'].uuid})
    assert got.result_set == [[made['Минфин'].summary]]
    kanada = by_entity(plan)[('Канада', 'Deutsche Welle')]['target_uuid']
    labels = g.ro_query('MATCH (n:Entity {uuid: $u}) RETURN labels(n)', {'u': kanada}).result_set[
        0
    ][0]
    assert sorted(labels) == ['Entity', 'Organization']


@needs_db
def test_plan_builds_no_indices_on_the_live_graph(db, monkeypatch):
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from tools.unmerge_old_dedupe import make_plan

    async def run():
        driver, episodes, made, sanctions = await seed()
        scheduled = []

        # Recorded when a driver schedules its build, not when the build happens to run.
        def record(self):
            scheduled.append(self._database)

            async def nothing():
                return None

            return nothing()

        monkeypatch.setattr(FalkorDriver, '_build_indices_once', record)
        # Marks left by earlier tests could match a reused object id and hide a missing mark.
        monkeypatch.setattr(FalkorDriver, '_indexed_graphs', set())
        await make_plan(cases(episodes, sanctions), [GROUP], services())
        return scheduled

    assert asyncio.run(run()) == []


@needs_db
def test_snapshot_reads_past_the_result_set_ceiling(db, monkeypatch):
    import tools.unmerge_old_dedupe as tool

    asyncio.run(seed())
    whole = tool.snapshot(db.select_graph(GROUP), GROUP)
    monkeypatch.setattr(tool, 'PAGE', 3)
    paged = tool.snapshot(db.select_graph(GROUP), GROUP)
    assert paged == whole
    assert len(paged['mentions']) > 3


@needs_db
def test_a_new_node_keeps_the_attributes_the_plan_gave_it(db):
    import tools.unmerge_old_dedupe as tool

    asyncio.run(seed())
    node = {
        'uuid': 'n-attr',
        'graph': GROUP,
        'name': 'Атом',
        'group_id': GROUP,
        'labels': ['Entity'],
        'created_at': T0.isoformat(),
        'summary': 'x',
        'name_embedding': vector('Атом'),
        'attributes': {'description': 'Госкорпорация'},
    }
    ops = {
        'new_nodes': [node],
        'mentions_add': [],
        'mentions_drop': [],
        'edge_moves': [],
        'summary_edits': [],
    }
    asyncio.run(tool.write(tool.config_and_connection()[1], GROUP, ops))
    got = db.select_graph(GROUP).ro_query("MATCH (n:Entity {uuid: 'n-attr'}) RETURN n.description")
    assert got.result_set == [['Госкорпорация']]


@needs_db
def test_apply_refuses_a_backup_that_differs_in_content_not_in_counts(db, capsys):
    from tools.unmerge_old_dedupe import apply

    plan, _, made, _ = asyncio.run(made_plan())
    copy_graph(db, GROUP, 'main_backup')
    db.select_graph('main_backup').query(
        "MATCH (n:Entity {uuid: $u}) SET n.summary = 'старое'", {'u': made['Gmail'].uuid}
    )
    code = asyncio.run(
        apply(json.loads(json.dumps(plan)), GROUP, GROUP, 'main_backup', server_stopped=True)
    )
    assert code == 2


@needs_db
def test_apply_builds_no_indices_on_the_graph_it_repairs(db, monkeypatch, capsys):
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from tools.unmerge_old_dedupe import apply

    plan, _, _, _ = asyncio.run(made_plan())
    copy_graph(db, GROUP, 'main_copy')
    scheduled = []

    def record(self):
        scheduled.append(self._database)

        async def nothing():
            return None

        return nothing()

    monkeypatch.setattr(FalkorDriver, '_build_indices_once', record)
    monkeypatch.setattr(FalkorDriver, '_indexed_graphs', set())
    assert asyncio.run(apply(json.loads(json.dumps(plan)), GROUP, 'main_copy', None)) == 0
    assert scheduled == []


@needs_db
def test_apply_refuses_a_backup_that_is_the_source_itself(db, capsys):
    from tools.unmerge_old_dedupe import apply

    plan, _, _, _ = asyncio.run(made_plan())
    code = asyncio.run(
        apply(json.loads(json.dumps(plan)), GROUP, GROUP, GROUP, server_stopped=True)
    )
    assert code == 2


@needs_db
def test_a_target_gains_no_line_for_an_end_it_already_held(db):
    plan, _, made, _ = asyncio.run(made_plan())
    guard = 'Росгвардия передала дела ФСИН'
    c = by_entity(plan)
    assert c[('ФСИН России', 'МВД2')]['target_uuid'] == made['ФСИН'].uuid
    rosgvardia = c[('Росгвардия', 'Минобороны')]['target_uuid']
    (move,) = by_fact(plan, guard)
    assert (move['new_source'], move['new_target']) == (rosgvardia, made['ФСИН'].uuid)
    edits = {e['node_uuid']: e for e in plan['summary_edits']}
    assert made['ФСИН'].uuid not in edits
    assert edits[made['Минобороны'].uuid]['after'] == 'Ведомство.'
    new = {n['name']: n for n in plan['new_nodes']}
    assert new['Росгвардия']['summary'].count(guard) == 1


@needs_db
def test_a_move_that_makes_a_loop_is_planned_as_add_episode_would_have_made_it(db, capsys):
    from tools.unmerge_old_dedupe import apply

    plan, _, made, _ = asyncio.run(made_plan())
    quote = 'BBC News цитирует BBC'
    bbc = made['BBC'].uuid
    (move,) = by_fact(plan, quote)
    assert (move['new_source'], move['new_target']) == (bbc, bbc)
    edits = {e['node_uuid']: e for e in plan['summary_edits']}
    assert edits[bbc]['after'] == f'{quote}\n{quote}'
    assert edits[made['ViT'].uuid]['after'] == ''
    copy_graph(db, GROUP, 'main_copy')
    assert asyncio.run(apply(json.loads(json.dumps(plan)), GROUP, 'main_copy', None)) == 0


@needs_db
def test_plan_shows_a_moved_fact_that_later_episodes_joined(db):
    from tools.unmerge_old_dedupe import report

    plan, _, _, _ = asyncio.run(made_plan())
    (move,) = by_fact(plan, 'Росгвардия передала дела ФСИН')
    assert move['episodes'] == 2
    assert 'joined by 1 later episode' in report(plan)


@needs_db
def test_a_new_node_after_a_failed_check_is_marked(db):
    plan, _, _, _ = asyncio.run(made_plan())
    record = by_entity(plan)[('Чемезов', 'Ростех')]
    assert record['decision'] == 'new'
    assert 'failed the check' in record['dedupe_warning']
    assert not by_entity(plan)[('Пентагон', 'Чернобыльская АЭС')].get('dedupe_warning')


@needs_db
def test_apply_refuses_when_the_graph_grew_a_node_of_a_new_nodes_name(db, capsys):
    from tools.unmerge_old_dedupe import apply

    plan, _, _, _ = asyncio.run(made_plan())
    copy_graph(db, GROUP, 'main_copy')
    db.select_graph('main_copy').query(
        "CREATE (:Entity {uuid: 'later', name: 'пентагон', group_id: 'main'})"
    )
    assert asyncio.run(apply(json.loads(json.dumps(plan)), GROUP, 'main_copy', None)) == 3


@needs_db
def test_apply_refuses_when_a_target_is_gone(db, capsys):
    from tools.unmerge_old_dedupe import apply

    plan, _, made, _ = asyncio.run(made_plan())
    copy_graph(db, GROUP, 'main_copy')
    db.select_graph('main_copy').query(
        'MATCH (n:Entity {uuid: $u}) DETACH DELETE n', {'u': made['Яндекс'].uuid}
    )
    assert asyncio.run(apply(json.loads(json.dumps(plan)), GROUP, 'main_copy', None)) == 3


@needs_db
def test_a_moved_fact_keeps_the_recorded_end_it_did_not_move(db, capsys):
    from tools.unmerge_old_dedupe import apply

    plan, _, made, _ = asyncio.run(made_plan())
    copy_graph(db, GROUP, 'main_copy')
    move = next(m for m in plan['edge_moves'] if m['old_target'] == made['Ozon'].uuid)
    g = db.select_graph('main_copy')
    g.query(
        "MATCH ()-[r:RELATES_TO {uuid: $u}]->() SET r.target_node_uuid = 'stale'",
        {'u': move['uuid']},
    )
    assert asyncio.run(apply(json.loads(json.dumps(plan)), GROUP, 'main_copy', None)) == 0, (
        capsys.readouterr().out
    )
    got = g.ro_query(
        'MATCH ()-[r:RELATES_TO {uuid: $u}]->() RETURN r.source_node_uuid, r.target_node_uuid',
        {'u': move['uuid']},
    ).result_set
    assert got == [[move['new_source'], 'stale']]


@needs_db
def test_mentions_without_uuid_are_counted_one_by_one(db):
    from tools.unmerge_old_dedupe import snapshot

    g = db.select_graph('probe')
    g.query(
        "CREATE (e:Episodic {uuid: 'e'}), (n:Entity {uuid: 'n', group_id: 'probe'}), "
        '(e)-[:MENTIONS]->(n), (e)-[:MENTIONS]->(n)'
    )
    snap = snapshot(g, 'probe')
    assert len(snap['mentions']) == 2
    assert snap['checks']['entity_without_mention'] == 0
    assert snap['checks']['mention_without_uuid'] == 2


@needs_db
def test_a_mention_to_drop_without_uuid_stops_apply(db, capsys):
    from tools.unmerge_old_dedupe import apply, make_plan

    async def run():
        driver, episodes, made, sanctions = await seed()
        await driver.execute_query(
            'MATCH (e:Episodic {uuid: $e})-[m:MENTIONS]->(n:Entity {uuid: $n}) '
            'DELETE m CREATE (e)-[:MENTIONS]->(n)',
            e=episodes['e1'].uuid,
            n=made['Чернобыльская АЭС'].uuid,
        )
        return await make_plan(cases(episodes, sanctions), [GROUP], services())

    plan = asyncio.run(run())
    assert [p for p in plan['problems'] if 'without a uuid' in p['why'] and not p['skipped']]
    copy_graph(db, GROUP, 'main_copy')
    assert asyncio.run(apply(json.loads(json.dumps(plan)), GROUP, 'main_copy', None)) == 2


@needs_db
def test_apply_finishes_an_interrupted_write_on_the_source(db, capsys):
    from tools.unmerge_old_dedupe import apply, config_and_connection, ops_for, write

    plan, _, _, _ = asyncio.run(made_plan())
    plan = json.loads(json.dumps(plan))
    copy_graph(db, GROUP, 'main_backup')
    half = copy.deepcopy(ops_for(plan, GROUP))
    half['edge_moves'] = half['edge_moves'][:2]
    half['mentions_drop'] = half['mentions_drop'][:3]
    half['summary_edits'] = []
    asyncio.run(write(config_and_connection()[1], GROUP, half))
    code = asyncio.run(apply(plan, GROUP, GROUP, 'main_backup', server_stopped=True))
    assert code == 0, capsys.readouterr()


@needs_db
def test_plan_leaves_out_a_graph_that_does_not_exist(db):
    from tools.unmerge_old_dedupe import make_plan

    async def run():
        driver, episodes, made, sanctions = await seed()
        return await make_plan(cases(episodes, sanctions), [GROUP, 'no_such_graph'], services())

    plan = asyncio.run(run())
    assert plan['graphs'] == [GROUP]
    assert plan['missing_graphs'] == ['no_such_graph']


@needs_db
def test_preflight_names_a_node_the_plan_sends_to_that_is_gone(db):
    from tools.unmerge_old_dedupe import preflight

    g = db.select_graph('probe')
    g.query("CREATE (:Episodic {uuid: 'e'}), (:Entity {uuid: 'here', name: 'Здесь'})")
    ops = {
        'new_nodes': [],
        'mentions_add': [{'uuid': 'm', 'graph': 'probe', 'episode': 'e', 'entity': 'ghost'}],
        'mentions_drop': [],
        'edge_moves': [],
        'summary_edits': [],
    }
    assert preflight(g, ops) == ['node ghost the plan sends to is gone']


@needs_db
def test_a_mention_without_uuid_is_the_one_kept_when_one_of_two_goes(db):
    from tools.unmerge_old_dedupe import make_plan

    async def run():
        driver, episodes, made, sanctions = await seed()
        await driver.execute_query(
            'MATCH (e:Episodic {uuid: $e})-[m:MENTIONS]->(n:Entity {uuid: $n}) '
            'WITH e, n, m LIMIT 1 DELETE m CREATE (e)-[:MENTIONS]->(n)',
            e=episodes['e15'].uuid,
            n=made['Ростех'].uuid,
        )
        plan = await make_plan(cases(episodes, sanctions), [GROUP], services())
        return plan, episodes['e15'].uuid

    plan, e15 = asyncio.run(run())
    assert not [p for p in plan['problems'] if 'without a uuid' in p['why']]
    dropped = [m for m in plan['mentions_drop'] if m['episode'] == e15]
    assert len(dropped) == 1 and dropped[0]['uuid'] is not None


@needs_db
def test_namesakes_already_there_at_plan_time_do_not_stop_apply(db, capsys):
    from tools.unmerge_old_dedupe import apply, make_plan, report

    async def run():
        driver, episodes, made, sanctions = await seed()
        # Two nodes named like Пентагон before the plan: Graphiti's exact match cannot pick one,
        # and the model says neither, so the plan creates a third.
        await driver.execute_query(
            "CREATE (:Entity {uuid: 'p1', name: 'Пентагон', group_id: 'main', summary: '', labels: ['Entity']}), "
            "(:Entity {uuid: 'p2', name: 'Пентагон', group_id: 'main', summary: '', labels: ['Entity']})"
        )
        return await make_plan(cases(episodes, sanctions), [GROUP], services())

    plan = asyncio.run(run())
    new = {n['name']: n for n in plan['new_nodes']}
    assert new['Пентагон']['namesakes'] == ['p1', 'p2']
    assert 'namesakes' in report(plan)
    copy_graph(db, GROUP, 'main_copy')
    assert asyncio.run(apply(json.loads(json.dumps(plan)), GROUP, 'main_copy', None)) == 0


@needs_db
def test_a_summary_rewritten_without_moved_facts_says_so(db):
    plan, _, made, _ = asyncio.run(made_plan())
    edits = {e['node_uuid']: e for e in plan['summary_edits']}
    assert edits[made['Прага'].uuid]['why'] == 'summary as add_episode would write it'


@needs_db
def test_removing_the_last_mention_of_a_node_stops_apply(db, capsys):
    from tools.unmerge_old_dedupe import apply, make_plan

    async def run():
        driver, episodes, made, sanctions = await seed()
        # Чернобыльская АЭС keeps no mention but the one the undone merge gave it.
        await driver.execute_query(
            'MATCH (e:Episodic)-[m:MENTIONS]->(n:Entity {uuid: $n}) WHERE e.uuid <> $e DELETE m',
            n=made['Чернобыльская АЭС'].uuid,
            e=episodes['e1'].uuid,
        )
        one = [c for c in cases(episodes, sanctions) if c['entity'] == 'Пентагон']
        return await make_plan(one, [GROUP], services())

    plan = asyncio.run(run())
    orphan = [p for p in plan['problems'] if 'last mention' in p['why']]
    assert orphan and not orphan[0]['skipped']
    copy_graph(db, GROUP, 'main_copy')
    assert asyncio.run(apply(json.loads(json.dumps(plan)), GROUP, 'main_copy', None)) == 2


@needs_db
def test_a_target_summary_the_model_rewrites_says_so(db):
    from tools.unmerge_old_dedupe import make_plan

    async def run():
        driver, episodes, made, sanctions = await seed()
        # Long enough that Graphiti no longer appends facts but asks the model for a new summary.
        await driver.execute_query(
            'MATCH (n:Entity {uuid: $u}) SET n.summary = $s', u=made['Яндекс'].uuid, s='x' * 1995
        )
        plan = await make_plan(cases(episodes, sanctions), [GROUP], services())
        return plan, made

    plan, made = asyncio.run(run())
    edits = {e['node_uuid']: e for e in plan['summary_edits']}
    assert edits[made['Яндекс'].uuid]['why'] == 'rewritten by the model with facts that moved here'


@needs_db
def test_a_target_summary_with_facts_appended_says_so(db):
    plan, _, made, _ = asyncio.run(made_plan())
    edits = {e['node_uuid']: e for e in plan['summary_edits']}
    assert edits[made['Яндекс'].uuid]['why'] == 'facts that moved here, appended'


@needs_db
def test_a_merge_the_model_made_twice_into_one_node_with_facts_is_left_whole(db):
    from tools.unmerge_old_dedupe import make_plan

    async def run():
        driver, episodes, made, sanctions = await seed()
        # The trace shows the answer also sent another entity into Deutsche Welle.
        crowded = [
            dict(c, merges_into_wrong=2) if c['entity'] == 'Канада' else c
            for c in cases(episodes, sanctions)
        ]
        return await make_plan(crowded, [GROUP], services())

    plan = asyncio.run(run())
    assert by_entity(plan)[('Канада', 'Deutsche Welle')]['decision'] == 'skipped'


@needs_db
def test_a_moved_fact_the_target_already_has_is_shown(db):
    from tools.unmerge_old_dedupe import make_plan, report

    async def run():
        driver, episodes, made, sanctions = await seed()
        await driver.execute_query(
            'MATCH (a:Entity {uuid: $a}), (b:Entity {uuid: $b}) '
            "CREATE (a)-[:RELATES_TO {uuid: 'dup', fact: $f, episodes: ['x'], "
            'source_node_uuid: $a, target_node_uuid: $b}]->(b)',
            a=made['BBC'].uuid,
            b=made['BBC'].uuid,
            f='BBC News цитирует BBC',
        )
        return await make_plan(cases(episodes, sanctions), [GROUP], services())

    plan = asyncio.run(run())
    (move,) = by_fact(plan, 'BBC News цитирует BBC')
    assert move['duplicates_existing'] == ['dup']
    assert 'same fact already there' in report(plan)
