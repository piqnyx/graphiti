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
from tools.unmerge_old_dedupe import compare, norm, remove_fact_lines

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
            'm': {'episode': 'e', 'entity': 'w', 'from_episodic': True, 'to_entity': True}
        },
        'checks': {'entity_without_mention': 0, 'edge_self_loop': 0},
    }
    base.update(over)
    return base


NO_OPS = {
    'new_nodes': [],
    'summary_edits': [],
    'edge_moves': [],
    'mentions_add': [],
    'mentions_drop': [],
}


def test_compare_accepts_identical_snapshots():
    assert compare(snap(), snap(), NO_OPS) == []


def test_compare_names_a_summary_change_the_plan_did_not_make():
    after = snap()
    after['entities'] = {'w': dict(after['entities']['w'], summary='other')}
    assert any('beyond the plan' in p for p in compare(snap(), after, NO_OPS))


def test_compare_names_a_fact_moved_without_the_plan():
    after = snap()
    after['edges'] = {'r': dict(after['edges']['r'], source='x', source_prop='x')}
    assert any('fact r' in p for p in compare(snap(), after, NO_OPS))


def test_compare_names_a_lost_episode():
    after = snap(labels={"['Episodic']": 1, "['Entity']": 2})
    assert any('Episodic' in p for p in compare(snap(), after, NO_OPS))


def test_compare_names_a_new_orphan_entity():
    after = snap(checks={'entity_without_mention': 1, 'edge_self_loop': 0})
    assert any('entity_without_mention' in p for p in compare(snap(), after, NO_OPS))


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
                hit = next((c for c in e.get('candidates', []) if c['name'] == want), None)
                rows.append(
                    {
                        'id': e['id'],
                        'name': e['name'],
                        'duplicate_candidate_id': hit['candidate_id'] if hit else -1,
                        'duplicate_candidate_name': hit['name'] if hit else '',
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
            or {'BBC News': 'BBC', 'Google Mail': 'Gmail', 'Prague': 'Прага', 'Brno': 'Брно'}
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
        ],
        facts=[('Чернобыльская АЭС', 'Киев', 'Чернобыльская АЭС находится рядом с Киевом')],
        summaries={
            'Deutsche Welle': 'Немецкое СМИ.',
            'Euronews': 'Европейское СМИ.',
            'Оксфорд': 'Университет.',
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
    await episode('e6', 60, 'BBC News сообщает, BBC подтверждает', ['ViT', 'BBC'])
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
        {'entity': 'BBC News', 'wrong': 'ViT', 'episode_content': text('e6'), 'entity_facts': []},
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
        {'entity': 'Prague', 'wrong': 'Брно', 'episode_content': text('e9'), 'entity_facts': []},
        {'entity': 'Brno', 'wrong': 'Прага', 'episode_content': text('e9'), 'entity_facts': []},
        {'entity': 'Повтор', 'wrong': 'ViT', 'episode_content': text('e10'), 'entity_facts': []},
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
    assert {n['name'] for n in plan['new_nodes']} == {'ВСУ', 'Канада', 'Пентагон', 'Swift'}
    assert sorted(p['entity'] for p in plan['problems'] if 'entity' in p) == [
        'Нет такого',
        'Повтор',
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
    assert asyncio.run(apply(json.loads(json.dumps(plan)), GROUP, GROUP, 'main_backup')) == 0
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
    assert not [m for m in plan['mentions_add'] if m['episode'] == episodes['e8'].uuid]


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
        x={'episode': 'e', 'entity': 'z', 'from_episodic': True, 'to_entity': True},
    )
    assert any('mentions' in p for p in compare(snap(), after, NO_OPS))


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
