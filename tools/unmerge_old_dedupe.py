#!/usr/bin/env python3
"""Undo entity merges the old dedupe made into a node that was never that entity's candidate.

Before the dedupe rework every entity in an episode was shown one shared list of
candidates, numbered from zero like the entities themselves, and the answer was
taken unchecked. The model sometimes answered with the entity's own number, which
pointed at somebody else's candidate: "Пентагон" was merged into "Чернобыльская
АЭС". The graph kept the episode's mention and the entity's facts on that wrong
node, appended those facts to its summary, and never created the entity itself.

This tool puts such a merge right as if the current dedupe had made it from the
start. Only the dedupe is repeated, with Graphiti's own functions against the
graph as it is now: the entity goes to the twin the new dedupe finds, or becomes a
node of its own. The dedupe runs on a scratch copy of the graph where the nodes the
plan creates appear as it goes, episode by episode, so a later entity can find one. Extraction is not repeated -- the facts it produced were right,
only the node they hung on was wrong.

    plan    read the graph, ask the new dedupe on a scratch copy it removes after, write a plan file
    apply   carry out a plan file on one graph: a copy first, the real graph with --backup
    check   integrity counters of a graph, read only

What one undone merge changes, and nothing else:
  - the entity's node is created (or found) and the episode mentions it;
  - the episode's mentions of the wrong node shrink by the merges undone, down to one if the
    episode named that node itself (Graphiti wrote one mention per entity merged into it);
  - every fact extraction gave the entity moves the end the entity held to the right node -- the
    same edge, uuid and every property kept; the other end stays, even when it is the wrong node;
  - the lines those facts appended to the wrong node's summary are removed, one per end it lost.
Episodes, sagas, NEXT_EPISODE, HAS_EPISODE, fact texts, embeddings and provenance are never touched.

A case is left whole, and reported, when the trace cannot say what extraction gave the entity, when
its episode or wrong node cannot be found exactly once, when one of its facts was not created by its
episode (dedupe folded it into an older fact of the wrong node, which is not the entity's to take
away), or when the old run filed the wrong node more than once in the episode and facts are involved
(several merges into it, or the episode naming it): how many summary lines they left is not known.
Any other problem in the plan -- removing a node's last mention, a mention without a uuid that would
have to go -- stops apply.

apply on the graph the plan was made from needs --server-stopped (graphiti-server down: its own
writes in between would undo a summary edit or duplicate a moved fact) and --backup naming another
graph, a copy that leads to the same graph under the plan (so an interrupted write can be finished).
After writing, the whole graph must equal the plan laid over the graph before, exactly.

The cases file is made from the Graphiti trace: one entry per merge to undo, with the entity,
the node it was wrongly merged into, the episode text, the facts extraction gave the entity by
name, whether the episode named the wrong node by itself, and how many entities the model's
answer sent into it.

Run it inside the Graphiti image, with the configuration the server uses:

    docker compose run --rm --no-deps -T -v /tmp/dedupe-old:/work --entrypoint python \\
        graphiti-server /work/unmerge_old_dedupe.py plan --cases /work/cases.json --out /work/plan.json
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import copy
import json
import logging
import os
import sys
import uuid as uuidlib
from datetime import datetime, timezone
from typing import Any

PLAN_VERSION = 1
# The server caps a result at RESULTSET_SIZE rows (10000 by default) and says nothing when it does,
# so every whole-graph read goes in pages and is checked against a count.
PAGE = 5000
# Graphiti appends a node's new facts to its summary through the edges it is handed; an edge whose other
# end is this placeholder counts for one node only, which is how a gained end is handed over exactly.
NO_NODE = '__unmerge_no_node__'
# The plan's scratch copy of a graph is named after it with this ending, and removed when it is done.
SCRATCH_SUFFIX = '__unmerge_plan'


# ---------------------------------------------------------------- pure helpers


def norm(name: str | None) -> str:
    return ' '.join((name or '').lower().replace('ё', 'е').split())


def remove_fact_lines(summary: str | None, facts: list[str]) -> tuple[str, list[str]]:
    """Remove one line per fact, the last occurrence first; report facts with no line.

    Graphiti appends the facts of a node's new edges to its summary one per line
    (`summary + '\\n' + facts`, stripped) when the result stays short, so a fact that
    moved away left exactly its own text as a line. A summary rewritten by the model
    instead carries no such line, and the fact is reported rather than guessed at.
    """
    lines = (summary or '').split('\n')
    missing = []
    for fact in facts:
        wanted = (fact or '').strip()
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == wanted:
                del lines[i]
                break
        else:
            missing.append(fact)
    return '\n'.join(lines).strip(), missing


class _Collect(logging.Handler):
    """Keeps the warnings Graphiti's dedupe logs, so a decision taken after a failed check shows."""

    def __init__(self, into: list[str]):
        super().__init__(logging.WARNING)
        self.into = into

    def emit(self, record: logging.LogRecord) -> None:
        self.into.append(record.getMessage())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- database access


def config_and_connection() -> tuple[Any, dict[str, Any]]:
    """The server's own configuration and database settings, or a test override."""
    if os.environ.get('UNMERGE_DB_HOST'):
        return None, {
            'host': os.environ['UNMERGE_DB_HOST'],
            'port': int(os.environ.get('UNMERGE_DB_PORT', '6379')),
            'username': None,
            'password': os.environ.get('UNMERGE_DB_PASSWORD') or None,
        }
    sys.path.insert(0, os.environ.get('UNMERGE_MCP_SRC', '/opt/graphiti/mcp_server/src'))
    os.environ.setdefault('CONFIG_PATH', '/etc/graphiti/config.yaml')
    from config.schema import GraphitiConfig
    from services.factories import DatabaseDriverFactory

    config = GraphitiConfig()
    db = DatabaseDriverFactory.create_config(config.database)
    return config, {k: db[k] for k in ('host', 'port', 'username', 'password')}


def falkor(conn: dict[str, Any]):
    from falkordb import FalkorDB

    return FalkorDB(
        host=conn['host'], port=conn['port'], username=conn['username'], password=conn['password']
    )


def rows(graph, query: str, **params) -> list[list[Any]]:
    return graph.ro_query(query, params).result_set


def all_rows(graph, match: str, returns: str, order: str) -> list[list[Any]]:
    """Every row of `MATCH ... RETURN ...`, read in pages and checked against its count."""
    out: list[list[Any]] = []
    while True:
        page = rows(
            graph, f'{match} RETURN {returns} ORDER BY {order} SKIP $s LIMIT $l', s=len(out), l=PAGE
        )
        out += page
        if len(page) < PAGE:
            break
    # count(*) is not used: FalkorDB 4.2.1 counts two parallel edges between one pair as one there.
    total = rows(graph, f'{match} RETURN count({order.split(".")[0]})')[0][0]
    if total != len(out):
        raise RuntimeError(f'read {len(out)} rows of {total}: {match}')
    return out


# ---------------------------------------------------------------- integrity snapshot


def snapshot(graph, group_id: str) -> dict[str, Any]:
    """Everything apply compares before and after. Read only."""
    snap: dict[str, Any] = {}
    # Keyed by the sorted labels: a typed node gets its type as a label next to Entity.
    snap['labels'] = collections.Counter()
    for lab, c in rows(graph, 'MATCH (n) RETURN labels(n), count(n)'):
        snap['labels'][str(sorted(lab))] += c
    snap['labels'] = dict(snap['labels'])
    snap['types'] = {t: c for t, c in rows(graph, 'MATCH ()-[r]->() RETURN type(r), count(r)')}
    # FalkorDB gives no size for a stored vector; its type is what says it is one.
    snap['entities'] = {
        u: {'name': n, 'summary': s, 'group_id': g, 'labels': sorted(lab or []), 'embedding': t}
        for u, n, s, g, lab, t in all_rows(
            graph,
            'MATCH (n:Entity)',
            'n.uuid, n.name, n.summary, n.group_id, n.labels, typeof(n.name_embedding)',
            'n.uuid',
        )
    }
    snap['edges'] = {
        u: {
            'source': a,
            'target': b,
            'source_prop': sp,
            'target_prop': tp,
            'fact': f,
            'episodes': e,
            'group_id': g,
            'name': nm,
            'valid_at': va,
            'invalid_at': ia,
            'expired_at': xa,
            'has_embedding': he,
        }
        for u, a, b, sp, tp, f, e, g, nm, va, ia, xa, he in all_rows(
            graph,
            'MATCH (a)-[r:RELATES_TO]->(b)',
            'r.uuid, a.uuid, b.uuid, r.source_node_uuid, r.target_node_uuid, r.fact, r.episodes, '
            'r.group_id, r.name, r.valid_at, r.invalid_at, r.expired_at, typeof(r.fact_embedding)',
            'r.uuid',
        )
    }
    # A mention written without a uuid (the fork's merge tool MERGEs them so) is keyed by its id.
    snap['mentions'] = {
        (u if u is not None else f'id:{i}'): {
            'episode': e,
            'entity': n,
            'from_episodic': fe,
            'to_entity': te,
            'has_uuid': u is not None,
        }
        for u, i, e, n, fe, te in all_rows(
            graph,
            'MATCH (a)-[m:MENTIONS]->(b)',
            "m.uuid, id(m), a.uuid, b.uuid, 'Episodic' IN labels(a), 'Entity' IN labels(b)",
            'id(m)',
        )
    }
    snap['episodes'] = sorted(
        u for (u,) in all_rows(graph, 'MATCH (e:Episodic)', 'e.uuid', 'e.uuid')
    )
    snap['checks'] = checks_of(snap, group_id)
    return snap


def checks_of(snap: dict, group_id: str) -> dict[str, Any]:
    """Integrity counters of a snapshot, real or expected."""
    episodes = set(snap['episodes'])
    edges, entities, mentions = snap['edges'].values(), snap['entities'], snap['mentions'].values()
    mentioned = collections.Counter(m['entity'] for m in mentions)
    return {
        'edge_endpoint_props_disagree': sum(
            1 for e in edges if e['source_prop'] != e['source'] or e['target_prop'] != e['target']
        ),
        'edge_episode_missing': sum(
            1 for e in edges for x in (e['episodes'] or []) if x not in episodes
        ),
        'edge_without_episodes': sum(1 for e in edges if not e['episodes']),
        'edge_embedding_types': sorted({str(e['has_embedding']) for e in edges}),
        'edge_self_loop': sum(1 for e in edges if e['source'] == e['target']),
        'edge_group_other': sum(1 for e in edges if e['group_id'] != group_id),
        'entity_without_mention': sum(1 for u in entities if not mentioned[u]),
        'entity_embedding_types': sorted({str(e['embedding']) for e in entities.values()}),
        'entity_group_other': sum(1 for e in entities.values() if e['group_id'] != group_id),
        'mention_not_episode_to_entity': sum(
            1 for m in mentions if not (m['from_episodic'] and m['to_entity'])
        ),
        'mention_without_uuid': sum(1 for m in mentions if not m['has_uuid']),
        'mention_duplicate_pair': sum(
            c - 1
            for c in collections.Counter((m['episode'], m['entity']) for m in mentions).values()
            if c > 1
        ),
    }


def embedding_sizes(graph, uuids: list[str]) -> dict[str, int]:
    """Length of each node's name vector, read back whole; only for the few nodes asked."""
    return {
        u: len(v or [])
        for u, v in rows(
            graph,
            'MATCH (n:Entity) WHERE n.uuid IN $u RETURN n.uuid, n.name_embedding',
            u=uuids,
        )
    }


def expected_after(before: dict, ops: dict, group_id: str) -> dict[str, Any]:
    """The snapshot the plan leads to from `before`: every step laid over it, done or not yet."""
    exp = copy.deepcopy(
        {k: before[k] for k in ('labels', 'types', 'entities', 'edges', 'mentions', 'episodes')}
    )
    for n in ops['new_nodes']:
        if n['uuid'] not in exp['entities']:
            key = str(sorted(set(n['labels']) | {'Entity'}))
            exp['labels'][key] = exp['labels'].get(key, 0) + 1
        exp['entities'][n['uuid']] = {
            'name': n['name'],
            'summary': n['summary'],
            'group_id': n['group_id'],
            'labels': sorted(set(n['labels']) | {'Entity'}),
            'embedding': 'Vectorf32',
        }
    for m in ops['mentions_add']:
        if m['uuid'] not in exp['mentions']:
            exp['types']['MENTIONS'] = exp['types'].get('MENTIONS', 0) + 1
        exp['mentions'][m['uuid']] = {
            'episode': m['episode'],
            'entity': m['entity'],
            'from_episodic': True,
            'to_entity': True,
            'has_uuid': True,
        }
    for m in ops['mentions_drop']:
        if exp['mentions'].pop(m['uuid'], None) is not None:
            exp['types']['MENTIONS'] -= 1
    for mv in ops['edge_moves']:
        e = exp['edges'].get(mv['uuid'])
        if e is None:
            continue
        if (e['source'], e['target']) == (mv['old_source'], mv['old_target']):
            # Only the end that moves has its recorded uuid rewritten.
            if mv['new_source'] != mv['old_source']:
                e['source_prop'] = mv['new_source']
            if mv['new_target'] != mv['old_target']:
                e['target_prop'] = mv['new_target']
            e['source'], e['target'] = mv['new_source'], mv['new_target']
    for s in ops['summary_edits']:
        if s['node_uuid'] in exp['entities']:
            exp['entities'][s['node_uuid']]['summary'] = s['after']
    exp['checks'] = checks_of(exp, group_id)
    return exp


def compare(before: dict, after: dict, ops: dict, group_id: str) -> list[str]:
    """Every way the graph after differs from what the plan leads to from the graph before."""
    exp = expected_after(before, ops, group_id)
    bad: list[str] = []
    for key in ('labels', 'types', 'episodes', 'checks'):
        if exp[key] != after[key]:
            bad.append(f'{key}: expected {exp[key]}, found {after[key]}')
    for key, what in (('entities', 'entity'), ('edges', 'fact'), ('mentions', 'mention')):
        want, got = exp[key], after[key]
        for u in sorted(set(want) | set(got)):
            if want.get(u) != got.get(u):
                bad.append(f'{what} {u}: expected {want.get(u)}, found {got.get(u)}')
    return bad


# ---------------------------------------------------------------- plan


def locate(db, graphs: list[str], case: dict) -> tuple[str, str, str, str] | str:
    """(graph, episode uuid, wrong node uuid, its summary) or the reason it cannot be found."""
    head = (case.get('episode_content') or '').strip()[:200]
    if not head:
        return 'no episode text'
    found = []
    for name in graphs:
        found += [
            (name, u)
            for (u,) in rows(
                db.select_graph(name),
                'MATCH (e:Episodic) WHERE e.content CONTAINS $h RETURN e.uuid',
                h=head,
            )
        ]
    if len(found) != 1:
        return f'{len(found)} episodes carry this text'
    graph, episode = found[0]
    wrong = rows(
        db.select_graph(graph),
        'MATCH (e:Episodic {uuid: $e})-[:MENTIONS]->(n:Entity) WHERE n.name = $w '
        'RETURN DISTINCT n.uuid, n.summary',
        e=episode,
        w=case['wrong'],
    )
    if len(wrong) != 1:
        return f'the episode mentions {len(wrong)} nodes named {case["wrong"]!r}'
    return graph, episode, wrong[0][0], wrong[0][1] or ''


def fact_role(entity: str, fact: dict) -> str | None:
    """Which end of the extracted fact the entity was: 'source', 'target', or None."""
    if norm(fact.get('source')) == norm(entity):
        return 'source'
    if norm(fact.get('target')) == norm(entity):
        return 'target'
    return None


def ends(source: str, target: str) -> collections.Counter:
    """How many ends of an edge each node holds. Graphiti files a loop under its node twice."""
    return collections.Counter([source, target])


async def make_plan(cases: list[dict], graphs: list[str], services: Any = None) -> dict:
    """Read the graph and ask the current dedupe; return the plan. Writes nothing to the graph."""
    from falkordb.asyncio import FalkorDB as AsyncFalkorDB

    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from graphiti_core.edges import EntityEdge
    from graphiti_core.graphiti_types import GraphitiClients
    from graphiti_core.nodes import EntityNode, EpisodicNode
    from graphiti_core.search.search_utils import RELEVANT_SCHEMA_LIMIT
    from graphiti_core.tracer import create_tracer
    from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
    from graphiti_core.utils.maintenance.node_operations import (
        extract_attributes_from_nodes,
        resolve_extracted_nodes,
    )

    config, conn = config_and_connection()
    db = falkor(conn)
    present = set(db.list_graphs())
    missing = [name for name in graphs if name not in present]
    graphs = [name for name in graphs if name in present]
    if not graphs:
        raise SystemExit(f'none of the graphs exists: {missing}')
    if services is None:
        from services.factories import CrossEncoderFactory, EmbedderFactory, LLMClientFactory
        from utils.type_config import build_entity_types

        services = {
            'llm': LLMClientFactory.create(config.llm),
            'embedder': EmbedderFactory.create(config.embedder),
            'cross_encoder': CrossEncoderFactory.create(config.llm, config.embedder),
            'entity_types': build_entity_types(config.graphiti.entity_types),
        }

    # The dedupe runs on a scratch copy of each graph, and the nodes this plan creates are put into it
    # as it goes: a later episode then finds them by Graphiti's own search -- an entity's fifteen
    # nearest above the threshold, an exact name -- as it would have in a run that went episode by
    # episode. The graphs themselves are only read; the copies are removed at the end.
    scratch = {name: f'{name}{SCRATCH_SUFFIX}' for name in graphs}
    for name, copy_name in scratch.items():
        if copy_name in present:
            db.select_graph(copy_name).delete()
        db.connection.execute_command('GRAPH.COPY', name, copy_name)
    # One connection for every copy. A new driver schedules an index build on its graph; marking the
    # copies as built keeps the plan from sending CREATE INDEX (the copy has the graph's indices).
    client = AsyncFalkorDB(
        host=conn['host'], port=conn['port'], username=conn['username'], password=conn['password']
    )
    for copy_name in scratch.values():
        FalkorDriver._indexed_graphs.add((id(client), copy_name))
    drivers = {name: FalkorDriver(falkor_db=client, database=scratch[name]) for name in graphs}

    plan: dict[str, Any] = {
        'version': PLAN_VERSION,
        'made_at': now_iso(),
        'graphs': graphs,
        'missing_graphs': missing,
        'cases': [],
        'new_nodes': [],
        'mentions_add': [],
        'mentions_drop': [],
        'edge_moves': [],
        'summary_edits': [],
        'problems': [],
    }

    def problem(case: dict, why: str, skipped: bool = True) -> None:
        plan['problems'].append(
            {
                'case': case['index'],
                'entity': case['entity'],
                'wrong': case['wrong'],
                'why': why,
                'skipped': skipped,
            }
        )

    # Locate everything first; a case that cannot be located is reported, never guessed at.
    located: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for i, case in enumerate(cases):
        case = dict(case, index=i)
        if case.get('unknown'):
            problem(case, 'the trace did not show what extraction gave this entity')
            continue
        where = locate(db, graphs, case)
        if isinstance(where, str):
            problem(case, where)
            continue
        graph, episode, wrong_uuid, wrong_summary = where
        located[(graph, episode)].append(
            dict(
                case,
                graph=graph,
                episode=episode,
                wrong_uuid=wrong_uuid,
                wrong_summary=wrong_summary,
            )
        )

    def episode_time(key):
        graph, episode = key
        found = rows(
            db.select_graph(graph), 'MATCH (e:Episodic {uuid: $e}) RETURN e.valid_at', e=episode
        )
        return found[0][0] or ''

    # (graph, uuid) -> the EntityNode planned as new
    planned_new: dict[tuple[str, str], Any] = {}
    # (graph, node uuid) -> lines to remove, summaries before and after
    summaries: dict[tuple[str, str], dict] = {}
    moves: dict[str, dict] = {}  # edge uuid -> move

    try:
        for key in sorted(located, key=episode_time):
            graph, episode_uuid = key
            group_cases = located[key]
            driver = drivers[graph]
            clients = GraphitiClients(
                driver=driver,
                llm_client=services['llm'],
                embedder=services['embedder'],
                cross_encoder=services['cross_encoder'],
                tracer=create_tracer(None, 'unmerge'),
            )
            g = db.select_graph(graph)
            episode = await EpisodicNode.get_by_uuid(driver, episode_uuid)
            valid_at, source = rows(
                g, 'MATCH (e:Episodic {uuid: $e}) RETURN e.valid_at, e.source', e=episode_uuid
            )[0]
            previous_uuids = [
                u
                for (u,) in rows(
                    g,
                    'MATCH (e:Episodic) WHERE e.valid_at < $t AND e.source = $s AND e.uuid <> $e '
                    'RETURN e.uuid ORDER BY e.valid_at DESC LIMIT $n',
                    t=valid_at,
                    s=source,
                    e=episode_uuid,
                    n=RELEVANT_SCHEMA_LIMIT,
                )
            ]
            previous = (
                await EpisodicNode.get_by_uuids(driver, previous_uuids) if previous_uuids else []
            )
            previous.sort(key=lambda e: e.valid_at)

            # The facts each case's entity was given, found on its wrong node by text and by the end
            # it held, and only where this episode created the edge. A fact this episode merely joined
            # (dedupe judged it the same as an older one) is not the entity's to take away, and the
            # case is then left whole rather than half undone.
            fact_ends: dict[
                int, list[tuple[str, str]]
            ] = {}  # case index -> [(edge uuid, 'source'|'target')]
            on_episode = rows(
                g,
                'MATCH (a)-[r:RELATES_TO]->(b) WHERE $e IN r.episodes '
                'RETURN r.uuid, r.fact, a.uuid, b.uuid, r.episodes[0] = $e, size(r.episodes)',
                e=episode_uuid,
            )
            for c in group_cases:
                wanted = collections.Counter()
                bad = None
                for f in c.get('entity_facts') or []:
                    role = fact_role(c['entity'], f)
                    if role is None:
                        bad = f'the entity is neither end of an extracted fact: {(f.get("fact") or "")[:120]}'
                        break
                    wanted[((f.get('fact') or '').strip(), role)] += 1
                found: list[tuple[str, str]] = []
                for (fact, role), count in [] if bad else wanted.items():
                    at = 2 if role == 'source' else 3
                    hits = [
                        r
                        for r in on_episode
                        if (r[1] or '').strip() == fact and r[at] == c['wrong_uuid']
                    ]
                    created = [r for r in hits if r[4]]
                    if hits and not created:
                        bad = f'fact joined an older one, not created by this episode: {fact[:120]}'
                        break
                    if len(created) != count:
                        bad = f'fact given {count} time(s), {len(created)} found on the wrong node: {fact[:120]}'
                        break
                    found += [(r[0], role) for r in created]
                if bad:
                    problem(c, bad)
                    c['skipped'] = True
                else:
                    fact_ends[c['index']] = found

            # Two merges into one node in one episode: the old run filed that node once per merge,
            # so its facts went into the summary more than once and how often is not known. Such
            # merges are left whole when facts are involved.
            by_wrong = collections.defaultdict(list)
            for c in group_cases:
                if not c.get('skipped'):
                    by_wrong[c['wrong_uuid']].append(c)
            # The trace counts every merge the model's answer sent into the node (kept and unknown ones
            # too), and an episode that named the node itself filed it once more.
            for crowd in by_wrong.values():
                filed_again = len(crowd) > 1 or any(
                    c.get('merges_into_wrong', 1) > 1 or c.get('keep_mention') for c in crowd
                )
                if filed_again and any(fact_ends.get(c['index']) for c in crowd):
                    for c in crowd:
                        problem(
                            c,
                            'more than one merge into one node in one episode (the episode naming it '
                            'counts), with facts: how many summary lines they left there is not known',
                        )
                        c['skipped'] = True
            active = [c for c in group_cases if not c.get('skipped')]
            extracted = [
                EntityNode(
                    name=c['entity'],
                    group_id=episode.group_id,
                    labels=list(c.get('entity_labels') or ['Entity']),
                    summary='',
                    created_at=episode.created_at,
                )
                for c in active
            ]
            resolved = []
            caught: list[str] = []
            if extracted:
                node_log = logging.getLogger('graphiti_core.utils.maintenance.node_operations')
                handler = _Collect(caught)
                node_log.addHandler(handler)
                try:
                    resolved, _, _ = await resolve_extracted_nodes(
                        clients,
                        extracted,
                        episode,
                        previous,
                        services['entity_types'],
                    )
                finally:
                    node_log.removeHandler(handler)

            for c in group_cases:
                if c.get('skipped'):
                    plan['cases'].append(
                        {
                            'index': c['index'],
                            'graph': graph,
                            'episode': episode_uuid,
                            'entity': c['entity'],
                            'wrong': c['wrong'],
                            'wrong_uuid': c['wrong_uuid'],
                            'decision': 'skipped',
                        }
                    )
            targets: dict[int, Any] = {}  # case index -> target node, for cases that change
            for c, e, t in zip(active, extracted, resolved, strict=True):
                record = {
                    'index': c['index'],
                    'graph': graph,
                    'episode': episode_uuid,
                    'entity': c['entity'],
                    'wrong': c['wrong'],
                    'wrong_uuid': c['wrong_uuid'],
                }
                warned = [w for w in caught if repr(c['entity']) in w]
                if warned:
                    # A failed check or a failed second question also ends as "no duplicate".
                    record['dedupe_warning'] = ' | '.join(warned)
                if t.uuid == c['wrong_uuid']:
                    record.update(decision='same_as_before', target_uuid=t.uuid, target_name=t.name)
                    plan['cases'].append(record)
                    continue
                planned = planned_new.get((graph, t.uuid))
                if planned is not None:
                    # The dedupe found a node this plan creates in an earlier episode.
                    t = planned
                    record.update(decision='new_planned_earlier')
                elif t.uuid == e.uuid:
                    planned_new[(graph, e.uuid)] = e
                    record.update(decision='new')
                else:
                    record.update(decision='existing')
                record.update(target_uuid=t.uuid, target_name=t.name)
                plan['cases'].append(record)
                targets[c['index']] = t
            changing = [c for c in active if c['index'] in targets]

            # Facts: each end the entity held goes to its target; the other end stays where it is.
            for c in changing:
                t = targets[c['index']]
                for edge_uuid, role in fact_ends[c['index']]:
                    a, b = next((r[2], r[3]) for r in on_episode if r[0] == edge_uuid)
                    move = moves.setdefault(
                        edge_uuid,
                        {
                            'uuid': edge_uuid,
                            'graph': graph,
                            'fact': next(r[1] for r in on_episode if r[0] == edge_uuid),
                            'old_source': a,
                            'old_target': b,
                            'new_source': a,
                            'new_target': b,
                            'episodes': next(r[5] for r in on_episode if r[0] == edge_uuid),
                        },
                    )
                    # An end lies on exactly one wrong node, and two merges into one wrong node with
                    # facts are left whole above, so no two entities can claim the same end here.
                    side = 'new_source' if role == 'source' else 'new_target'
                    move[side] = t.uuid

            moved_here = [
                m
                for m in moves.values()
                if m['graph'] == graph and m.get('_episode') in (None, episode_uuid)
            ]
            for m in moved_here:
                m['_episode'] = episode_uuid
                # Not merged into it: shown, so the eye sees a fact the target now holds twice.
                m['duplicates_existing'] = [
                    u
                    for (u,) in rows(
                        g,
                        'MATCH (a:Entity {uuid: $s})-[r:RELATES_TO]->(b:Entity {uuid: $t}) '
                        'WHERE r.uuid <> $u AND r.fact = $f RETURN r.uuid',
                        s=m['new_source'],
                        t=m['new_target'],
                        u=m['uuid'],
                        f=m['fact'],
                    )
                ]
            # Lines a node loses: one per end of a moved fact it no longer holds.
            for m in moved_here:
                before, after = (
                    ends(m['old_source'], m['old_target']),
                    ends(m['new_source'], m['new_target']),
                )
                for node, count in before.items():
                    lost = count - after.get(node, 0)
                    if lost > 0:
                        entry = summaries.setdefault((graph, node), {'facts': [], 'node': node})
                        entry['facts'] += [m['fact']] * lost

            # Mentions: one per target the episode does not mention yet; the wrong node keeps as many
            # as the episode had for reasons other than the merges undone here, and at least one if
            # the episode named it itself, it is a target, or a merge into it stays.
            mentions = collections.defaultdict(list)
            for m, n in rows(
                g,
                'MATCH (e:Episodic {uuid: $e})-[m:MENTIONS]->(n:Entity) RETURN m.uuid, n.uuid',
                e=episode_uuid,
            ):
                mentions[n].append(m)
            for t in {t.uuid: t for t in targets.values()}.values():
                if not mentions.get(t.uuid):
                    mentions[t.uuid].append('planned')
                    plan['mentions_add'].append(
                        {
                            'uuid': str(uuidlib.uuid4()),
                            'graph': graph,
                            'episode': episode_uuid,
                            'entity': t.uuid,
                            'group_id': episode.group_id,
                            'created_at': episode.created_at.isoformat(),
                        }
                    )
            target_uuids = {t.uuid for t in targets.values()}
            for wrong in sorted({c['wrong_uuid'] for c in changing}):
                undone = sum(1 for c in changing if c['wrong_uuid'] == wrong)
                floor = (
                    1
                    if (
                        any(c.get('keep_mention') for c in changing if c['wrong_uuid'] == wrong)
                        or wrong in target_uuids
                        or any(c['wrong_uuid'] == wrong for c in group_cases if c not in changing)
                    )
                    else 0
                )
                # Mentions without a uuid come first, so the ones kept are those if any.
                have = sorted(mentions.get(wrong, []), key=lambda m: (m is not None, m or ''))
                keep = max(len(have) - undone, floor)
                for m in have[keep:]:
                    if m is None:
                        owner = next(c for c in changing if c['wrong_uuid'] == wrong)
                        problem(
                            owner, 'a mention of the wrong node to remove is without a uuid', False
                        )
                        continue
                    plan['mentions_drop'].append(
                        {'uuid': m, 'graph': graph, 'episode': episode_uuid, 'entity': wrong}
                    )
                # The wrong node existed before this episode, so some other episode mentions it; if
                # none does, removing this mention would leave it an orphan, and a person decides.
                dropping = sum(1 for d in plan['mentions_drop'] if d['entity'] == wrong)
                total = rows(
                    g,
                    'MATCH (:Episodic)-[m:MENTIONS]->(n:Entity {uuid: $n}) RETURN count(m)',
                    n=wrong,
                )[0][0]
                if dropping and total - dropping <= 0:
                    owner = next(c for c in changing if c['wrong_uuid'] == wrong)
                    problem(
                        owner,
                        'removing the last mention of the wrong node leaves it an orphan',
                        False,
                    )

            # Summaries of the targets, exactly as add_episode would have written them for this episode.
            unique_targets = list({t.uuid: t for t in targets.values()}.values())
            existing = [t for t in unique_targets if not any(t is n for n in planned_new.values())]
            for t in existing:
                earlier = summaries.get((graph, t.uuid), {})
                if 'target_after' in earlier:
                    # This node already gains facts from an earlier episode.
                    t.summary = earlier['target_after']
            before_summary = {t.uuid: t.summary for t in unique_targets}
            if unique_targets:
                # A target is handed one edge per end it gains, never one it already held: it got that
                # line when the fact was made. A loop made by the move gains its node one end, not two.
                moved_edges = []
                target_ids = {t.uuid for t in unique_targets}
                gained_here: set[str] = set()
                for m in moved_here:
                    gained = ends(m['new_source'], m['new_target']) - ends(
                        m['old_source'], m['old_target']
                    )
                    if not target_ids & set(gained):
                        continue
                    edge = await EntityEdge.get_by_uuid(driver, m['uuid'])
                    for node, count in gained.items():
                        if node in target_ids:
                            gained_here.add(node)
                            moved_edges += [
                                edge.model_copy(
                                    update={
                                        'uuid': str(uuidlib.uuid4()),
                                        'source_node_uuid': node,
                                        'target_node_uuid': NO_NODE,
                                    }
                                )
                                for _ in range(count)
                            ]
                # One pass for the episode, as add_episode makes it: Graphiti appends short facts
                # itself and asks the model only for a node with no summary yet or one grown too long.
                await extract_attributes_from_nodes(
                    clients,
                    unique_targets,
                    episode,
                    previous,
                    services['entity_types'],
                    edges=moved_edges or None,
                )
            for t in existing:
                if t.summary != before_summary[t.uuid]:
                    entry = summaries.setdefault((graph, t.uuid), {'facts': [], 'node': t.uuid})
                    entry.setdefault('target_before', before_summary[t.uuid])
                    entry['target_after'] = t.summary
                    entry['name'] = t.name
                    entry['gained'] = entry.get('gained') or t.uuid in gained_here

            for c in changing:
                entry = summaries.get((graph, c['wrong_uuid']))
                if entry is not None and entry['facts']:
                    # Only a node that loses lines; one that is merely another case's target is not.
                    entry.setdefault('wrong_before', c['wrong_summary'])
                    entry['name'] = c['wrong']

            # This episode's new nodes, as they stand now, go into the scratch copy for the next one.
            made_here = [
                t
                for t in {t.uuid: t for t in targets.values()}.values()
                if (graph, t.uuid) in planned_new
            ]
            for n in made_here:
                if n.name_embedding is None:
                    await n.generate_name_embedding(services['embedder'])
            if made_here:
                await add_nodes_and_edges_bulk(driver, [], [], made_here, [], services['embedder'])
    finally:
        await next(iter(drivers.values())).close()
        left = set(db.list_graphs())
        for copy_name in scratch.values():
            if copy_name in left:
                db.select_graph(copy_name).delete()

    for m in moves.values():
        m.pop('_episode', None)
    plan['edge_moves'] = [
        m
        for m in moves.values()
        if (m['old_source'], m['old_target']) != (m['new_source'], m['new_target'])
    ]

    for (graph, node), entry in summaries.items():
        if 'wrong_before' in entry and 'target_after' in entry:
            plan['problems'].append(
                {
                    'node': node,
                    'why': 'a node both loses and gains facts; its summary is left alone',
                }
            )
            continue
        if 'wrong_before' in entry:
            after, missing = remove_fact_lines(entry['wrong_before'], entry['facts'])
            for fact in missing:
                plan['problems'].append(
                    {'node': node, 'why': f'summary has no line for a moved fact: {fact[:120]}'}
                )
            if after != entry['wrong_before']:
                plan['summary_edits'].append(
                    {
                        'graph': graph,
                        'node_uuid': node,
                        'node_name': entry.get('name'),
                        'before': entry['wrong_before'],
                        'after': after,
                        'why': 'lines of facts that moved away',
                    }
                )
        if 'target_after' in entry:
            plan['summary_edits'].append(
                {
                    'graph': graph,
                    'node_uuid': node,
                    'node_name': entry.get('name'),
                    'before': entry['target_before'],
                    'after': entry['target_after'],
                    'why': (
                        'summary as add_episode would write it'
                        if not entry.get('gained')
                        # Graphiti appends a short summary's new facts; a long one the model rewrites.
                        else 'facts that moved here, appended'
                        if entry['target_after'].startswith(entry['target_before'].strip())
                        else 'rewritten by the model with facts that moved here'
                    ),
                }
            )

    # Nodes already named like a new one: shown to the eye, and not taken by apply for ones that
    # appeared since. Graphiti's exact match leaves such a name to the model when it is not unique.
    names_by_graph: dict[str, dict[str, list[str]]] = {}
    for graph in {g for g, _ in planned_new}:
        names = collections.defaultdict(list)
        for u, n in all_rows(
            db.select_graph(graph), 'MATCH (n:Entity)', 'n.uuid, n.name', 'n.uuid'
        ):
            names[norm(n)].append(u)
        names_by_graph[graph] = names
    for (graph, _), node in planned_new.items():
        if node.name_embedding is None:
            await node.generate_name_embedding(services['embedder'])
        plan['new_nodes'].append(
            {
                'uuid': node.uuid,
                'graph': graph,
                'name': node.name,
                'group_id': node.group_id,
                'labels': node.labels,
                'created_at': node.created_at.isoformat(),
                'summary': node.summary,
                'attributes': dict(node.attributes or {}),
                'name_embedding': list(node.name_embedding),
                'namesakes': sorted(names_by_graph[graph].get(norm(node.name), [])),
            }
        )
    return plan


def report(plan: dict) -> str:
    out = []
    names = {n['uuid']: n['name'] for n in plan['new_nodes']}
    for c in plan['cases']:
        out.append(
            f'{c["graph"]} {c["entity"]!r} <- was in {c["wrong"]!r}: {c["decision"]} -> {c.get("target_name")!r}'
        )
        if c.get('dedupe_warning'):
            out.append(f'    dedupe warned: {c["dedupe_warning"][:300]}')
    if plan.get('missing_graphs'):
        out.append(f'graphs that do not exist, left out: {plan["missing_graphs"]}')
    out.append('')
    out.append(
        f'new nodes {len(plan["new_nodes"])}, mentions added {len(plan["mentions_add"])}, '
        f'mentions removed {len(plan["mentions_drop"])}, facts moved {len(plan["edge_moves"])}, '
        f'summaries edited {len(plan["summary_edits"])}, problems {len(plan["problems"])}'
    )
    for m in plan['edge_moves']:
        later = m.get('episodes', 1) - 1
        out.append(
            f'  fact {m["fact"][:120]!r}: {m["old_source"][:8]}->{m["old_target"][:8]} becomes '
            f'{m["new_source"][:8]}->{m["new_target"][:8]}'
            + (
                f' (joined by {later} later episode{"s" if later > 1 else ""}, moved with it)'
                if later
                else ''
            )
            + (
                f' (same fact already there: {m["duplicates_existing"]})'
                if m.get('duplicates_existing')
                else ''
            )
        )
    for e in plan['summary_edits']:
        out.append(
            f'  summary of {e.get("node_name") or names.get(e["node_uuid"], e["node_uuid"])} ({e["why"]}):'
        )
        out.append('    before: ' + e['before'].replace('\n', ' / '))
        out.append('    after:  ' + e['after'].replace('\n', ' / '))
    for n in plan['new_nodes']:
        out.append(f'  new node {n["name"]!r}: {n["summary"][:200]!r}')
        if n.get('namesakes'):
            out.append(f'    namesakes already in the graph: {n["namesakes"]}')
    for p in plan['problems']:
        out.append(f'  PROBLEM {p}')
    return '\n'.join(out)


# ---------------------------------------------------------------- apply


def preflight(g, ops: dict) -> list[str]:
    """What in the graph no longer matches the plan. Already-done steps are not problems."""
    bad = []
    # The graph may have moved on between plan and apply: a node the plan would create may exist
    # under another uuid by now, and a node the plan sends something to may be gone.
    names = collections.defaultdict(set)
    for u, n in all_rows(g, 'MATCH (n:Entity)', 'n.uuid, n.name', 'n.uuid'):
        names[norm(n)].add(u)
    for n in ops['new_nodes']:
        got = rows(g, 'MATCH (n:Entity {uuid: $u}) RETURN n.name', u=n['uuid'])
        if got and got[0][0] != n['name']:
            bad.append(f'node {n["uuid"]} exists under another name')
        others = names.get(norm(n['name']), set()) - {n['uuid']} - set(n.get('namesakes') or [])
        if others:
            bad.append(f'a node named {n["name"]!r} appeared since the plan: {sorted(others)}')
    planned = {n['uuid'] for n in ops['new_nodes']}
    needed = {m['entity'] for m in ops['mentions_add']} | {
        e['node_uuid'] for e in ops['summary_edits']
    }
    needed |= {u for mv in ops['edge_moves'] for u in (mv['new_source'], mv['new_target'])}
    present = {u for us in names.values() for u in us}
    for u in sorted(needed - planned - present):
        bad.append(f'node {u} the plan sends to is gone')
    for m in ops['mentions_add']:
        if not rows(g, 'MATCH (e:Episodic {uuid: $e}) RETURN e.uuid', e=m['episode']):
            bad.append(f'episode {m["episode"]} is missing')
    for m in ops['mentions_drop']:
        got = rows(
            g,
            'MATCH (e:Episodic)-[m:MENTIONS {uuid: $u}]->(n:Entity) RETURN e.uuid, n.uuid',
            u=m['uuid'],
        )
        if got and got[0] != [m['episode'], m['entity']]:
            bad.append(f'mention {m["uuid"]} joins other nodes')
    for mv in ops['edge_moves']:
        got = rows(
            g, 'MATCH (a)-[r:RELATES_TO {uuid: $u}]->(b) RETURN a.uuid, b.uuid', u=mv['uuid']
        )
        if [list(x) for x in got] not in (
            [[mv['old_source'], mv['old_target']]],
            [[mv['new_source'], mv['new_target']]],
        ):
            bad.append(f'fact {mv["uuid"]} is not where the plan found it: {got}')
    for e in ops['summary_edits']:
        got = rows(g, 'MATCH (n:Entity {uuid: $u}) RETURN n.summary', u=e['node_uuid'])
        if not got or (got[0][0] or '') not in (e['before'], e['after']):
            bad.append(f'summary of {e["node_uuid"]} changed since the plan')
    return bad


async def write(conn: dict, graph_name: str, ops: dict) -> collections.Counter:
    """Carry out the plan on one graph. Every step checks whether it is already done."""
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from graphiti_core.nodes import EntityNode
    from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk

    done = collections.Counter()
    g = falkor(conn).select_graph(graph_name)

    fresh = []
    for n in ops['new_nodes']:
        if rows(g, 'MATCH (n:Entity {uuid: $u}) RETURN n.uuid', u=n['uuid']):
            continue
        fresh.append(
            EntityNode(
                uuid=n['uuid'],
                name=n['name'],
                group_id=n['group_id'],
                labels=n['labels'],
                summary=n['summary'],
                created_at=datetime.fromisoformat(n['created_at']),
                name_embedding=n['name_embedding'],
                attributes=n.get('attributes') or {},
            )
        )
    if fresh:
        from falkordb.asyncio import FalkorDB as AsyncFalkorDB

        client = AsyncFalkorDB(
            host=conn['host'],
            port=conn['port'],
            username=conn['username'],
            password=conn['password'],
        )
        # The graph has its indices; a new driver would schedule building them again.
        FalkorDriver._indexed_graphs.add((id(client), graph_name))
        driver = FalkorDriver(falkor_db=client, database=graph_name)
        try:
            # Graphiti's own bulk save, the one add_episode uses, so the node has exactly its shape.
            await add_nodes_and_edges_bulk(driver, [], [], fresh, [], None)
        finally:
            await driver.close()
        done['nodes created'] += len(fresh)

    for m in ops['mentions_add']:
        if rows(g, 'MATCH ()-[m:MENTIONS {uuid: $u}]->() RETURN m.uuid', u=m['uuid']):
            continue
        g.query(
            'MATCH (e:Episodic {uuid: $e}), (n:Entity {uuid: $n}) '
            'CREATE (e)-[:MENTIONS {uuid: $u, group_id: $g, created_at: $t}]->(n)',
            {
                'e': m['episode'],
                'n': m['entity'],
                'u': m['uuid'],
                'g': m['group_id'],
                't': m['created_at'],
            },
        )
        done['mentions added'] += 1

    for mv in ops['edge_moves']:
        at = rows(g, 'MATCH (a)-[r:RELATES_TO {uuid: $u}]->(b) RETURN a.uuid, b.uuid', u=mv['uuid'])
        if [list(x) for x in at] == [[mv['new_source'], mv['new_target']]]:
            continue
        # The same edge on new ends: every property copied as it is, then the two end uuids it records.
        # Only the end that moves has its recorded uuid rewritten; the other keeps what it records.
        moved_ends = []
        if mv['new_source'] != mv['old_source']:
            moved_ends.append('r2.source_node_uuid = $ns')
        if mv['new_target'] != mv['old_target']:
            moved_ends.append('r2.target_node_uuid = $nt')
        g.query(
            'MATCH (a:Entity {uuid: $os})-[r:RELATES_TO {uuid: $u}]->(b:Entity {uuid: $ot}) '
            'MATCH (na:Entity {uuid: $ns}), (nb:Entity {uuid: $nt}) '
            'CREATE (na)-[r2:RELATES_TO]->(nb) SET r2 = properties(r) '
            f'SET {", ".join(moved_ends)} DELETE r',
            {
                'u': mv['uuid'],
                'os': mv['old_source'],
                'ot': mv['old_target'],
                'ns': mv['new_source'],
                'nt': mv['new_target'],
            },
        )
        done['facts moved'] += 1

    for m in ops['mentions_drop']:
        if not rows(g, 'MATCH ()-[m:MENTIONS {uuid: $u}]->() RETURN m.uuid', u=m['uuid']):
            continue
        g.query(
            'MATCH (e:Episodic {uuid: $e})-[m:MENTIONS {uuid: $u}]->(n:Entity {uuid: $n}) DELETE m',
            {'e': m['episode'], 'u': m['uuid'], 'n': m['entity']},
        )
        done['mentions removed'] += 1

    for e in ops['summary_edits']:
        res = g.query(
            'MATCH (n:Entity {uuid: $u}) WHERE n.summary = $b SET n.summary = $a RETURN n.uuid',
            {'u': e['node_uuid'], 'b': e['before'], 'a': e['after']},
        ).result_set
        if res:
            done['summaries edited'] += len(res)
    return done


def ops_for(plan: dict, source: str) -> dict:
    keys = ('new_nodes', 'mentions_add', 'mentions_drop', 'edge_moves', 'summary_edits')
    return {k: [x for x in plan[k] if x['graph'] == source] for k in keys}


async def apply(
    plan: dict, source: str, target: str, backup: str | None, server_stopped: bool = False
) -> int:
    blocking = [p for p in plan['problems'] if not p.get('skipped')]
    if blocking:
        # A skipped case is left whole and harms nothing; anything else needs a person first.
        print(
            'the plan has problems that are not whole skipped cases; nothing was written:',
            file=sys.stderr,
        )
        for p in blocking:
            print(f'  {p}', file=sys.stderr)
        return 2
    _, conn = config_and_connection()
    db = falkor(conn)
    graphs = set(db.list_graphs())
    if target not in graphs:
        print(f'graph {target!r} does not exist', file=sys.stderr)
        return 2
    g = db.select_graph(target)
    ops = ops_for(plan, source)
    if target == source:
        # The real graph: only with the server stopped -- a write of its own in between would undo
        # a summary edit or duplicate a moved fact -- and with a copy taken just now that still
        # matches it in every node, fact and mention.
        if not server_stopped:
            print(
                'writing to the graph the plan was made from needs --server-stopped: '
                'stop graphiti-server first, so nothing else writes meanwhile',
                file=sys.stderr,
            )
            return 2
        if not backup or backup not in graphs or backup in (source, target):
            print(
                'writing to the graph the plan was made from needs --backup naming an existing copy '
                'that is another graph',
                file=sys.stderr,
            )
            return 2
    before = snapshot(g, source)
    # The copy and the graph -- untouched, or part way through a run that stopped -- must lead to
    # the same graph under the plan; a copy that is stale or of something else does not.
    if target == source and expected_after(
        snapshot(db.select_graph(backup), source), ops, source
    ) != expected_after(before, ops, source):
        print(f'backup {backup!r} does not match {target!r}: take the copy again', file=sys.stderr)
        return 2
    bad = preflight(g, ops)
    if bad:
        print('the graph does not match the plan; nothing was written:', file=sys.stderr)
        for b in bad:
            print('  ' + b, file=sys.stderr)
        return 3
    done = await write(conn, target, ops)
    after = snapshot(g, source)
    again = await write(conn, target, ops)
    print('written:', dict(done))
    print(
        'a second pass found nothing left to do'
        if not sum(again.values())
        else f'SECOND PASS WROTE {dict(again)}'
    )
    problems = compare(before, after, ops, source)
    if ops['new_nodes']:
        # The vectors written must be as long as the ones already there.
        reference = next(
            iter(
                u for u in before['entities'] if before['entities'][u]['embedding'] == 'Vectorf32'
            ),
            None,
        )
        sizes = embedding_sizes(
            g, [n['uuid'] for n in ops['new_nodes']] + ([reference] if reference else [])
        )
        want = sizes.get(reference)
        for n in ops['new_nodes']:
            if (
                after['entities'].get(n['uuid'], {}).get('embedding') != 'Vectorf32'
                or sizes.get(n['uuid']) != want
            ):
                problems.append(
                    f"new node {n['name']!r}: vector {sizes.get(n['uuid'])} long, the graph's are {want}"
                )
    print('checks before:', before['checks'])
    print('checks after: ', after['checks'])
    if problems or sum(again.values()):
        print('DIFFERENCES THE PLAN DOES NOT ACCOUNT FOR:')
        for p in problems:
            print('  ' + p)
        return 4
    print('every difference is the planned one')
    return 0


# ---------------------------------------------------------------- command line


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan', help='read the graph and ask the new dedupe; write a plan file')
    p.add_argument('--cases', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--graphs', nargs='+', default=['main', 'igor', 'red'])
    a = sub.add_parser('apply', help='carry out a plan on one graph')
    a.add_argument('--plan', required=True)
    a.add_argument('--source', required=True, help='the graph the plan was made from, e.g. main')
    a.add_argument(
        '--graph',
        required=True,
        help='the graph to write: a copy, or the source itself with --backup',
    )
    a.add_argument('--backup', help='an existing copy of the source, taken just before')
    a.add_argument(
        '--server-stopped',
        action='store_true',
        help='say graphiti-server is stopped; required to write the graph the plan was made from',
    )
    c = sub.add_parser('check', help='integrity counters of a graph, read only')
    c.add_argument('--graph', required=True)
    c.add_argument('--group', help='the group id its nodes carry (default: the graph name)')
    args = parser.parse_args(argv[1:])

    if args.command == 'plan':
        with open(args.cases, encoding='utf-8') as f:
            cases = json.load(f)
        plan = asyncio.run(make_plan(cases, args.graphs))
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(plan, f, ensure_ascii=False, indent=1)
        print(report(plan))
        return 0
    if args.command == 'apply':
        with open(args.plan, encoding='utf-8') as f:
            plan = json.load(f)
        if plan.get('version') != PLAN_VERSION:
            print('unknown plan version', file=sys.stderr)
            return 2
        return asyncio.run(apply(plan, args.source, args.graph, args.backup, args.server_stopped))
    _, conn = config_and_connection()
    snap = snapshot(falkor(conn).select_graph(args.graph), args.group or args.graph)
    print(
        json.dumps(
            {'labels': snap['labels'], 'types': snap['types'], 'checks': snap['checks']},
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
