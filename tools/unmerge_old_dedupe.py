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
node of its own. Extraction is not repeated -- the facts it produced were right,
only the node they hung on was wrong.

    plan    read the graph, ask the new dedupe, write a plan file; writes nothing to the graph
    apply   carry out a plan file on one graph: a copy first, the real graph with --backup
    check   integrity counters of a graph, read only

What one undone merge changes, and nothing else:
  - the entity's node is created (or found) and the episode mentions it;
  - the episode's mention of the wrong node is removed, unless the episode named that node itself;
  - every fact extraction gave to the entity moves from the wrong node to the right one
    (the same edge, uuid and every property kept; only the endpoint changes);
  - the lines those facts appended to the wrong node's summary are removed.
Episodes, sagas, NEXT_EPISODE, HAS_EPISODE, fact texts, embeddings and provenance are never touched.

The cases file is made from the Graphiti trace: one entry per merge to undo, with the
entity, the node it was wrongly merged into, the episode text, the facts extraction
gave the entity, and whether the episode named the wrong node by itself.

Run it inside the Graphiti image, with the configuration the server uses:

    docker compose run --rm --no-deps -T -v /tmp/dedupe-old:/work --entrypoint python \\
        graphiti-server /work/unmerge_old_dedupe.py plan --cases /work/cases.json --out /work/plan.json
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import sys
import uuid as uuidlib
from datetime import datetime, timezone
from typing import Any

PLAN_VERSION = 1
SOURCE_GRAPH_CHECKS = ('Episodic', 'Saga', 'NEXT_EPISODE', 'HAS_EPISODE')


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


# ---------------------------------------------------------------- integrity snapshot


def snapshot(graph, group_id: str) -> dict[str, Any]:
    """Everything apply compares before and after. Read only."""
    snap: dict[str, Any] = {}
    snap['labels'] = {str(lab): c for lab, c in rows(graph, 'MATCH (n) RETURN labels(n), count(n)')}
    snap['types'] = {t: c for t, c in rows(graph, 'MATCH ()-[r]->() RETURN type(r), count(r)')}
    # FalkorDB gives no size for a stored vector; its type is what says it is one.
    snap['entities'] = {
        u: {'name': n, 'summary': s, 'group_id': g, 'labels': lab, 'embedding': t}
        for u, n, s, g, lab, t in rows(
            graph,
            'MATCH (n:Entity) RETURN n.uuid, n.name, n.summary, n.group_id, n.labels, typeof(n.name_embedding)',
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
        for u, a, b, sp, tp, f, e, g, nm, va, ia, xa, he in rows(
            graph,
            'MATCH (a)-[r:RELATES_TO]->(b) RETURN r.uuid, a.uuid, b.uuid, r.source_node_uuid, '
            'r.target_node_uuid, r.fact, r.episodes, r.group_id, r.name, r.valid_at, r.invalid_at, '
            'r.expired_at, typeof(r.fact_embedding)',
        )
    }
    snap['mentions'] = {
        u: {'episode': e, 'entity': n, 'from_episodic': fe, 'to_entity': te}
        for u, e, n, fe, te in rows(
            graph,
            "MATCH (a)-[m:MENTIONS]->(b) RETURN m.uuid, a.uuid, b.uuid, 'Episodic' IN labels(a), 'Entity' IN labels(b)",
        )
    }
    episodes = {u for (u,) in rows(graph, 'MATCH (e:Episodic) RETURN e.uuid')}
    mentioned = collections.Counter(m['entity'] for m in snap['mentions'].values())
    snap['checks'] = {
        'edge_endpoint_props_disagree': sum(
            1
            for e in snap['edges'].values()
            if e['source_prop'] != e['source'] or e['target_prop'] != e['target']
        ),
        'edge_episode_missing': sum(
            1 for e in snap['edges'].values() for x in (e['episodes'] or []) if x not in episodes
        ),
        'edge_without_episodes': sum(1 for e in snap['edges'].values() if not e['episodes']),
        'edge_embedding_types': sorted({str(e['has_embedding']) for e in snap['edges'].values()}),
        'edge_self_loop': sum(1 for e in snap['edges'].values() if e['source'] == e['target']),
        'edge_group_other': sum(1 for e in snap['edges'].values() if e['group_id'] != group_id),
        'entity_without_mention': sum(1 for u in snap['entities'] if not mentioned[u]),
        'entity_embedding_types': sorted({str(e['embedding']) for e in snap['entities'].values()}),
        'entity_group_other': sum(
            1 for e in snap['entities'].values() if e['group_id'] != group_id
        ),
        'mention_not_episode_to_entity': sum(
            1 for m in snap['mentions'].values() if not (m['from_episodic'] and m['to_entity'])
        ),
        'mention_duplicate_pair': sum(
            c - 1
            for c in collections.Counter(
                (m['episode'], m['entity']) for m in snap['mentions'].values()
            ).values()
            if c > 1
        ),
    }
    return snap


def embedding_sizes(graph, uuids: list[str]) -> dict[str, int]:
    """Length of each node's name vector, read back whole; only for the few nodes asked."""
    return {
        u: len(v or [])
        for u, v in rows(
            graph, 'MATCH (n:Entity) WHERE n.uuid IN $u RETURN n.uuid, n.name_embedding', u=uuids
        )
    }


def compare(before: dict, after: dict, plan_ops: dict) -> list[str]:
    """Every difference between two snapshots that the plan does not account for."""
    bad: list[str] = []
    new_nodes = {n['uuid'] for n in plan_ops['new_nodes']}
    edits = {e['node_uuid']: e['after'] for e in plan_ops['summary_edits']}
    moves = {m['uuid']: m for m in plan_ops['edge_moves']}
    added = {m['uuid'] for m in plan_ops['mentions_add']}
    dropped = {m['uuid'] for m in plan_ops['mentions_drop']}

    for key in SOURCE_GRAPH_CHECKS:
        b = before['labels'].get(f"['{key}']", before['types'].get(key, 0))
        a = after['labels'].get(f"['{key}']", after['types'].get(key, 0))
        if a != b:
            bad.append(f'{key}: {b} before, {a} after')

    if set(after['entities']) != set(before['entities']) | new_nodes:
        extra = set(after['entities']) - set(before['entities']) - new_nodes
        lost = set(before['entities']) - set(after['entities'])
        bad.append(f'entities: unexpected {sorted(extra)[:5]}, lost {sorted(lost)[:5]}')
    for u, b in before['entities'].items():
        a = after['entities'].get(u)
        if a is None:
            continue
        want = dict(b, summary=edits.get(u, b['summary']))
        if a != want:
            bad.append(f'entity {u} ({b["name"]}) changed beyond the plan')
    for u in new_nodes:
        if u not in after['entities']:
            bad.append(f'new entity {u} missing')

    if set(after['edges']) != set(before['edges']):
        bad.append('the set of facts changed')
    for u, b in before['edges'].items():
        a = after['edges'].get(u)
        if a is None:
            continue
        want = dict(b)
        if u in moves:
            m = moves[u]
            want.update(
                source=m['new_source'],
                target=m['new_target'],
                source_prop=m['new_source'],
                target_prop=m['new_target'],
            )
        if a != want:
            bad.append(f'fact {u} changed beyond the plan')

    want_mentions = (set(before['mentions']) - dropped) | added
    if set(after['mentions']) != want_mentions:
        bad.append(
            f'mentions: unexpected {sorted(set(after["mentions"]) - want_mentions)[:5]}, '
            f'missing {sorted(want_mentions - set(after["mentions"]))[:5]}'
        )
    for u, b in before['mentions'].items():
        if u in after['mentions'] and after['mentions'][u] != b:
            bad.append(f'mention {u} changed')

    for key, b in before['checks'].items():
        a = after['checks'][key]
        if key == 'entity_without_mention':
            if a > b:
                bad.append(f'{key}: {b} before, {a} after')
        elif a != b:
            bad.append(f'{key}: {b} before, {a} after')
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
        'MATCH (e:Episodic {uuid: $e})-[:MENTIONS]->(n:Entity) WHERE n.name = $w RETURN n.uuid, n.summary',
        e=episode,
        w=case['wrong'],
    )
    if len(wrong) != 1:
        return f'the episode mentions {len(wrong)} nodes named {case["wrong"]!r}'
    return graph, episode, wrong[0][0], wrong[0][1] or ''


async def make_plan(cases: list[dict], graphs: list[str], services: Any = None) -> dict:
    """Read the graph and ask the current dedupe; return the plan. Writes nothing to the graph."""
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from graphiti_core.edges import EntityEdge
    from graphiti_core.graphiti_types import GraphitiClients
    from graphiti_core.nodes import EntityNode, EpisodicNode
    from graphiti_core.search.search_utils import RELEVANT_SCHEMA_LIMIT
    from graphiti_core.tracer import create_tracer
    from graphiti_core.utils.maintenance.node_operations import (
        extract_attributes_from_nodes,
        resolve_extracted_nodes,
    )

    config, conn = config_and_connection()
    db = falkor(conn)
    if services is None:
        from services.factories import CrossEncoderFactory, EmbedderFactory, LLMClientFactory
        from utils.type_config import build_entity_types

        services = {
            'llm': LLMClientFactory.create(config.llm),
            'embedder': EmbedderFactory.create(config.embedder),
            'cross_encoder': CrossEncoderFactory.create(config.llm, config.embedder),
            'entity_types': build_entity_types(config.graphiti.entity_types),
        }

    plan: dict[str, Any] = {
        'version': PLAN_VERSION,
        'made_at': now_iso(),
        'graphs': graphs,
        'cases': [],
        'new_nodes': [],
        'mentions_add': [],
        'mentions_drop': [],
        'edge_moves': [],
        'summary_edits': [],
        'problems': [],
    }

    # Locate everything first; a case that cannot be located is reported, never guessed at.
    located: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for i, case in enumerate(cases):
        where = locate(db, graphs, case)
        if isinstance(where, str):
            plan['problems'].append(
                {'case': i, 'entity': case['entity'], 'wrong': case['wrong'], 'why': where}
            )
            continue
        graph, episode, wrong_uuid, wrong_summary = where
        located[(graph, episode)].append(
            dict(
                case,
                index=i,
                graph=graph,
                episode=episode,
                wrong_uuid=wrong_uuid,
                wrong_summary=wrong_summary,
            )
        )

    def episode_time(key):
        graph, episode = key
        return (
            rows(
                db.select_graph(graph), 'MATCH (e:Episodic {uuid: $e}) RETURN e.valid_at', e=episode
            )[0][0]
            or ''
        )

    planned_new: dict[
        tuple[str, str], Any
    ] = {}  # (graph, normalized name) -> EntityNode planned new
    summaries: dict[
        tuple[str, str], dict
    ] = {}  # (graph, node uuid) -> {'name','before','after','why'}
    moves: dict[str, dict] = {}  # edge uuid -> move

    for key in sorted(located, key=episode_time):
        graph, episode_uuid = key
        group_cases = located[key]
        driver = FalkorDriver(
            host=conn['host'],
            port=conn['port'],
            username=conn['username'],
            password=conn['password'],
            database=graph,
        )
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
        previous = await EpisodicNode.get_by_uuids(driver, previous_uuids) if previous_uuids else []
        previous.sort(key=lambda e: e.valid_at)

        extracted = [
            EntityNode(
                name=c['entity'],
                group_id=episode.group_id,
                labels=list(c.get('entity_labels') or ['Entity']),
                summary='',
                created_at=episode.created_at,
            )
            for c in group_cases
        ]
        resolved, _, _ = await resolve_extracted_nodes(
            clients, extracted, episode, previous, services['entity_types']
        )

        targets = []
        for c, e, t in zip(group_cases, extracted, resolved, strict=True):
            record = {
                'index': c['index'],
                'graph': graph,
                'episode': episode_uuid,
                'entity': c['entity'],
                'wrong': c['wrong'],
                'wrong_uuid': c['wrong_uuid'],
            }
            if t.uuid == c['wrong_uuid']:
                record.update(decision='same_as_before', target_uuid=t.uuid, target_name=t.name)
                plan['cases'].append(record)
                targets.append(None)
                continue
            if t.uuid == e.uuid:
                twin = planned_new.get((graph, norm(e.name)))
                if twin is not None:
                    t = twin
                    record.update(decision='new_planned_earlier')
                else:
                    planned_new[(graph, norm(e.name))] = e
                    record.update(decision='new')
            else:
                record.update(decision='existing')
            record.update(target_uuid=t.uuid, target_name=t.name)
            plan['cases'].append(record)
            targets.append(t)

        # Mentions: the episode mentions every target once; it keeps the wrong node only if it named it.
        mentioned = {
            u
            for (u,) in rows(
                g,
                'MATCH (e:Episodic {uuid: $e})-[:MENTIONS]->(n:Entity) RETURN n.uuid',
                e=episode_uuid,
            )
        }
        keep_wrong = {
            c['wrong_uuid']
            for c, t in zip(group_cases, targets, strict=True)
            if t is None or c.get('keep_mention')
        }
        keep_wrong |= {t.uuid for t in targets if t is not None}
        for t in targets:
            if t is None:
                continue
            if t.uuid not in mentioned:
                mentioned.add(t.uuid)
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
        for wrong in {c['wrong_uuid'] for c in group_cases} - keep_wrong:
            for (m,) in rows(
                g,
                'MATCH (e:Episodic {uuid: $e})-[m:MENTIONS]->(n:Entity {uuid: $n}) RETURN m.uuid',
                e=episode_uuid,
                n=wrong,
            ):
                plan['mentions_drop'].append(
                    {'uuid': m, 'graph': graph, 'episode': episode_uuid, 'entity': wrong}
                )

        # Facts: each fact extraction gave the entity, found on the wrong node by its text, as often as it was given.
        replace = {
            c['wrong_uuid']: t.uuid
            for c, t in zip(group_cases, targets, strict=True)
            if t is not None
        }
        moved_by_target: dict[str, list] = collections.defaultdict(list)
        for c, t in zip(group_cases, targets, strict=True):
            if t is None:
                continue
            on_wrong = rows(
                g,
                'MATCH (a)-[r:RELATES_TO]->(b) WHERE (a.uuid = $w OR b.uuid = $w) AND $e IN r.episodes '
                'RETURN r.uuid, r.fact, a.uuid, b.uuid',
                w=c['wrong_uuid'],
                e=episode_uuid,
            )
            wanted = collections.Counter(
                (f.get('fact') or '').strip() for f in c.get('entity_facts') or []
            )
            for fact, count in wanted.items():
                hits = [r for r in on_wrong if (r[1] or '').strip() == fact]
                if len(hits) != count:
                    plan['problems'].append(
                        {
                            'case': c['index'],
                            'entity': c['entity'],
                            'wrong': c['wrong'],
                            'why': f'fact given {count} time(s), found {len(hits)} on the wrong node: {fact[:120]}',
                        }
                    )
                    continue
                for edge_uuid, edge_fact, a, b in hits:
                    if edge_uuid in moves:
                        continue  # both ends were wrong: already moved with both replaced
                    moves[edge_uuid] = {
                        'uuid': edge_uuid,
                        'graph': graph,
                        'fact': edge_fact,
                        'old_source': a,
                        'old_target': b,
                        'new_source': replace.get(a, a),
                        'new_target': replace.get(b, b),
                    }
                    for old in (a, b):
                        if old in replace:
                            entry = summaries.setdefault((graph, old), {'facts': [], 'node': old})
                            entry['facts'].append(edge_fact)

        for move in moves.values():
            if move['graph'] != graph or move.get('_targets_done'):
                continue
            move['_targets_done'] = True
            edge = await EntityEdge.get_by_uuid(driver, move['uuid'])
            edge.source_node_uuid, edge.target_node_uuid = move['new_source'], move['new_target']
            for t in targets:
                if t is not None and t.uuid in (move['new_source'], move['new_target']):
                    moved_by_target[t.uuid].append(edge)

        # Summaries of the targets, exactly as add_episode would have written them for this episode.
        unique_targets = list({t.uuid: t for t in targets if t is not None}.values())
        existing = [t for t in unique_targets if not any(t is n for n in planned_new.values())]
        for t in existing:
            earlier = summaries.get((graph, t.uuid), {})
            if 'target_after' in earlier:
                t.summary = earlier[
                    'target_after'
                ]  # this node already gains facts from an earlier episode
        before = {t.uuid: t.summary for t in unique_targets}
        if unique_targets:
            # One pass for the episode, as add_episode makes it: Graphiti appends short facts itself
            # and asks the model only for a node that has no summary yet or would grow too long.
            # An edge with both ends replaced is listed under both targets; Graphiti must see it once.
            moved_edges = list(
                {e.uuid: e for edges in moved_by_target.values() for e in edges}.values()
            )
            await extract_attributes_from_nodes(
                clients,
                unique_targets,
                episode,
                previous,
                services['entity_types'],
                edges=moved_edges or None,
            )
        for t in existing:
            if t.summary != before[t.uuid]:
                entry = summaries.setdefault((graph, t.uuid), {'facts': [], 'node': t.uuid})
                entry.setdefault('target_before', before[t.uuid])
                entry['target_after'] = t.summary
                entry['name'] = t.name

        for c in group_cases:
            c_entry = summaries.get((graph, c['wrong_uuid']))
            if c_entry is not None:
                c_entry.setdefault('wrong_before', c['wrong_summary'])
                c_entry['name'] = c['wrong']

    for move in moves.values():
        move.pop('_targets_done', None)
    plan['edge_moves'] = list(moves.values())

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
                    'why': 'facts that moved here',
                }
            )

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
                'name_embedding': list(node.name_embedding),
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
    out.append('')
    out.append(
        f'new nodes {len(plan["new_nodes"])}, mentions added {len(plan["mentions_add"])}, '
        f'mentions removed {len(plan["mentions_drop"])}, facts moved {len(plan["edge_moves"])}, '
        f'summaries edited {len(plan["summary_edits"])}, problems {len(plan["problems"])}'
    )
    for m in plan['edge_moves']:
        out.append(
            f'  fact {m["fact"][:120]!r}: {m["old_source"][:8]}->{m["old_target"][:8]} becomes '
            f'{m["new_source"][:8]}->{m["new_target"][:8]}'
        )
    for e in plan['summary_edits']:
        out.append(
            f'  summary of {e.get("node_name") or names.get(e["node_uuid"], e["node_uuid"])} ({e["why"]}):'
        )
        out.append('    before: ' + e['before'].replace('\n', ' / '))
        out.append('    after:  ' + e['after'].replace('\n', ' / '))
    for n in plan['new_nodes']:
        out.append(f'  new node {n["name"]!r}: {n["summary"][:200]!r}')
    for p in plan['problems']:
        out.append(f'  PROBLEM {p}')
    return '\n'.join(out)


# ---------------------------------------------------------------- apply


def preflight(g, ops: dict) -> list[str]:
    """What in the graph no longer matches the plan. Already-done steps are not problems."""
    bad = []
    for n in ops['new_nodes']:
        got = rows(g, 'MATCH (n:Entity {uuid: $u}) RETURN n.name', u=n['uuid'])
        if got and got[0][0] != n['name']:
            bad.append(f'node {n["uuid"]} exists under another name')
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
    driver = FalkorDriver(
        host=conn['host'],
        port=conn['port'],
        username=conn['username'],
        password=conn['password'],
        database=graph_name,
    )

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
            )
        )
    if fresh:
        # Graphiti's own bulk save, the one add_episode uses, so the node has exactly its shape.
        await add_nodes_and_edges_bulk(driver, [], [], fresh, [], None)
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
        g.query(
            'MATCH (a:Entity {uuid: $os})-[r:RELATES_TO {uuid: $u}]->(b:Entity {uuid: $ot}) '
            'MATCH (na:Entity {uuid: $ns}), (nb:Entity {uuid: $nt}) '
            'CREATE (na)-[r2:RELATES_TO]->(nb) SET r2 = properties(r) '
            'SET r2.source_node_uuid = $ns, r2.target_node_uuid = $nt DELETE r',
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


def counts(snap: dict) -> dict:
    return {'labels': snap['labels'], 'types': snap['types']}


async def apply(plan: dict, source: str, target: str, backup: str | None) -> int:
    _, conn = config_and_connection()
    db = falkor(conn)
    graphs = set(db.list_graphs())
    if target not in graphs:
        print(f'graph {target!r} does not exist', file=sys.stderr)
        return 2
    g = db.select_graph(target)
    if target == source:
        # The real graph: only with a copy taken just now, which still matches it.
        if not backup or backup not in graphs:
            print(
                'writing to the graph the plan was made from needs --backup naming an existing copy',
                file=sys.stderr,
            )
            return 2
        if counts(snapshot(db.select_graph(backup), source)) != counts(snapshot(g, source)):
            print(
                f'backup {backup!r} does not match {target!r}: take the copy again', file=sys.stderr
            )
            return 2
    ops = ops_for(plan, source)
    bad = preflight(g, ops)
    if bad:
        print('the graph does not match the plan; nothing was written:', file=sys.stderr)
        for b in bad:
            print('  ' + b, file=sys.stderr)
        return 3
    before = snapshot(g, source)
    done = await write(conn, target, ops)
    after = snapshot(g, source)
    again = await write(conn, target, ops)
    print('written:', dict(done))
    print(
        'a second pass found nothing left to do'
        if not sum(again.values())
        else f'SECOND PASS WROTE {dict(again)}'
    )
    problems = compare(before, after, ops)
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
        return asyncio.run(apply(plan, args.source, args.graph, args.backup))
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
