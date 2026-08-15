#!/usr/bin/env python3
"""Read-only validator for Graphiti Saga structure and semantic expectations.

One Graphiti group is expected to map to one physical Falkor graph. The tool
never mutates graph data and never prints credentials.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from falkordb import FalkorDB


@dataclass(frozen=True)
class Episode:
    uuid: str
    name: str
    group_id: str
    created_at: Any
    valid_at: Any


@dataclass(frozen=True)
class Saga:
    uuid: str
    name: str
    group_id: str
    first_episode_uuid: str | None
    last_episode_uuid: str | None
    created_at: Any


@dataclass
class Report:
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)


@dataclass(frozen=True)
class SemanticExpectation:
    episode_selector: str
    entity_terms: tuple[str, ...] = ()
    fact_terms: tuple[str, ...] = ()
    required: bool = True


def _norm(value: Any) -> str:
    return str(value or '').casefold().strip()


def _contains_term(values: Iterable[str], term: str) -> bool:
    needle = _norm(term)
    return any(needle in _norm(value) for value in values)


def validate_chain(
    saga: Saga,
    episodes: list[Episode],
    next_edges: list[tuple[str, str]],
    report: Report,
) -> list[Episode]:
    """Validate one directed predecessor->successor chain for a Saga."""
    by_uuid = {episode.uuid: episode for episode in episodes}
    episode_ids = set(by_uuid)

    if len(by_uuid) != len(episodes):
        report.fail(f'{saga.name}: duplicate episode UUID rows returned')
        return []

    if not episodes:
        if saga.first_episode_uuid is not None or saga.last_episode_uuid is not None:
            report.fail(f'{saga.name}: empty saga has first/last episode UUID state')
        return []

    if saga.first_episode_uuid not in episode_ids:
        report.fail(
            f'{saga.name}: first_episode_uuid={saga.first_episode_uuid!r} is not a HAS_EPISODE member'
        )
    if saga.last_episode_uuid not in episode_ids:
        report.fail(
            f'{saga.name}: last_episode_uuid={saga.last_episode_uuid!r} is not a HAS_EPISODE member'
        )

    outgoing: dict[str, list[str]] = {uuid: [] for uuid in episode_ids}
    incoming: dict[str, list[str]] = {uuid: [] for uuid in episode_ids}

    for source, target in next_edges:
        source_in = source in episode_ids
        target_in = target in episode_ids
        if source_in != target_in:
            report.fail(f'{saga.name}: NEXT_EPISODE crosses saga boundary: {source} -> {target}')
            continue
        if not source_in:
            continue
        outgoing[source].append(target)
        incoming[target].append(source)

    expected_edges = len(episodes) - 1
    internal_edges = sum(len(targets) for targets in outgoing.values())
    if internal_edges != expected_edges:
        report.fail(
            f'{saga.name}: expected {expected_edges} internal NEXT_EPISODE edges, found {internal_edges}'
        )

    for uuid, parents in incoming.items():
        expected = 0 if uuid == saga.first_episode_uuid else 1
        if len(parents) != expected:
            report.fail(
                f'{saga.name}: episode {by_uuid[uuid].name} has {len(parents)} predecessor(s), expected {expected}'
            )
    for uuid, children in outgoing.items():
        expected = 0 if uuid == saga.last_episode_uuid else 1
        if len(children) != expected:
            report.fail(
                f'{saga.name}: episode {by_uuid[uuid].name} has {len(children)} successor(s), expected {expected}'
            )

    ordered: list[Episode] = []
    current = saga.first_episode_uuid
    seen: set[str] = set()
    while current is not None and current in by_uuid:
        if current in seen:
            report.fail(f'{saga.name}: cycle detected at episode {by_uuid[current].name}')
            break
        seen.add(current)
        ordered.append(by_uuid[current])
        children = outgoing[current]
        current = children[0] if len(children) == 1 else None

    if len(seen) != len(episodes):
        missing = sorted(by_uuid[uuid].name for uuid in episode_ids - seen)
        report.fail(f'{saga.name}: chain traversal did not reach all episodes: {missing}')

    if ordered and ordered[-1].uuid != saga.last_episode_uuid:
        report.fail(
            f'{saga.name}: traversed last episode {ordered[-1].name} does not match stored last_episode_uuid'
        )

    for previous, current_episode in zip(ordered, ordered[1:], strict=False):
        if (
            previous.valid_at is not None
            and current_episode.valid_at is not None
            and str(previous.valid_at) > str(current_episode.valid_at)
        ):
            report.warn(
                f'{saga.name}: valid_at decreases: {previous.name} ({previous.valid_at}) -> '
                f'{current_episode.name} ({current_episode.valid_at})'
            )

    return ordered


def validate_semantics(
    saga: Saga,
    ordered: list[Episode],
    mentions: dict[str, list[str]],
    facts: dict[str, list[str]],
    expectations: list[SemanticExpectation],
    report: Report,
) -> None:
    by_name = {episode.name: episode for episode in ordered}
    by_uuid = {episode.uuid: episode for episode in ordered}

    for episode in ordered:
        entity_names = mentions.get(episode.uuid, [])
        episode_facts = facts.get(episode.uuid, [])
        if not entity_names:
            report.warn(
                f'{saga.name}: {episode.name} has zero MENTIONS entity edges '
                '(structurally valid; inspect content/expectations if entities were expected)'
            )
        report.note(f'{episode.name}: entities={len(entity_names)} facts={len(episode_facts)}')

    for expectation in expectations:
        episode = by_name.get(expectation.episode_selector) or by_uuid.get(expectation.episode_selector)
        if episode is None:
            message = (
                f'{saga.name}: semantic expectation references unknown episode '
                f'{expectation.episode_selector!r}'
            )
            (report.fail if expectation.required else report.warn)(message)
            continue

        entity_names = mentions.get(episode.uuid, [])
        episode_facts = facts.get(episode.uuid, [])
        for term in expectation.entity_terms:
            if not _contains_term(entity_names, term):
                message = (
                    f'{saga.name}: {episode.name} missing expected entity term {term!r}; '
                    f'actual={entity_names}'
                )
                (report.fail if expectation.required else report.warn)(message)
        for term in expectation.fact_terms:
            if not _contains_term(episode_facts, term):
                message = f'{saga.name}: {episode.name} missing expected fact term {term!r}'
                (report.fail if expectation.required else report.warn)(message)


def connect() -> FalkorDB:
    host = os.getenv('FALKORDB_HOST') or os.getenv('REDIS_HOST') or '127.0.0.1'
    port = int(os.getenv('FALKORDB_PORT') or os.getenv('REDIS_PORT') or '6379')
    username = os.getenv('FALKORDB_USER') or os.getenv('REDIS_USER') or None
    password = os.getenv('FALKORDB_PASSWORD') or os.getenv('REDIS_PASSWORD') or None
    kwargs: dict[str, Any] = {'host': host, 'port': port}
    if username:
        kwargs['username'] = username
    if password:
        kwargs['password'] = password
    return FalkorDB(**kwargs)


def rows(graph: Any, query: str, **params: Any) -> list[list[Any]]:
    return list(graph.query(query, params).result_set)


def load_sagas(graph: Any, group_id: str, saga_name: str | None) -> list[Saga]:
    if saga_name is None:
        query = """
            MATCH (s:Saga)
            WHERE s.group_id = $group_id
            RETURN s.uuid, s.name, s.group_id, s.first_episode_uuid, s.last_episode_uuid, s.created_at
            ORDER BY s.created_at, s.name
        """
        result = rows(graph, query, group_id=group_id)
    else:
        query = """
            MATCH (s:Saga)
            WHERE s.group_id = $group_id AND s.name = $saga_name
            RETURN s.uuid, s.name, s.group_id, s.first_episode_uuid, s.last_episode_uuid, s.created_at
            ORDER BY s.created_at, s.name
        """
        result = rows(graph, query, group_id=group_id, saga_name=saga_name)
    return [Saga(*row) for row in result]


def load_saga_episodes(graph: Any, saga_uuid: str) -> list[Episode]:
    result = rows(
        graph,
        """
        MATCH (s:Saga {uuid: $saga_uuid})-[:HAS_EPISODE]->(e:Episodic)
        RETURN e.uuid, e.name, e.group_id, e.created_at, e.valid_at
        """,
        saga_uuid=saga_uuid,
    )
    return [Episode(*row) for row in result]


def load_next_edges(graph: Any) -> list[tuple[str, str]]:
    return [
        (str(row[0]), str(row[1]))
        for row in rows(
            graph,
            """
            MATCH (a:Episodic)-[:NEXT_EPISODE]->(b:Episodic)
            RETURN a.uuid, b.uuid
            """,
        )
    ]


def load_mentions(graph: Any, episode_ids: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {uuid: [] for uuid in episode_ids}
    if not episode_ids:
        return result
    for episode_uuid, entity_name in rows(
        graph,
        """
        MATCH (e:Episodic)-[:MENTIONS]->(n:Entity)
        WHERE e.uuid IN $episode_ids
        RETURN e.uuid, n.name
        ORDER BY e.uuid, n.name
        """,
        episode_ids=episode_ids,
    ):
        result.setdefault(str(episode_uuid), []).append(str(entity_name))
    return result


def load_facts(graph: Any, episode_ids: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {uuid: [] for uuid in episode_ids}
    if not episode_ids:
        return result
    for episode_uuid, fact in rows(
        graph,
        """
        MATCH (:Entity)-[r:RELATES_TO]->(:Entity)
        UNWIND coalesce(r.episodes, []) AS episode_uuid
        WITH episode_uuid, r.fact AS fact
        WHERE episode_uuid IN $episode_ids
        RETURN episode_uuid, fact
        ORDER BY episode_uuid
        """,
        episode_ids=episode_ids,
    ):
        result.setdefault(str(episode_uuid), []).append(str(fact))
    return result


def parse_terms(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(',') if part.strip())


def collect_expectations_interactive() -> list[SemanticExpectation]:
    expectations: list[SemanticExpectation] = []
    answer = input('Добавить семантические ожидания по эпизодам? [y/N]: ').strip().casefold()
    if answer not in {'y', 'yes', 'д', 'да'}:
        return expectations

    print('Пустой selector завершает ввод. Selector = точное имя episode или UUID.')
    while True:
        selector = input('episode selector: ').strip()
        if not selector:
            break
        entities = parse_terms(input('ожидаемые entity terms через запятую (можно пусто): '))
        facts = parse_terms(input('ожидаемые fact terms через запятую (можно пусто): '))
        mode = input('если не найдено: fail или warn? [fail]: ').strip().casefold()
        expectations.append(
            SemanticExpectation(
                episode_selector=selector,
                entity_terms=entities,
                fact_terms=facts,
                required=mode not in {'warn', 'w', 'предупреждение'},
            )
        )
    return expectations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', help='Graphiti group_id / physical Falkor graph')
    parser.add_argument('--saga', help='exact Saga.name; omit to validate every Saga in group')
    parser.add_argument('--expected-count', type=int, help='expected episode count for selected Saga')
    parser.add_argument('--non-interactive', action='store_true', help='do not prompt for missing inputs')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    group_id = (args.group or '').strip()
    if not group_id and not args.non_interactive:
        group_id = input('Graphiti group_id / Falkor graph [main]: ').strip() or 'main'
    if not group_id:
        print('ERROR: --group is required in non-interactive mode', file=sys.stderr)
        return 2

    saga_name = args.saga.strip() if args.saga else None
    if not args.non_interactive and args.saga is None:
        saga_name = input('Saga name (пусто = проверить все Saga структурно): ').strip() or None

    if args.expected_count is not None and saga_name is None:
        print('ERROR: --expected-count requires an exact --saga', file=sys.stderr)
        return 2

    expectations: list[SemanticExpectation] = []
    if not args.non_interactive and saga_name is not None:
        expectations = collect_expectations_interactive()

    try:
        db = connect()
        graph = db.select_graph(group_id)
        sagas = load_sagas(graph, group_id, saga_name)
        if not sagas:
            print(f'FAIL: no Saga nodes found for group={group_id!r} saga={saga_name!r}')
            return 1
        if saga_name is not None and len(sagas) != 1:
            print(f'FAIL: expected one exact Saga named {saga_name!r}, found {len(sagas)}')
            return 1

        all_next = load_next_edges(graph)
        overall = Report()

        print(f'group={group_id} sagas={len(sagas)} read_only=true')
        for saga in sagas:
            report = Report()
            if saga.group_id != group_id:
                report.fail(f'{saga.name}: Saga group_id={saga.group_id!r}, expected {group_id!r}')

            episodes = load_saga_episodes(graph, saga.uuid)
            if args.expected_count is not None and len(episodes) != args.expected_count:
                report.fail(
                    f'{saga.name}: expected {args.expected_count} episodes, found {len(episodes)}'
                )
            for episode in episodes:
                if episode.group_id != group_id:
                    report.fail(
                        f'{saga.name}: episode {episode.name} group_id={episode.group_id!r}, expected {group_id!r}'
                    )

            ordered = validate_chain(saga, episodes, all_next, report)
            episode_ids = [episode.uuid for episode in episodes]
            mentions = load_mentions(graph, episode_ids)
            facts = load_facts(graph, episode_ids)
            validate_semantics(saga, ordered, mentions, facts, expectations, report)

            status = 'FAIL' if report.failures else ('WARN' if report.warnings else 'PASS')
            print(f'\n[{status}] saga={saga.name!r} uuid={saga.uuid} episodes={len(episodes)}')
            if ordered:
                print('  chain: ' + ' -> '.join(episode.name for episode in ordered))
            for note in report.notes:
                print(f'  INFO: {note}')
            for warning in report.warnings:
                print(f'  WARN: {warning}')
            for failure in report.failures:
                print(f'  FAIL: {failure}')

            overall.failures.extend(report.failures)
            overall.warnings.extend(report.warnings)

        print(
            f'\nSUMMARY: failures={len(overall.failures)} warnings={len(overall.warnings)} '
            f'sagas={len(sagas)}'
        )
        return 1 if overall.failures else 0
    except Exception as exc:
        print(f'ERROR: validator could not read FalkorDB: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
