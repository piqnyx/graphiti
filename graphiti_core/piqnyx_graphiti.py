"""piqnyx Graphiti ingestion extensions.

The fork keeps request-scoped Falkor isolation and caller-reserved episode UUIDs,
but treats graph mutation as a commit boundary. Extraction may take minutes and may
fail freely; no episode/entity/fact/Saga mutation is allowed to leak into Falkor
until the complete result is ready. For FalkorDB the final mutation is emitted as
one Cypher query, because its Python driver session is not a rollback-capable
transaction and the old multi-query save path could leave half an episode behind.
"""

from datetime import datetime
from time import time
from typing import Any

from pydantic import BaseModel

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge, HasEpisodeEdge, NextEpisodeEdge
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.graphiti import AddEpisodeResults, Graphiti as _Graphiti
from graphiti_core.helpers import (
    semaphore_gather,
    validate_excluded_entity_types,
    validate_node_labels,
)
from graphiti_core.nodes import EpisodeType, EpisodicNode, SagaNode
from graphiti_core.search.search_utils import RELEVANT_SCHEMA_LIMIT
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.community_operations import update_community
from graphiti_core.utils.maintenance.edge_operations import build_episodic_edges
from graphiti_core.utils.maintenance.node_operations import (
    extract_attributes_from_nodes,
    extract_nodes,
    resolve_extracted_nodes,
)
from graphiti_core.utils.ontology_utils.entity_types_utils import validate_entity_types


class Graphiti(_Graphiti):
    """Graphiti with caller UUIDs and crash-safe Falkor episode commits."""

    @staticmethod
    async def _get_or_create_caller_episode(
        *,
        driver: Any,
        uuid: str | None,
        name: str,
        group_id: str,
        source: EpisodeType,
        episode_body: str,
        source_description: str,
        created_at: datetime,
        reference_time: datetime,
    ) -> EpisodicNode:
        """Return an existing UUID target or construct a new unsaved episode."""
        if uuid is not None:
            try:
                return await EpisodicNode.get_by_uuid(driver, uuid)
            except NodeNotFoundError:
                pass

        episode_kwargs: dict[str, Any] = {
            'name': name,
            'group_id': group_id,
            'labels': [],
            'source': source,
            'content': episode_body,
            'source_description': source_description,
            'created_at': created_at,
            'valid_at': reference_time,
        }
        if uuid is not None:
            episode_kwargs['uuid'] = uuid

        return EpisodicNode(**episode_kwargs)

    async def _existing_committed_caller_episode(
        self,
        *,
        driver: Any,
        uuid: str | None,
        name: str,
        group_id: str,
        saga: str | SagaNode | None,
    ) -> EpisodicNode | None:
        """Recognize an idempotent replay, and fail closed on partial/divergent state.

        The caller-side spool may repeat a UUID after losing an HTTP response. If
        that exact episode is already the committed tail of the requested Saga,
        replay is a no-op. If the UUID exists but is detached or belongs somewhere
        else, silently running extraction on top of it would convert corruption
        into more corruption, so we stop instead.
        """
        if uuid is None:
            return None

        try:
            episode = await EpisodicNode.get_by_uuid(driver, uuid)
        except NodeNotFoundError:
            episode = None

        if saga is None:
            # Preserve upstream-style behavior for non-Saga callers. The strict
            # idempotency contract is for ordered conversation ingestion.
            return None

        saga_name = saga.name if isinstance(saga, SagaNode) else saga
        duplicate_name_records, _, _ = await driver.execute_query(
            """
            MATCH (s:Saga {name: $saga_name, group_id: $group_id})-[:HAS_EPISODE]->(e:Episodic {name: $episode_name})
            WHERE e.uuid <> $episode_uuid
            RETURN e.uuid AS uuid
            LIMIT 1
            """,
            saga_name=saga_name,
            group_id=group_id,
            episode_name=name,
            episode_uuid=uuid,
            routing_='r',
        )
        if duplicate_name_records:
            raise RuntimeError(
                f'Saga {group_id}/{saga_name} already contains episode name {name} '
                f'under different UUID {duplicate_name_records[0]["uuid"]}; refusing to fork chronology'
            )

        if episode is None:
            return None

        records, _, _ = await driver.execute_query(
            """
            MATCH (s:Saga {name: $saga_name, group_id: $group_id})-[:HAS_EPISODE]->(e:Episodic {uuid: $episode_uuid})
            RETURN s.first_episode_uuid AS first_episode_uuid,
                   s.last_episode_uuid AS last_episode_uuid
            LIMIT 1
            """,
            saga_name=saga_name,
            group_id=group_id,
            episode_uuid=uuid,
            routing_='r',
        )
        if records and records[0].get('last_episode_uuid') == uuid:
            return episode

        raise RuntimeError(
            f'Episode UUID {uuid} already exists but is not the committed tail of '
            f'Saga {group_id}/{saga_name}; refusing to process a partial or divergent episode'
        )

    async def _resolve_saga_without_writing(
        self,
        saga: str | SagaNode,
        group_id: str,
        created_at: datetime,
        driver: Any,
    ) -> SagaNode:
        if isinstance(saga, SagaNode):
            return saga

        records, _, _ = await driver.execute_query(
            """
            MATCH (s:Saga {name: $name, group_id: $group_id})
            RETURN s.uuid AS uuid
            LIMIT 1
            """,
            name=saga,
            group_id=group_id,
            routing_='r',
        )
        if records:
            return await SagaNode.get_by_uuid(driver, records[0]['uuid'])
        return SagaNode(name=saga, group_id=group_id, created_at=created_at)

    async def _process_episode_data(
        self,
        episode: EpisodicNode | list[EpisodicNode],
        nodes: list[Any],
        entity_edges: list[EntityEdge],
        now: datetime,
        group_id: str,
        saga: str | SagaNode | None = None,
        saga_previous_episode_uuid: str | None = None,
        node_episode_index_map: dict[str, list[int]] | None = None,
        clients: Any = None,
    ):
        """Use one Falkor query as the commit boundary for an ordered episode.

        Upstream's helper calls ``execute_write``, but FalkorDriverSession implements
        that as several immediately-visible graph queries rather than a rollback
        transaction. A kill between those queries used to leave an Episodic node,
        some entities/facts, and no Saga edges. Retrying then resolved against that
        debris and could produce a different second half.

        For Saga-backed Falkor ingestion we prepare embeddings first, then write the
        episode, entities, facts, MENTIONS, NEXT_EPISODE, HAS_EPISODE and Saga tail in
        one Cypher command. One Falkor query is the only mutation boundary.
        """
        clients = clients or self.clients
        driver = clients.driver
        if driver.provider != GraphProvider.FALKORDB or saga is None:
            return await super()._process_episode_data(
                episode,
                nodes,
                entity_edges,
                now,
                group_id,
                saga,
                saga_previous_episode_uuid,
                node_episode_index_map,
                clients=clients,
            )

        episodes = episode if isinstance(episode, list) else [episode]
        if len(episodes) != 1:
            # The OpenClaw path is one durable batch per call. Keep bulk behavior
            # upstream rather than pretending this single-episode transaction covers it.
            return await super()._process_episode_data(
                episode,
                nodes,
                entity_edges,
                now,
                group_id,
                saga,
                saga_previous_episode_uuid,
                node_episode_index_map,
                clients=clients,
            )

        primary_episode = episodes[0]
        episodic_edges = build_episodic_edges(
            nodes,
            [primary_episode.uuid],
            now,
            node_episode_index_map,
        )
        primary_episode.entity_edges = [edge.uuid for edge in entity_edges]
        if not self.store_raw_episode_content:
            primary_episode.content = ''

        # No remote embedding call may occur after the database mutation starts.
        node_embedding_tasks = [
            node.generate_name_embedding(clients.embedder)
            for node in nodes
            if node.name_embedding is None
        ]
        if node_embedding_tasks:
            await semaphore_gather(*node_embedding_tasks, max_coroutines=self.max_coroutines)
        edge_embedding_tasks = [
            edge.generate_embedding(clients.embedder)
            for edge in entity_edges
            if edge.fact_embedding is None
        ]
        if edge_embedding_tasks:
            await semaphore_gather(*edge_embedding_tasks, max_coroutines=self.max_coroutines)

        saga_node = await self._resolve_saga_without_writing(saga, group_id, now, driver)
        previous_episode_uuid = (
            saga_previous_episode_uuid
            if saga_previous_episode_uuid is not None
            else saga_node.last_episode_uuid
        )

        # A non-first commit starts with structural prerequisites. If the durable
        # caller supplies a stale/missing predecessor, the query performs zero
        # mutation instead of quietly creating a second chain head or fork.
        params: dict[str, Any] = {
            'group_id': group_id,
            'saga_uuid': saga_node.uuid,
            'saga_name': saga_node.name,
            'episode_uuid': primary_episode.uuid,
        }
        if previous_episode_uuid is None:
            prefix = """
            MERGE (saga:Saga {uuid: $saga_uuid})
            ON CREATE SET saga.name = $saga_name,
                          saga.group_id = $group_id,
                          saga.created_at = $saga_created_at,
                          saga.summary = $saga_summary,
                          saga.first_episode_uuid = NULL,
                          saga.last_episode_uuid = NULL,
                          saga.last_summarized_at = $saga_last_summarized_at,
                          saga.last_summarized_episode_valid_at = $saga_last_summarized_episode_valid_at
            WITH saga
            OPTIONAL MATCH (saga)-[:HAS_EPISODE]->(existing:Episodic)
            WITH saga, count(existing) AS existing_count
            WHERE existing_count = 0
              AND saga.group_id = $group_id
              AND saga.name = $saga_name
              AND saga.first_episode_uuid IS NULL
              AND saga.last_episode_uuid IS NULL
            WITH saga, NULL AS previous
            """
        else:
            params['previous_episode_uuid'] = previous_episode_uuid
            prefix = """
            MATCH (saga:Saga {uuid: $saga_uuid, group_id: $group_id})
            WHERE saga.name = $saga_name
              AND saga.last_episode_uuid = $previous_episode_uuid
            MATCH (previous:Episodic {uuid: $previous_episode_uuid, group_id: $group_id})
            MATCH (saga)-[:HAS_EPISODE]->(previous)
            OPTIONAL MATCH (previous)-[:NEXT_EPISODE]->(existing_successor:Episodic)
            WITH saga, previous, collect(existing_successor.uuid) AS successor_uuids
            WHERE size(successor_uuids) = 0
            WITH saga, previous
            """

        saga_data = {
            'uuid': saga_node.uuid,
            'name': saga_node.name,
            'group_id': group_id,
            'created_at': saga_node.created_at,
            'summary': saga_node.summary,
            'first_episode_uuid': saga_node.first_episode_uuid or primary_episode.uuid,
            'last_episode_uuid': primary_episode.uuid,
            'last_summarized_at': saga_node.last_summarized_at,
            'last_summarized_episode_valid_at': saga_node.last_summarized_episode_valid_at,
        }
        params.update(
            {
                'saga_created_at': saga_node.created_at,
                'saga_summary': saga_node.summary,
                'saga_last_summarized_at': saga_node.last_summarized_at,
                'saga_last_summarized_episode_valid_at': saga_node.last_summarized_episode_valid_at,
                'saga_data': saga_data,
            }
        )

        episode_data = {
            'uuid': primary_episode.uuid,
            'name': primary_episode.name,
            'group_id': primary_episode.group_id,
            'source_description': primary_episode.source_description,
            'source': primary_episode.source.value,
            'content': primary_episode.content,
            'entity_edges': primary_episode.entity_edges,
            'created_at': primary_episode.created_at,
            'valid_at': primary_episode.valid_at,
        }
        params['episodes'] = [episode_data]

        node_groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for node in nodes:
            labels = sorted(set([*node.labels, 'Entity']))
            validate_node_labels(labels)
            node_data: dict[str, Any] = {
                'uuid': node.uuid,
                'name': node.name,
                'group_id': node.group_id,
                'summary': node.summary,
                'created_at': node.created_at,
                'name_embedding': node.name_embedding,
                'labels': labels,
            }
            for key, value in (node.attributes or {}).items():
                if key not in node_data:
                    node_data[key] = value
            node_groups.setdefault(tuple(labels), []).append(node_data)

        prepared_edges: list[dict[str, Any]] = []
        for edge in entity_edges:
            edge_data: dict[str, Any] = {
                'uuid': edge.uuid,
                'source_node_uuid': edge.source_node_uuid,
                'target_node_uuid': edge.target_node_uuid,
                'name': edge.name,
                'fact': edge.fact,
                'group_id': edge.group_id,
                'episodes': edge.episodes,
                'created_at': edge.created_at,
                'expired_at': edge.expired_at,
                'valid_at': edge.valid_at,
                'invalid_at': edge.invalid_at,
                'reference_time': edge.reference_time,
                'fact_embedding': edge.fact_embedding,
            }
            for key, value in (edge.attributes or {}).items():
                if key not in edge_data:
                    edge_data[key] = value
            prepared_edges.append(edge_data)
        params['entity_edges'] = prepared_edges
        params['episodic_edges'] = [edge.model_dump() for edge in episodic_edges]

        query_parts = [prefix]
        query_parts.append(
            """
            UNWIND $episodes AS episode
            MERGE (ep:Episodic {uuid: episode.uuid})
            SET ep = episode
            WITH saga, previous, ep
            """
        )

        for index, (labels, group_nodes) in enumerate(node_groups.items()):
            key = f'entity_nodes_{index}'
            params[key] = group_nodes
            static_labels = ':'.join(labels)
            query_parts.append(
                f"""
                UNWIND ${key} AS node
                MERGE (n:Entity {{uuid: node.uuid}})
                SET n:{static_labels}
                SET n = node
                SET n.name_embedding = vecf32(node.name_embedding)
                WITH saga, previous, ep, count(n) AS _nodes_{index}
                """
            )

        if prepared_edges:
            query_parts.append(
                """
                UNWIND $entity_edges AS edge
                MATCH (source:Entity {uuid: edge.source_node_uuid})
                MATCH (target:Entity {uuid: edge.target_node_uuid})
                MERGE (source)-[fact:RELATES_TO {uuid: edge.uuid}]->(target)
                SET fact = edge
                SET fact.fact_embedding = vecf32(edge.fact_embedding)
                WITH saga, previous, ep, count(fact) AS _facts
                """
            )

        if episodic_edges:
            query_parts.append(
                """
                UNWIND $episodic_edges AS edge
                MATCH (mention_episode:Episodic {uuid: edge.source_node_uuid})
                MATCH (mentioned:Entity {uuid: edge.target_node_uuid})
                MERGE (mention_episode)-[mention:MENTIONS {uuid: edge.uuid}]->(mentioned)
                SET mention.group_id = edge.group_id,
                    mention.created_at = edge.created_at
                WITH saga, previous, ep, count(mention) AS _mentions
                """
            )

        has_episode_edge = HasEpisodeEdge(
            source_node_uuid=saga_node.uuid,
            target_node_uuid=primary_episode.uuid,
            group_id=group_id,
            created_at=now,
        )
        params.update(
            {
                'has_episode_uuid': has_episode_edge.uuid,
                'has_episode_created_at': has_episode_edge.created_at,
            }
        )
        query_parts.append(
            """
            MERGE (saga)-[has_episode:HAS_EPISODE {uuid: $has_episode_uuid}]->(ep)
            SET has_episode.group_id = $group_id,
                has_episode.created_at = $has_episode_created_at
            """
        )

        if previous_episode_uuid is not None:
            next_episode_edge = NextEpisodeEdge(
                source_node_uuid=previous_episode_uuid,
                target_node_uuid=primary_episode.uuid,
                group_id=group_id,
                created_at=now,
            )
            params.update(
                {
                    'next_episode_uuid': next_episode_edge.uuid,
                    'next_episode_created_at': next_episode_edge.created_at,
                }
            )
            query_parts.append(
                """
                MERGE (previous)-[next_episode:NEXT_EPISODE {uuid: $next_episode_uuid}]->(ep)
                SET next_episode.group_id = $group_id,
                    next_episode.created_at = $next_episode_created_at
                """
            )

        query_parts.append(
            """
            SET saga = $saga_data
            RETURN ep.uuid AS committed_uuid,
                   saga.first_episode_uuid AS first_episode_uuid,
                   saga.last_episode_uuid AS last_episode_uuid
            """
        )

        records, _, _ = await driver.execute_query(''.join(query_parts), **params)
        if not records or records[0].get('committed_uuid') != primary_episode.uuid:
            raise RuntimeError(
                f'Atomic Falkor commit precondition failed for {group_id}/{saga_node.name} '
                f'episode {primary_episode.name} ({primary_episode.uuid}); chronology was not mutated'
            )

        saga_node.first_episode_uuid = saga_data['first_episode_uuid']
        saga_node.last_episode_uuid = primary_episode.uuid
        return episodic_edges, primary_episode

    async def add_episode(
        self,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
        source: EpisodeType = EpisodeType.message,
        group_id: str | None = None,
        uuid: str | None = None,
        update_communities: bool = False,
        entity_types: dict[str, type[BaseModel]] | None = None,
        excluded_entity_types: list[str] | None = None,
        previous_episode_uuids: list[str] | None = None,
        edge_types: dict[str, type[BaseModel]] | None = None,
        edge_type_map: dict[tuple[str, str], list[str]] | None = None,
        custom_extraction_instructions: str | None = None,
        saga: str | SagaNode | None = None,
        saga_previous_episode_uuid: str | None = None,
    ) -> AddEpisodeResults:
        """Process one episode with caller UUID idempotency and Falkor commit safety."""
        start = time()
        now = utc_now()

        validate_entity_types(entity_types)
        validate_excluded_entity_types(excluded_entity_types, entity_types)
        group_id, driver, clients = self._resolve_request_scope(group_id)

        with self.tracer.start_span('add_episode') as span:
            try:
                committed = await self._existing_committed_caller_episode(
                    driver=driver,
                    uuid=uuid,
                    name=name,
                    group_id=group_id,
                    saga=saga,
                )
                if committed is not None:
                    return AddEpisodeResults(
                        episode=committed,
                        episodic_edges=[],
                        nodes=[],
                        edges=[],
                        communities=[],
                        community_edges=[],
                    )

                previous_episodes = (
                    await self.retrieve_episodes(
                        reference_time,
                        last_n=RELEVANT_SCHEMA_LIMIT,
                        group_ids=[group_id],
                        source=source,
                        driver=driver,
                    )
                    if previous_episode_uuids is None
                    else await EpisodicNode.get_by_uuids(driver, previous_episode_uuids)
                )

                episode = await self._get_or_create_caller_episode(
                    driver=driver,
                    uuid=uuid,
                    name=name,
                    group_id=group_id,
                    source=source,
                    episode_body=episode_body,
                    source_description=source_description,
                    created_at=now,
                    reference_time=reference_time,
                )

                edge_type_map_default = (
                    {('Entity', 'Entity'): list(edge_types.keys())}
                    if edge_types is not None
                    else {('Entity', 'Entity'): []}
                )

                extracted_nodes, node_episode_index_map = await extract_nodes(
                    clients,
                    episode,
                    previous_episodes,
                    entity_types,
                    excluded_entity_types,
                    custom_extraction_instructions,
                )

                nodes, uuid_map, _ = await resolve_extracted_nodes(
                    clients,
                    extracted_nodes,
                    episode,
                    previous_episodes,
                    entity_types,
                )

                resolved_edges, invalidated_edges, new_edges = await self._extract_and_resolve_edges(
                    episode,
                    extracted_nodes,
                    previous_episodes,
                    edge_type_map or edge_type_map_default,
                    group_id,
                    edge_types,
                    nodes,
                    uuid_map,
                    custom_extraction_instructions,
                    clients=clients,
                )
                entity_edges = resolved_edges + invalidated_edges

                hydrated_nodes = await extract_attributes_from_nodes(
                    clients,
                    nodes,
                    episode,
                    previous_episodes,
                    entity_types,
                    edges=new_edges,
                )

                episodic_edges, episode = await self._process_episode_data(
                    episode,
                    hydrated_nodes,
                    entity_edges,
                    now,
                    group_id,
                    saga,
                    saga_previous_episode_uuid,
                    node_episode_index_map,
                    clients=clients,
                )

                communities = []
                community_edges = []
                if update_communities:
                    communities, community_edges = await semaphore_gather(
                        *[
                            update_community(driver, clients.llm_client, clients.embedder, node)
                            for node in nodes
                        ],
                        max_coroutines=self.max_coroutines,
                    )

                end = time()
                span.add_attributes(
                    {
                        'episode.uuid': episode.uuid,
                        'episode.source': source.value,
                        'episode.reference_time': reference_time.isoformat(),
                        'group_id': group_id,
                        'node.count': len(hydrated_nodes),
                        'edge.count': len(entity_edges),
                        'edge.invalidated_count': len(invalidated_edges),
                        'previous_episodes.count': len(previous_episodes),
                        'entity_types.count': len(entity_types) if entity_types else 0,
                        'edge_types.count': len(edge_types) if edge_types else 0,
                        'update_communities': update_communities,
                        'communities.count': len(communities) if update_communities else 0,
                        'duration_ms': (end - start) * 1000,
                    }
                )

                return AddEpisodeResults(
                    episode=episode,
                    episodic_edges=episodic_edges,
                    nodes=hydrated_nodes,
                    edges=entity_edges,
                    communities=communities,
                    community_edges=community_edges,
                )
            except Exception as e:
                span.set_status('error', str(e))
                span.record_exception(e)
                raise
