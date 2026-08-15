"""piqnyx Graphiti compatibility extensions.

The fork preserves upstream behavior for existing episode UUIDs and additionally
allows a caller to reserve the UUID of a brand-new episode before asynchronous MCP
processing starts.
"""

from datetime import datetime
from time import time
from typing import Any

from pydantic import BaseModel

from graphiti_core.errors import NodeNotFoundError
from graphiti_core.graphiti import AddEpisodeResults, Graphiti as _Graphiti
from graphiti_core.helpers import semaphore_gather, validate_excluded_entity_types
from graphiti_core.nodes import EpisodeType, EpisodicNode, SagaNode
from graphiti_core.search.search_utils import RELEVANT_SCHEMA_LIMIT
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.community_operations import update_community
from graphiti_core.utils.maintenance.node_operations import (
    extract_attributes_from_nodes,
    extract_nodes,
    resolve_extracted_nodes,
)
from graphiti_core.utils.ontology_utils.entity_types_utils import validate_entity_types


class Graphiti(_Graphiti):
    """Graphiti with backwards-compatible caller-assigned episode UUID creation."""

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
        """Return an existing UUID target or construct a new unsaved episode.

        Upstream semantics are retained when ``uuid`` already exists. When it does
        not exist, the caller-provided UUID becomes the UUID of a newly constructed
        episode. Construction happens only after previous-episode retrieval, so a
        new episode cannot accidentally appear in its own automatic context window.
        """
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
        """Process an episode, creating a new caller-assigned UUID when needed."""
        start = time()
        now = utc_now()

        validate_entity_types(entity_types)
        validate_excluded_entity_types(excluded_entity_types, entity_types)

        # Keep the request-scoped FalkorDB isolation carried by this fork.
        group_id, driver, clients = self._resolve_request_scope(group_id)

        with self.tracer.start_span('add_episode') as span:
            try:
                # Preserve upstream ordering: resolve previous context before the
                # current episode exists, so automatic retrieval cannot see itself.
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

                (
                    resolved_edges,
                    invalidated_edges,
                    new_edges,
                ) = await self._extract_and_resolve_edges(
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
