"""piqnyx Saga compatibility fix.

Existing sagas must be reloaded through the canonical SagaNode loader so all
persisted state (including first/last episode UUIDs and summarization watermarks)
survives subsequent add_episode calls.
"""

from datetime import datetime

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.nodes import SagaNode
from graphiti_core.piqnyx_graphiti import Graphiti as _Graphiti


class Graphiti(_Graphiti):
    """Graphiti with piqnyx episode UUID and Saga state fixes."""

    async def _get_or_create_saga(
        self,
        saga_name: str,
        group_id: str,
        created_at: datetime,
        driver: GraphDriver | None = None,
    ) -> SagaNode:
        """Load an existing saga without dropping persisted state, or create it."""
        driver = driver or self.driver

        records, _, _ = await driver.execute_query(
            """
            MATCH (s:Saga {name: $name, group_id: $group_id})
            RETURN s.uuid AS uuid
            """,
            name=saga_name,
            group_id=group_id,
            routing_='r',
        )

        if records:
            return await SagaNode.get_by_uuid(driver, records[0]['uuid'])

        saga = SagaNode(name=saga_name, group_id=group_id, created_at=created_at)
        await saga.save(driver)
        return saga
