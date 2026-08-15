# piqnyx caller-visible episode UUIDs

The MCP server reserves an episode UUID before asynchronous queueing so callers can
chain saga episodes without polling `get_episodes`.

Behavior:

- successful `add_memory` responses include an optional structured `uuid` field;
- caller-provided UUIDs that already exist keep upstream re-processing semantics;
- caller-provided UUIDs that do not yet exist are materialized as the new episode UUID;
- failed enqueue/validation responses do not claim a UUID was accepted.

The production MCP entrypoint installs this compatibility layer before FastMCP tool
registration. The underlying Graphiti subclass is exported from `graphiti_core` so
the queue and other public top-level callers use the same UUID semantics.
